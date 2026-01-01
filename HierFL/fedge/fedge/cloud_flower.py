# cloud_flower.py

import os
import time
import pickle
import toml
import threading
import sys
import signal
import warnings
import logging
import torch
import math

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, NDArrays, Parameters, FitRes, parameters_to_ndarrays

from fedge.utils import fs
from fedge.task import Net, load_data, test, set_weights
from fedge.stats import _mean_std_ci
import csv

# ──────────────────────────────────────────────────────────────────────────────
#  Read **all** of our hierarchy config from pyproject.toml
# ──────────────────────────────────────────────────────────────────────────────

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
cfg = toml.load(project_root / "pyproject.toml")
hier = cfg["tool"]["flwr"]["hierarchy"]

NUM_SERVERS = hier["num_servers"]
GLOBAL_ROUNDS = hier["global_rounds"]
SERVER_ROUNDS_PER_GLOBAL = hier["server_rounds_per_global"]
CLOUD_PORT = hier["cloud_port"]

# Check if we should use the new directory structure
use_new_dir_structure = os.environ.get("USE_NEW_DIR_STRUCTURE", "0") == "1"

# Get the current global round from environment (0-indexed in code)
current_global_round = int(os.environ.get("GLOBAL_ROUND", "0"))

# Use 0-indexed naming consistently for directory structure
dir_round = current_global_round

# Create necessary directories
if use_new_dir_structure:
    # Create signals directory and global round directory
    signals_dir = fs.get_signals_dir(project_root)
    signals_dir.mkdir(exist_ok=True, parents=True)
    
    global_round_dir = fs.get_global_round_dir(project_root, dir_round)
    global_round_dir.mkdir(exist_ok=True, parents=True)
    
    # Paths for signals & output using new structure
    def get_global_round_signal_path(round_num: int) -> Path:
            # Note: round_num is 0-indexed for directory paths
        return fs.get_global_round_dir(project_root, round_num) / "complete.signal"

    def get_global_model_path(round_num: int) -> Path:
            # Note: round_num is 0-indexed for directory paths
        return fs.get_global_round_dir(project_root, round_num) / "model.pkl"

    # Cloud signals are in the signals directory
    start_signal = signals_dir / "cloud_started.signal"
    complete_signal = signals_dir / "cloud_complete.signal"
    
    print(f"[Cloud Server] Using consistent 0-indexed round directories")
else:
    # Use old directory structure as fallback
    # Create directory for this global round
    try:
        global_dir = Path(project_root) / "rounds" / f"round_{current_global_round}" / "global"
        global_dir.mkdir(exist_ok=True, parents=True)
        
        # Define signal paths using old structure
        def get_global_round_signal_path(round_num: int) -> Path:
            old_global_dir = Path(project_root) / "rounds" / f"round_{round_num}" / "global"
            return old_global_dir / f"global_round_{round_num}_complete.signal"

        def get_global_model_path(round_num: int) -> Path:
            old_global_dir = Path(project_root) / "rounds" / f"round_{round_num}" / "global"
            return old_global_dir / f"global_model_round_{round_num}.pkl"

        start_signal = global_dir / "cloud_started.signal"
        complete_signal = global_dir / "cloud_complete.signal" 
        
        print(f"[Cloud Server] Using old directory structure with 0-indexed round directories")
    except Exception as e:
        print(f"[Cloud Server] ERROR: Failed to create directories: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
#  Helper to create a “signal file” with a timestamp
# ──────────────────────────────────────────────────────────────────────────────

def create_signal_file(file_path: Path, message: str) -> bool:
    try:
        with open(file_path, "w") as f:
            f.write(str(time.time()))
        logging.info(f"[Cloud Server] {message}: {file_path}")
        return True
    except Exception as e:
        logging.error(f"[Cloud Server] Could not create {file_path}: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
#  Logging / Warnings suppression
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Drop “DEPRECATED FEATURE” on stdout/stderr
class _DropDeprecated:
    def __init__(self, out): self._out = out
    def write(self, txt):
        if "DEPRECATED FEATURE" not in txt:
            self._out.write(txt)
    def flush(self): self._out.flush()

sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)

# Statistical helpers now imported from shared stats module

# ──────────────────────────────────────────────────────────────────────────────
#  Signal‐handler: If someone hits Ctrl+C, write a final cloud_complete.signal
# ──────────────────────────────────────────────────────────────────────────────

def handle_signal(sig, frame):
    cloud_id = os.environ.get("SERVER_ID", "cloud")
    logger.info(f"[{cloud_id}] Received signal {sig}, saving final completion signal...")
    create_signal_file(complete_signal, "Created completion signal from SIGTERM handler")

    # We might save a last EMERGENCY model (if desired), but skip for brevity
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ──────────────────────────────────────────────────────────────────────────────
#  Create “cloud_started.signal” right away
# ──────────────────────────────────────────────────────────────────────────────

create_signal_file(start_signal, f"Created start signal for round {current_global_round}")

# ──────────────────────────────────────────────────────────────────────────────
#  Weighted‐average for leaf accuracies
# ──────────────────────────────────────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n * m.get("accuracy", 0) for n, m in metrics)
    count = sum(n for n, _ in metrics)
    if count == 0:
        return {"accuracy": 0.0}
    return {"accuracy": total / count}

# ──────────────────────────────────────────────────────────────────────────────
#  Subclass FedAvg to implement our “global‐round” logic
# ──────────────────────────────────────────────────────────────────────────────

class CloudFedAvg(FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_SERVERS,
            min_evaluate_clients=NUM_SERVERS,
            min_available_clients=NUM_SERVERS,
            #fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        self.cloud_id = os.environ.get("SERVER_ID", "cloud")
        self.current_global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
        self.round_start_time = None
        self.communication_metrics = []
        self.latest_parameters: Optional[Parameters] = None
        # Track previous round metrics for convergence calculation - load from file if exists
        self._load_previous_metrics()
        logger.info(f"[Cloud Server] Initialized for GLOBAL ROUND {self.current_global_round}")

    def _load_previous_metrics(self):
        """Load previous round metrics from standardized metrics/cloud/rounds.csv."""
        self.prev_central_loss = None
        self.prev_central_acc = None
        self.prev_avg_loss_across_servers = None
        self.prev_avg_acc_across_servers = None
        if self.current_global_round > 0:
            try:
                import pandas as pd
                rounds_csv = Path().resolve() / "metrics" / "cloud" / "rounds.csv"
                if rounds_csv.exists():
                    df = pd.read_csv(rounds_csv)
                    prev_round = self.current_global_round - 1
                    prev_row = df[df['global_round'] == prev_round]
                    if not prev_row.empty:
                        self.prev_central_loss = prev_row['global_test_loss_centralized'].iloc[0]
                        self.prev_central_acc = prev_row['global_test_accuracy_centralized'].iloc[0]
                        self.prev_avg_loss_across_servers = prev_row['avg_test_loss_across_servers'].iloc[0]
                        self.prev_avg_acc_across_servers = prev_row['avg_test_accuracy_across_servers'].iloc[0]
                        # sanitize
                        if pd.isna(self.prev_central_loss): self.prev_central_loss = None
                        if pd.isna(self.prev_central_acc):  self.prev_central_acc  = None
                        if pd.isna(self.prev_avg_loss_across_servers): self.prev_avg_loss_across_servers = None
                        if pd.isna(self.prev_avg_acc_across_servers):  self.prev_avg_acc_across_servers  = None
                        logger.info(f"[Cloud Server] Loaded previous metrics from round {prev_round}")
            except Exception as e:
                logger.warning(f"[Cloud Server] Could not load previous metrics: {e}")

    def aggregate_fit(self, rnd: int, results: List[Tuple[str, FitRes]], failures: List[Any]):
        """Aggregate fit results from leaf servers."""
        if self.round_start_time is None:
            self.round_start_time = time.time()
        
        # Track communication metrics
        total_bytes_up = sum(len(pickle.dumps(res.parameters)) for _, res in results)
        total_bytes_down = total_bytes_up  # Approximate
        
        # Handle failures
        if failures:
            for f in failures:
                pass

        # Call parent aggregation
        agg = super().aggregate_fit(rnd, results, failures)
        
        # Record metrics for this round
        round_time = time.time() - self.round_start_time if self.round_start_time else 0.0
        self.communication_metrics.append({
            "global_round": self.current_global_round,
            "round": rnd,
            "bytes_up": total_bytes_up,
            "bytes_down": total_bytes_down,
            "round_time": round_time,
            "compute_s": 0.0,  # Minimal compute at cloud level
        })
        if agg is None:
            return None

        # Keep latest aggregated parameters for centralized evaluation
        try:
            self.latest_parameters = agg[0]
        except Exception:
            self.latest_parameters = None

        # Each cloud_flower instance is launched fresh for exactly
        # `SERVER_ROUNDS_PER_GLOBAL` server rounds belonging to the global round
        # passed in via the `GLOBAL_ROUND` environment variable.  Therefore when
        # we finish those rounds, we should mark *that* global round complete.
        if rnd % SERVER_ROUNDS_PER_GLOBAL == 0:
            # Use the round number provided by orchestrator instead of
            # recomputing from `rnd`, which always starts at 1 for each process.
            this_global = self.current_global_round

            # 1) Save the new global model
            parameters = agg[0]
            model_path = get_global_model_path(this_global)
            with open(model_path, "wb") as f:
                pickle.dump(parameters_to_ndarrays(parameters), f)
            

            # 2) Write the global_round_{this_global}_complete.signal
            round_sig = get_global_round_signal_path(this_global)
            create_signal_file(round_sig, f"Created completion signal for GLOBAL ROUND {this_global}")
            

            # 3) If that was the _last_ global round, write the final "cloud_complete.signal" and exit
            if this_global == GLOBAL_ROUNDS - 1:
                create_signal_file(complete_signal, "Created final completion signal for whole job")
                
                def delayed_exit():
                    time.sleep(1)
                    sys.exit(0)

                threading.Thread(target=delayed_exit, daemon=True).start()

        return agg

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Don’t launch a round of fit until we have exactly NUM_SERVERS connected.
        """
        while len(client_manager.clients) < NUM_SERVERS:
            time.sleep(0.1)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(self, rnd: int, results: List[Tuple[str, Any]], failures: List[Any]):
        """
        Just “print-and-forward” the evaluation metrics.  If we are on the *last* server-round *of the final global round*, create a final completion signal.
        """
        current_global = self.current_global_round
        
        for sid, eval_res in results:
            acc = eval_res.metrics.get("accuracy", float("nan"))
            
        if failures:
            for f in failures:
                pass
                

        # Call super and unpack exactly two values
        merged = super().aggregate_evaluate(rnd, results, failures)
        if merged is None:
            return None
        loss, metrics = merged

        # Ensure 'accuracy' exists
        if "accuracy" not in metrics:
            metrics["accuracy"] = weighted_average(
                [(r.num_examples, r.metrics) for _, r in results]
            )["accuracy"]
        metrics["num_leaf_servers"] = len(results)
        print(f"Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        # Centralized evaluation on full HHAR test set using latest global params
        central_loss, central_acc = None, None
        try:
            if self.latest_parameters is not None:
                # Prepare model with aggregated weights
                model = Net()
                nds = parameters_to_ndarrays(self.latest_parameters)
                set_weights(model, nds)
                # Centralized (full dataset) evaluation — inline, no partitions
                from fedge.task import (
                    load_hhar_data, HHARDataset,
                    DATA_ROOT, USE_WATCHES, SAMPLE_RATE_HZ,
                    WINDOW_SECONDS, WINDOW_STRIDE_SECONDS,
                )
                from torch.utils.data import DataLoader, Subset
                import numpy as np

                X_all, y_all = load_hhar_data(
                    data_root=DATA_ROOT,
                    use_watches=USE_WATCHES,
                    sample_rate_hz=SAMPLE_RATE_HZ,
                    window_seconds=WINDOW_SECONDS,
                    window_stride_seconds=WINDOW_STRIDE_SECONDS,
                )
                ds = HHARDataset(X_all, y_all, normalize=True)
                n = len(ds)
                n_train = int(0.8 * n)
                test_idx = np.arange(n_train, n, dtype=np.int64)
                testloader_global = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False, num_workers=0)

                device = torch.device("cpu")
                central_loss, central_acc = test(model, testloader_global, device)
                metrics["central_loss"] = float(central_loss)
                metrics["central_accuracy"] = float(central_acc)
        except Exception as e:
            logger.error(f"[Cloud Server] Centralized evaluation failed: {e}")

        # Compute distribution stats across servers for their evaluation results
        server_losses = [float(ev.loss) for _, ev in results]
        server_accs = [float(ev.metrics.get("accuracy", float("nan"))) for _, ev in results]
        loss_mean, loss_std, loss_ci_low, loss_ci_high = _mean_std_ci(server_losses) if server_losses else (float("nan"),)*4
        acc_mean, acc_std, acc_ci_low, acc_ci_high = _mean_std_ci(server_accs) if server_accs else (float("nan"),)*4

        # Generalization gap (centralized vs avg across servers)
        gen_gap_acc_central_minus_servers = None if (central_acc is None or math.isnan(acc_mean)) else (central_acc - acc_mean)
        gen_gap_loss_central_minus_servers = None if (central_loss is None or math.isnan(loss_mean)) else (central_loss - loss_mean)

        # Convergence metrics (delta from previous round)
        delta_central_loss = None if (self.prev_central_loss is None or central_loss is None) else (central_loss - self.prev_central_loss)
        delta_central_acc = None if (self.prev_central_acc is None or central_acc is None) else (central_acc - self.prev_central_acc)
        delta_avg_loss_across_servers = (
            None if (self.prev_avg_loss_across_servers is None or math.isnan(loss_mean))
            else (loss_mean - self.prev_avg_loss_across_servers)
        )
        delta_avg_acc_across_servers = (
            None if (self.prev_avg_acc_across_servers is None or math.isnan(acc_mean))
            else (acc_mean - self.prev_avg_acc_across_servers)
        )
        

        # Update previous values for next round
        self.prev_central_loss = central_loss
        self.prev_central_acc = central_acc
        self.prev_avg_loss_across_servers = loss_mean if not math.isnan(loss_mean) else None
        self.prev_avg_acc_across_servers = acc_mean if not math.isnan(acc_mean) else None

        # Write standardized cloud metrics to metrics/cloud/rounds.csv
        cloud_dir = Path().resolve() / "metrics" / "cloud"
        cloud_dir.mkdir(parents=True, exist_ok=True)
        rounds_csv = cloud_dir / "rounds.csv"
        write_header = not rounds_csv.exists()
        with open(rounds_csv, "a", newline="") as gf:
            writer = csv.DictWriter(gf, fieldnames=[
                "global_round",
                "avg_test_loss_across_servers","std_test_loss_across_servers","ci95_low_test_loss_across_servers","ci95_high_test_loss_across_servers",
                "avg_test_accuracy_across_servers","std_test_accuracy_across_servers","ci95_low_test_accuracy_across_servers","ci95_high_test_accuracy_across_servers",
                "global_test_loss_centralized","global_test_accuracy_centralized",
                "gen_gap_loss_central_minus_servers","gen_gap_accuracy_central_minus_servers",
                "delta_global_loss_centralized","delta_global_accuracy_centralized",
                "delta_avg_loss_across_servers","delta_avg_accuracy_across_servers",
                "bytes_up_total","bytes_down_total",
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": current_global,
                "avg_test_loss_across_servers": float(loss),
                "std_test_loss_across_servers": loss_std,
                "ci95_low_test_loss_across_servers": loss_ci_low,
                "ci95_high_test_loss_across_servers": loss_ci_high,
                "avg_test_accuracy_across_servers": float(metrics["accuracy"]),
                "std_test_accuracy_across_servers": acc_std,
                "ci95_low_test_accuracy_across_servers": acc_ci_low,
                "ci95_high_test_accuracy_across_servers": acc_ci_high,
                "global_test_loss_centralized": None if central_loss is None else float(central_loss),
                "global_test_accuracy_centralized": None if central_acc is None else float(central_acc),
                "gen_gap_loss_central_minus_servers": gen_gap_loss_central_minus_servers,
                "gen_gap_accuracy_central_minus_servers": gen_gap_acc_central_minus_servers,
                "delta_global_loss_centralized": delta_central_loss,
                "delta_global_accuracy_centralized": delta_central_acc,
                "delta_avg_loss_across_servers": delta_avg_loss_across_servers,
                "delta_avg_accuracy_across_servers": delta_avg_acc_across_servers,
                "bytes_up_total": 0,
                "bytes_down_total": 0,
            })

        # Write cloud communication CSV
        self._write_cloud_comm_csv()
        
        # Final completion signals only on the very last leaf-round of the overall training
        is_last_global_round = current_global == GLOBAL_ROUNDS - 1
        is_last_server_round = rnd == SERVER_ROUNDS_PER_GLOBAL
        if is_last_global_round and is_last_server_round:
            create_signal_file(complete_signal, "Created completion signal after final aggregate_evaluate")

        return loss, metrics
    
    def _write_cloud_comm_csv(self):
        """Write cloud communication metrics to CSV."""
        if not self.communication_metrics:
            return
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.communication_metrics)
            out_path = Path(os.getenv("RUN_DIR", ".")) / "cloud_comm.csv"
            
            mode = "a" if out_path.exists() else "w"
            df.to_csv(out_path, index=False, mode=mode, header=not out_path.exists())
            logger.info(f"[Cloud Server] Wrote communication CSV → {out_path}")
        except Exception as e:
            logger.warning(f"[Cloud Server] Could not write comm CSV: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN: spin up the FedAvg server for exactly GLOBAL_ROUNDS * SERVER_ROUNDS_PER_GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_rounds = SERVER_ROUNDS_PER_GLOBAL
    config = ServerConfig(num_rounds=total_rounds)

    
    strategy = CloudFedAvg()

    try:
        bind_addr = os.getenv("BIND_ADDRESS", "0.0.0.0")
        history = start_server(
            server_address=f"{bind_addr}:{CLOUD_PORT}",
            config=config,
            strategy=strategy,
        )
        # In principle we should never get here unless the server shuts down cleanly
        if not complete_signal.exists():
            create_signal_file(complete_signal, "Created completion signal after shutdown")

        # Communication metrics are now written by the strategy itself
        pass
    except Exception as e:
        
        # Always ensure a final “complete” file
        create_signal_file(complete_signal, "Created completion signal after error")
        sys.exit(1)
