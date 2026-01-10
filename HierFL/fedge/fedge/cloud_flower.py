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
import numpy as np

# Get SEED for metrics folder organization
SEED = int(os.environ.get("FL_SEED", "42"))

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

# Parse clients_per_server (can be int or list)
raw_cps = hier.get("clients_per_server", 3)
if isinstance(raw_cps, list):
    CLIENTS_PER_SERVER = raw_cps
else:
    CLIENTS_PER_SERVER = [int(raw_cps)] * NUM_SERVERS

# Create seed-based metrics directory
metrics_dir = project_root / "metrics" / f"seed_{SEED}"
metrics_dir.mkdir(parents=True, exist_ok=True)

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
        return True
    except Exception as e:
        return False

# ──────────────────────────────────────────────────────────────────────────────
#  Logging / Warnings suppression (reduced verbosity)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc", "urllib3", "requests"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Drop "DEPRECATED FEATURE" and other noisy messages on stdout/stderr
class _DropNoisy:
    def __init__(self, out): self._out = out
    def write(self, txt):
        skip = ("DEPRECATED FEATURE", "INFO flwr", "DEBUG flwr", "gRPC")
        if not any(s in txt for s in skip):
            self._out.write(txt)
    def flush(self): self._out.flush()

sys.stdout = _DropNoisy(sys.stdout)
sys.stderr = _DropNoisy(sys.stderr)

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
        """Load previous round metrics from centralized_metrics.csv."""
        self.prev_central_loss = None
        self.prev_central_acc = None
        self.prev_avg_loss_across_servers = None
        self.prev_avg_acc_across_servers = None
        if self.current_global_round > 0:
            try:
                import pandas as pd
                central_csv = metrics_dir / "centralized_metrics.csv"
                if central_csv.exists():
                    df = pd.read_csv(central_csv)
                    prev_round = self.current_global_round - 1
                    prev_row = df[df['round'] == prev_round]
                    if not prev_row.empty:
                        self.prev_central_loss = prev_row['central_test_loss'].iloc[0]
                        self.prev_central_acc = prev_row['central_test_accuracy'].iloc[0]
                        # sanitize
                        if pd.isna(self.prev_central_loss): self.prev_central_loss = None
                        if pd.isna(self.prev_central_acc):  self.prev_central_acc = None
                # Also load distributed metrics for avg across servers
                dist_csv = metrics_dir / "distributed_metrics.csv"
                if dist_csv.exists():
                    df = pd.read_csv(dist_csv)
                    prev_round = self.current_global_round - 1
                    prev_row = df[df['round'] == prev_round]
                    if not prev_row.empty:
                        self.prev_avg_loss_across_servers = prev_row['avg_loss'].iloc[0] if 'avg_loss' in prev_row else None
                        self.prev_avg_acc_across_servers = prev_row['avg_accuracy'].iloc[0] if 'avg_accuracy' in prev_row else None
                        if pd.isna(self.prev_avg_loss_across_servers): self.prev_avg_loss_across_servers = None
                        if pd.isna(self.prev_avg_acc_across_servers):  self.prev_avg_acc_across_servers = None
            except Exception as e:
                pass  # Silently continue if no previous metrics

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
        """Aggregate evaluation results and write metrics CSVs."""
        current_global = self.current_global_round

        # Call super and unpack
        merged = super().aggregate_evaluate(rnd, results, failures)
        if merged is None:
            return None
        loss, metrics = merged

        # Ensure 'accuracy' exists
        if "accuracy" not in metrics:
            metrics["accuracy"] = weighted_average(
                [(r.num_examples, r.metrics) for _, r in results]
            )["accuracy"]

        # ══════════════════════════════════════════════════════════════════════
        # CENTRALIZED EVALUATION on full HHAR test set
        # ══════════════════════════════════════════════════════════════════════
        central_loss, central_acc = None, None
        central_train_loss, central_train_acc = None, None
        try:
            if self.latest_parameters is not None:
                model = Net()
                nds = parameters_to_ndarrays(self.latest_parameters)
                set_weights(model, nds)
                from fedge.task import (
                    load_hhar_data, HHARDataset,
                    DATA_ROOT, USE_WATCHES, SAMPLE_RATE_HZ,
                    WINDOW_SECONDS, WINDOW_STRIDE_SECONDS,
                )
                from torch.utils.data import DataLoader, Subset

                X_all, y_all = load_hhar_data(
                    data_root=DATA_ROOT, use_watches=USE_WATCHES,
                    sample_rate_hz=SAMPLE_RATE_HZ, window_seconds=WINDOW_SECONDS,
                    window_stride_seconds=WINDOW_STRIDE_SECONDS,
                )
                ds = HHARDataset(X_all, y_all, normalize=True)
                n = len(ds)
                n_train = int(0.8 * n)
                train_idx = np.arange(0, n_train, dtype=np.int64)
                test_idx = np.arange(n_train, n, dtype=np.int64)
                trainloader = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=False, num_workers=0)
                testloader = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False, num_workers=0)

                device = torch.device("cpu")
                central_train_loss, central_train_acc = test(model, trainloader, device)
                central_loss, central_acc = test(model, testloader, device)
        except Exception as e:
            pass  # Continue without centralized metrics

        # ══════════════════════════════════════════════════════════════════════
        # WRITE centralized_metrics.csv (matches CIFAR-10 format)
        # ══════════════════════════════════════════════════════════════════════
        conv_loss_rate = 0.0 if self.prev_central_loss is None or central_loss is None else (central_loss - self.prev_central_loss)
        conv_acc_rate = 0.0 if self.prev_central_acc is None or central_acc is None else (central_acc - self.prev_central_acc)
        central_loss_gap = (central_loss - central_train_loss) if (central_loss is not None and central_train_loss is not None) else 0.0
        central_acc_gap = (central_train_acc - central_acc) if (central_train_acc is not None and central_acc is not None) else 0.0

        central_csv = metrics_dir / "centralized_metrics.csv"
        write_header = not central_csv.exists()
        with open(central_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "round", "central_train_loss", "central_train_accuracy",
                "central_test_loss", "central_test_accuracy",
                "central_loss_gap", "central_accuracy_gap",
                "conv_loss_rate", "conv_acc_rate", "conv_loss_stability", "conv_acc_stability",
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "round": current_global,
                "central_train_loss": central_train_loss if central_train_loss else 0.0,
                "central_train_accuracy": central_train_acc if central_train_acc else 0.0,
                "central_test_loss": central_loss if central_loss else 0.0,
                "central_test_accuracy": central_acc if central_acc else 0.0,
                "central_loss_gap": central_loss_gap,
                "central_accuracy_gap": central_acc_gap,
                "conv_loss_rate": conv_loss_rate,
                "conv_acc_rate": conv_acc_rate,
                "conv_loss_stability": 0.0,
                "conv_acc_stability": 0.0,
            })

        # Update previous values
        self.prev_central_loss = central_loss
        self.prev_central_acc = central_acc

        # ══════════════════════════════════════════════════════════════════════
        # READ server partition metrics and WRITE distributed_metrics.csv + server_{sid}.csv
        # ══════════════════════════════════════════════════════════════════════
        self._write_distributed_and_servers_csv(current_global)

        # Print concise round summary
        central_str = f", Central: {central_acc:.4f}" if central_acc else ""
        print(f"Round {current_global}: Loss={loss:.4f}, Acc={metrics['accuracy']:.4f}{central_str}")

        # Final completion signals
        is_last_global_round = current_global == GLOBAL_ROUNDS - 1
        is_last_server_round = rnd == SERVER_ROUNDS_PER_GLOBAL
        if is_last_global_round and is_last_server_round:
            create_signal_file(complete_signal, "Training complete")

        return loss, metrics

    def _write_distributed_and_servers_csv(self, round_num: int):
        """
        Read SERVER PARTITION metrics from leaf servers and write:
        - distributed_metrics.csv (aggregated stats across servers)
        - server_{sid}.csv (one file per server with partition metrics)

        Uses server_partition_test_accuracy/loss directly - NO client-level data.
        """
        server_accs, server_losses = [], []
        total_bytes_up, total_bytes_down = 0, 0
        total_comp_time = 0.0

        for sid in range(NUM_SERVERS):
            base_dir = fs.leaf_server_dir(project_root, sid)
            # Read from servers/rounds.csv - the server partition metrics
            server_rounds_csv = base_dir / "metrics" / "servers" / "rounds.csv"

            srv_acc, srv_loss = None, None
            srv_bytes_up, srv_bytes_down = 0, 0
            srv_comp_time, srv_round_time = 0.0, 0.0

            if server_rounds_csv.exists():
                try:
                    with open(server_rounds_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            if int(r.get("global_round", -1)) == round_num:
                                # Get server partition metrics directly
                                srv_acc = float(r.get("server_partition_test_accuracy", 0.0) or 0.0)
                                srv_loss = float(r.get("server_partition_test_loss", 0.0) or 0.0)
                                srv_bytes_up = int(r.get("bytes_up_total", 0) or 0)
                                srv_bytes_down = int(r.get("bytes_down_total", 0) or 0)
                                srv_comp_time = float(r.get("comp_time_sec_mean", 0.0) or 0.0)
                                srv_round_time = float(r.get("round_time_sec", 0.0) or 0.0)
                                break
                except Exception:
                    pass

            # Collect for distributed stats
            if srv_acc is not None:
                server_accs.append(srv_acc)
                server_losses.append(srv_loss)
                total_bytes_up += srv_bytes_up
                total_bytes_down += srv_bytes_down
                total_comp_time += srv_comp_time

            # Write individual server_{sid}.csv file
            server_csv_path = metrics_dir / f"server_{sid}.csv"
            srv_write_header = not server_csv_path.exists()
            srv_fieldnames = [
                "round", "server_partition_accuracy", "server_partition_loss",
                "bytes_up", "bytes_down", "comp_time_sec", "round_time_sec"
            ]
            with open(server_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=srv_fieldnames)
                if srv_write_header:
                    writer.writeheader()
                writer.writerow({
                    "round": round_num,
                    "server_partition_accuracy": srv_acc if srv_acc is not None else 0.0,
                    "server_partition_loss": srv_loss if srv_loss is not None else 0.0,
                    "bytes_up": srv_bytes_up,
                    "bytes_down": srv_bytes_down,
                    "comp_time_sec": srv_comp_time,
                    "round_time_sec": srv_round_time,
                })

        # Compute statistics across servers using t-distribution CI
        if server_accs:
            acc_mean, acc_std, acc_ci_lo, acc_ci_hi = _mean_std_ci(server_accs)
            loss_mean, loss_std, loss_ci_lo, loss_ci_hi = _mean_std_ci(server_losses)
        else:
            acc_mean, acc_std, acc_ci_lo, acc_ci_hi = 0.0, 0.0, 0.0, 0.0
            loss_mean, loss_std, loss_ci_lo, loss_ci_hi = 0.0, 0.0, 0.0, 0.0

        # Build distributed record (aggregated server partition metrics)
        distributed = {
            "avg_accuracy": acc_mean,
            "avg_loss": loss_mean,
            "accuracy_std": acc_std,
            "loss_std": loss_std,
            "acc_ci95_lo": float(acc_ci_lo),
            "acc_ci95_hi": float(acc_ci_hi),
            "loss_ci95_lo": float(loss_ci_lo),
            "loss_ci95_hi": float(loss_ci_hi),
            "total_bytes_up": total_bytes_up,
            "total_bytes_down": total_bytes_down,
            "total_comp_time_sec": total_comp_time,
            "num_servers": len(server_accs),
        }

        # Write distributed_metrics.csv
        dist_path = metrics_dir / "distributed_metrics.csv"
        write_header = not dist_path.exists()
        with open(dist_path, "a", newline="") as f:
            fieldnames = ["round"] + list(distributed.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"round": round_num, **distributed})

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
