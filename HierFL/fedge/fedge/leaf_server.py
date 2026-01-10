# fedge/leaf_server.py

import argparse
import pickle
import os
import sys
import time
import signal
import warnings
import logging
import csv
import math
from typing import List, Tuple, Optional, Any
from pathlib import Path
import toml
from fedge.utils import fs

from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg

from fedge.task import Net, get_weights, set_weights, load_data, test
import torch

# Configure logging (minimal verbosity)
logging.basicConfig(
    level=logging.ERROR,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Suppress all noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc", "urllib3", "requests"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Drop noisy messages from stdout/stderr
class _DropNoisy:
    def __init__(self, out): self._out = out
    def write(self, txt):
        skip = ("DEPRECATED FEATURE", "INFO flwr", "DEBUG flwr", "gRPC", "INFO:flwr")
        if not any(s in txt for s in skip):
            self._out.write(txt)
    def flush(self): self._out.flush()
import sys
sys.stdout = _DropNoisy(sys.stdout)
sys.stderr = _DropNoisy(sys.stderr)

# Statistical helpers now imported from shared stats module


class LeafFedAvg(FedAvg):
    def __init__(
        self,
        server_id: int,
        num_rounds: int,
        fraction_fit: float,
        fraction_evaluate: float,
        clients_per_server: int,
        initial_parameters,
        global_round: int = 0,
    ):
        # Pass most args to FedAvg base class
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=clients_per_server,
            initial_parameters=initial_parameters,
            #fit_metrics_aggregation_fn=self.weighted_average,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )
        self.server_id = server_id
        self.num_rounds = num_rounds
        self.server_str = f"Leaf Server {server_id}"
        self.global_round = global_round
        # Store config for local-central evaluation
        self.clients_per_server = clients_per_server
        try:
            cfg = toml.load((Path(__file__).resolve().parent.parent) / "pyproject.toml")
            self.num_servers = cfg["tool"]["flwr"]["hierarchy"]["num_servers"]
            cfg_seed = int(cfg["tool"]["flwr"]["app"]["config"].get("seed", 42))
        except Exception:
            self.num_servers = 1
            cfg_seed = 42
        # Environment variable FL_SEED overrides config file
        self.seed = int(os.environ.get("FL_SEED", cfg_seed))
        # Project root and metrics directory
        script_dir = Path(__file__).resolve().parent
        self.project_root = script_dir.parent
        self.base_dir = fs.leaf_server_dir(self.project_root, server_id)
        (self.base_dir / "models").mkdir(parents=True, exist_ok=True)
        # Metrics directory for this seed
        self.metrics_dir = self.project_root / "metrics" / f"seed_{self.seed}"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        # Track previous round metrics for convergence calculation
        self.prev_avg_client_loss = None
        self.prev_avg_client_acc = None

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Same as in server_app.py
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        total_examples = sum([num_examples for num_examples, _ in metrics])
        return {"accuracy": sum(accuracies) / total_examples}

    def _write_fit_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        # Client metrics collection disabled - only server partition metrics are saved
        pass

    def _write_eval_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        # Client metrics collection disabled - only server partition metrics are saved
        pass

    def aggregate_fit(self, rnd, results, failures):
        
        # Print per-client train_loss and num_examples
        for cid, fit_res in results:
            friendly = fit_res.metrics.get("client_id", str(cid))
            loss = fit_res.metrics.get("train_loss", None)
            n = fit_res.num_examples
            
        if failures:
            for failure in failures:
                pass
                
        # Dump current round client metrics
        self._write_fit_metrics_csv(rnd, results)
        start_time = time.perf_counter()
        aggregated = super().aggregate_fit(rnd, results, failures)
        end_time = time.perf_counter()
        # Record server communication and computation metrics
        self.server_bytes_down = sum(fit_res.metrics.get("bytes_down", 0) for _, fit_res in results)
        self.server_bytes_up = sum(fit_res.metrics.get("bytes_up", 0) for _, fit_res in results)
        self.server_comp_time = end_time - start_time
        self.server_round_time = end_time - start_time

        if aggregated is not None and isinstance(aggregated, tuple) and len(aggregated) > 0:
            self.latest_parameters = aggregated[0]

        if rnd == self.num_rounds and aggregated is not None:
            # Calculate total number of training examples from this round's results
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            

            # Extract parameters from the returned tuple
            if isinstance(aggregated, tuple) and len(aggregated) > 0:
                parameters = aggregated[0]
            else:
                # Fallback if the return type is not a tuple
                parameters = aggregated

            # Save model in flat models/ directory
            model_dir = self.base_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"model_s{self.server_id}_g{self.global_round}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump((parameters_to_ndarrays(parameters), total_examples), f)
            

        return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        # Client metrics collection disabled - only server partition metrics are saved
        # Skip per-client evaluation metrics writing
        # self._write_eval_metrics_csv(rnd, results)

        # Aggregate client evaluation (FedAvg returns (loss, metrics))
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        # Server partition evaluation on full test set
        local_rnd = rnd - 1
        model = Net()
        if hasattr(self, "latest_parameters") and self.latest_parameters is not None:
            nds = parameters_to_ndarrays(self.latest_parameters)
            set_weights(model, nds)
        # 3-a) Server evaluation on its assigned user partitions (union of this server's clients)
        server_partition_loss, server_partition_acc = None, None
        try:
            # Load PARTITIONS_JSON and get union of all client indices for this server
            import json
            partitions_json = os.environ.get("PARTITIONS_JSON")
            if partitions_json and os.path.exists(partitions_json):
                with open(partitions_json, 'r') as f:
                    mapping = json.load(f)
                
                # Collect all indices for this server's clients
                server_indices = []
                if str(self.server_id) in mapping:
                    for cid in range(self.clients_per_server):
                        if str(cid) in mapping[str(self.server_id)]:
                            server_indices.extend(mapping[str(self.server_id)][str(cid)])
                
                if server_indices:
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
                    # Use server's specific indices for partition evaluation
                    testloader_server = DataLoader(Subset(ds, server_indices), batch_size=32, shuffle=False, num_workers=0)
                    device = torch.device("cpu")
                    server_partition_loss, server_partition_acc = test(model, testloader_server, device)
        except Exception as e:
            logger.error(f"[{self.server_str}] Server partition evaluation failed: {e}")

        # 3-b) Server evaluation on full dataset (for comparison - this is what cloud should do)
        full_dataset_loss, full_dataset_acc = None, None
        try:
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

            device_central = torch.device("cpu")
            full_dataset_loss, full_dataset_acc = test(model, testloader_global, device_central)
        except Exception as e:
            logger.error(f"[{self.server_str}] Full dataset evaluation failed: {e}")

        # Convergence metrics (delta from previous round) based on server partition
        delta_loss = None if (self.prev_avg_client_loss is None or server_partition_loss is None) else (server_partition_loss - self.prev_avg_client_loss)
        delta_acc = None if (self.prev_avg_client_acc is None or server_partition_acc is None) else (server_partition_acc - self.prev_avg_client_acc)

        # Update previous values for next round (using server partition metrics)
        self.prev_avg_client_loss = server_partition_loss
        self.prev_avg_client_acc = server_partition_acc

        # Server-level metrics CSV (no client metrics, path in metrics/seed_{SEED}/)
        server_csv = self.metrics_dir / f"server_{self.server_id}.csv"
        write_header = not server_csv.exists()
        with open(server_csv, "a", newline="") as f:
            fieldnames = [
                "global_round", "local_round",
                "server_partition_test_loss", "server_partition_test_accuracy",
                "conv_loss_delta", "conv_acc_delta",
                "bytes_down", "bytes_up", "comp_time", "round_time",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "server_partition_test_loss": server_partition_loss,
                "server_partition_test_accuracy": server_partition_acc,
                "conv_loss_delta": delta_loss if delta_loss is not None else "",
                "conv_acc_delta": delta_acc if delta_acc is not None else "",
                "bytes_down": getattr(self, "server_bytes_down", 0),
                "bytes_up": getattr(self, "server_bytes_up", 0),
                "comp_time": getattr(self, "server_comp_time", 0.0),
                "round_time": getattr(self, "server_round_time", 0.0),
            })
        return aggregated


def handle_signal(sig, frame):
    """Handle termination signals gracefully"""
    server_str = os.environ.get("SERVER_STR", "Leaf Server")
    logger.info(f"[{server_str}] Received signal {sig}, shutting down gracefully...")
    
    # If we know the server ID, try to save the model before exiting
    server_id = os.environ.get("SERVER_ID")
    if server_id:
        try:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            model_dir = project_root / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"server_{server_id}.pkl"
            
            # Try to get a model instance
            net = Net()
            ndarrays = get_weights(net)
            
            # Save with a default example count
            with open(model_path, "wb") as f:
                pickle.dump((ndarrays, 0), f)
            logger.info(f"[{server_str}] Saved emergency backup model during shutdown to {model_path}")
        except Exception as e:
            logger.error(f"[{server_str}] Failed to save emergency model: {e}")
    
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--clients_per_server", type=int, required=True)
    parser.add_argument("--num_rounds", type=int, required=True)
    parser.add_argument("--fraction_fit", type=float, required=True)
    parser.add_argument("--fraction_evaluate", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--initial_model_path", type=str, help="Path to global model to start from")
    parser.add_argument("--global_round", type=int, default=0, help="Current global round number (0-indexed)")
    parser.add_argument("--start_round", type=int, default=0, help="Starting round number (0-indexed)")
    parser.add_argument("--dir_round", type=int, help="Round number for directory naming (0-indexed)")
    
    args = parser.parse_args()

    # Get server ID for logging
    server_id = args.server_id
    server_str = f"Leaf Server {server_id}"
    
    # Store in environment for signal handlers
    os.environ["SERVER_ID"] = str(server_id)
    os.environ["SERVER_STR"] = server_str
    
    logger.info(f"[{server_str}] Starting server with {args.clients_per_server} clients, {args.num_rounds} rounds")
    
    # Initialize model and get parameters
    initial_parameters = None
    if args.initial_model_path:
        model_path = Path(args.initial_model_path)
        if model_path.exists():
            logger.info(f"[{server_str}] Loading initial model from {model_path}")
            try:
                with open(model_path, "rb") as f:
                    # The global model could be stored in different formats
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, tuple):
                        # If it's a tuple, first element should be the model parameters
                        ndarrays = loaded_data[0]
                    else:
                        # Otherwise, assume it's the raw parameters
                        ndarrays = loaded_data
                logger.info(f"[{server_str}] Starting with global model from round {args.global_round}")
                initial_parameters = ndarrays_to_parameters(ndarrays)
            except Exception as e:
                logger.error(f"[{server_str}] Error loading initial model: {e}")
                logger.info(f"[{server_str}] Starting with fresh model parameters")
    
    # Configure strategy
    strategy = LeafFedAvg(
        server_id=server_id,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        clients_per_server=args.clients_per_server,
        initial_parameters=initial_parameters,
        global_round=args.global_round,
    )
    
    # Start server with error handling
    logger.info(f"[{server_str}] Starting Flower server on port {args.port}")
    try:
        bind_addr = os.getenv("BIND_ADDRESS", "0.0.0.0")
        history = start_server(
            server_address=f"{bind_addr}:{args.port}",
            config=ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )
        logger.info(f"[{server_str}] Server has completed all rounds successfully")
        # ------------------------------------------------------------------
        # Persist per-round communication metrics
        # ------------------------------------------------------------------
        try:
            import pandas as pd, filelock
            mdf = getattr(history, "metrics_distributed_fit", {}) or {}
            def _collect(keys):
                for k in keys:
                    if k in mdf:
                        return mdf[k]
                return []
            up_entries = _collect(["bytes_up", "bytes_written"])
            down_entries = _collect(["bytes_down", "bytes_read"])
            # Evaluation-phase download traffic (e.g. parameters pushed during evaluate())
            eval_down_entries = _collect(["bytes_down_eval"])
            rt_entries = _collect(["round_time"])
            comp_entries = _collect(["compute_s"])
            by_round: dict[int, dict[str, float]] = {}
            for rnd, val in up_entries:
                by_round.setdefault(rnd, {})["bytes_up"] = int(val)
            for rnd, val in down_entries:
                by_round.setdefault(rnd, {})["bytes_down"] = int(val)
            # Merge evaluation download into totals and keep separate field
            for rnd, val in eval_down_entries:
                by_round.setdefault(rnd, {})["bytes_down_eval"] = int(val)
                by_round[rnd]["bytes_down"] = int(val) + int(by_round[rnd].get("bytes_down", 0))
            for rnd, val in rt_entries:
                by_round.setdefault(rnd, {})["round_time"] = float(val)
            for rnd, val in comp_entries:
                by_round.setdefault(rnd, {})["compute_s"] = float(val)
            rows = [
                {"global_round": args.global_round,
                 "round": rnd,
                 "bytes_up": int(vals.get("bytes_up", 0)),
                 "bytes_down": int(vals.get("bytes_down", 0)),
                 "bytes_down_eval": int(vals.get("bytes_down_eval", 0)),
                 "round_time": vals.get("round_time", 0.0),
                 "compute_s": vals.get("compute_s", 0.0)}
                for rnd, vals in sorted(by_round.items())]
            if rows:
                df = pd.DataFrame(rows)
                out = Path(os.getenv("RUN_DIR", ".")) / f"edge_comm_{server_id}.csv"
                with filelock.FileLock(out.with_suffix(".lock")):
                    mode = "a" if out.exists() else "w"
                    df.to_csv(out, index=False, mode=mode, header=not out.exists())
                logger.info(f"[{server_str}] Wrote communication CSV → {out}")
        except Exception as csv_err:
            logger.warning(f"[{server_str}] Could not write comm CSV: {csv_err}")
    except KeyboardInterrupt:
        logger.info(f"[{server_str}] Server interrupted by user")
    except Exception as e:
        logger.error(f"[{server_str}] Server error: {e}")
        # Try to save the model in case of unexpected error
        try:
            if hasattr(strategy, "parameters") and strategy.parameters is not None:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                model_dir = project_root / "models"
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / f"server_{server_id}.pkl"
                
                with open(model_path, "wb") as f:
                    pickle.dump((parameters_to_ndarrays(strategy.parameters), 0), f)
                logger.info(f"[{server_str}] Saved error recovery model to {model_path}")
        except Exception as save_error:
            logger.error(f"[{server_str}] Could not save error recovery model: {save_error}")


if __name__ == "__main__":
    main()
