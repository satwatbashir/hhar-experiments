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
from fedge.stats import _mean_std_ci
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
        except Exception:
            self.num_servers = 1
        # Flat per-server directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        self.base_dir = fs.leaf_server_dir(project_root, server_id)
        (self.base_dir / "models").mkdir(parents=True, exist_ok=True)
        # Track previous round metrics for convergence calculation
        self.prev_avg_client_loss = None
        self.prev_avg_client_acc = None

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Same as in server_app.py
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        total_examples = sum([num_examples for num_examples, _ in metrics])
        return {"accuracy": sum(accuracies) / total_examples}

    def _write_fit_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        # Save per-client fit metrics into strict round folder
        local_rnd = rnd - 1
        # Suppressed legacy per-server client_fit_metrics.csv; using standardized clients.csv only
        logger.info(f"[{self.server_str} | Round {local_rnd}] Recorded client fit metrics → standardized clients.csv")

        # Also append standardized per-client metrics to per-server metrics/clients.csv
        std_metrics_dir = self.base_dir / "metrics"
        std_metrics_dir.mkdir(parents=True, exist_ok=True)
        std_clients_csv = std_metrics_dir / "clients.csv"
        std_write_header = not std_clients_csv.exists()
        with open(std_clients_csv, "a", newline="") as f:
            std_fields = [
                "global_round","local_round","server_id","cid",
                "num_examples","train_loss","train_accuracy","test_loss","test_accuracy",
                "accuracy_gap","loss_gap","comp_time_sec","download_bytes","upload_bytes",
            ]
            std_writer = csv.DictWriter(f, fieldnames=std_fields)
            if std_write_header:
                std_writer.writeheader()
            for cid, fit_res in results:
                cid_str = fit_res.metrics.get("client_id", str(cid))
                std_writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "server_id": self.server_id,
                    "cid": cid_str,
                    "num_examples": fit_res.num_examples,
                    "train_loss": fit_res.metrics.get("train_loss", ""),
                    "train_accuracy": fit_res.metrics.get("accuracy", ""),
                    "test_loss": "",
                    "test_accuracy": "",
                    "accuracy_gap": "",
                    "loss_gap": "",
                    "comp_time_sec": fit_res.metrics.get("round_time", 0.0),
                    "download_bytes": fit_res.metrics.get("bytes_down", 0),
                    "upload_bytes": fit_res.metrics.get("bytes_up", 0),
                })

    def _write_eval_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client evaluation loss & accuracy for this round"""
        local_rnd = rnd - 1
        # Suppressed legacy per-server client_eval_metrics.csv; using standardized clients.csv only
        logger.info(f"[{self.server_str} | Round {local_rnd}] Recorded client eval metrics → standardized clients.csv")

        # Also upsert standardized per-client evaluation to per-server metrics/clients.csv
        std_metrics_dir = self.base_dir / "metrics"
        std_metrics_dir.mkdir(parents=True, exist_ok=True)
        std_clients_csv = std_metrics_dir / "clients.csv"
        # Load existing
        existing_rows = []
        existing_fields = []
        if std_clients_csv.exists():
            with open(std_clients_csv, "r", newline="") as rf:
                reader = csv.DictReader(rf)
                existing_rows = list(reader)
                existing_fields = list(reader.fieldnames) if reader.fieldnames else []
        # Ensure fields
        std_fields = [
            "global_round","local_round","server_id","cid",
            "num_examples","train_loss","train_accuracy","test_loss","test_accuracy",
            "accuracy_gap","loss_gap","conv_test_loss_delta","conv_test_acc_delta",
            "comp_time_sec","download_bytes","upload_bytes",
        ]
        for c in std_fields:
            if c not in existing_fields:
                existing_fields.append(c)
        # Index by unique key
        def _key(r):
            return (
                str(r.get("global_round","")),
                str(r.get("local_round","")),
                str(r.get("server_id","")),
                str(r.get("cid",""))
            )
        row_map = {_key(r): r for r in existing_rows}
        def _find_prev(gr: int, lr: int, sid: int, cid: str):
            return row_map.get((str(gr), str(lr-1), str(sid), cid))
        for cid, eval_res in results:
            cid_str = str(eval_res.metrics.get("client_id", str(cid)))
            k = (str(self.global_round), str(local_rnd), str(self.server_id), cid_str)
            row = row_map.get(k, {c: "" for c in existing_fields})
            row.update({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "server_id": self.server_id,
                "cid": cid_str,
                "num_examples": eval_res.num_examples,
                "test_loss": eval_res.loss,
                "test_accuracy": eval_res.metrics.get("accuracy", ""),
            })
            # Gaps if train stats present
            try:
                tr_loss = float(row.get("train_loss","")) if str(row.get("train_loss","")) != "" else None
            except ValueError:
                tr_loss = None
            try:
                tr_acc = float(row.get("train_accuracy","")) if str(row.get("train_accuracy","")) != "" else None
            except ValueError:
                tr_acc = None
            te_loss = float(row["test_loss"]) if row.get("test_loss") not in ("", None) else None
            te_acc  = float(row["test_accuracy"]) if row.get("test_accuracy") not in ("", None) else None
            row["loss_gap"] = (te_loss - tr_loss) if (te_loss is not None and tr_loss is not None) else ""
            row["accuracy_gap"] = (tr_acc - te_acc) if (tr_acc is not None and te_acc is not None) else ""
            # Convergence deltas vs previous local round
            prev = _find_prev(self.global_round, local_rnd, self.server_id, cid_str)
            if prev is not None:
                try:
                    prev_te_loss = float(prev.get("test_loss","")) if prev.get("test_loss","") != "" else None
                except ValueError:
                    prev_te_loss = None
                try:
                    prev_te_acc = float(prev.get("test_accuracy","")) if prev.get("test_accuracy","") != "" else None
                except ValueError:
                    prev_te_acc = None
                row["conv_test_loss_delta"] = (te_loss - prev_te_loss) if (te_loss is not None and prev_te_loss is not None) else ""
                row["conv_test_acc_delta"]  = (te_acc - prev_te_acc)   if (te_acc is not None and prev_te_acc is not None)   else ""
            else:
                row["conv_test_loss_delta"] = ""
                row["conv_test_acc_delta"]  = ""
            # Propagate bytes/comp if present
            row["comp_time_sec"]  = row.get("comp_time_sec", 0.0)
            row["download_bytes"] = eval_res.metrics.get("bytes_down_eval", row.get("download_bytes", 0))
            row["upload_bytes"]   = row.get("upload_bytes", 0)
            row_map[k] = row
        # Write merged file
        with open(std_clients_csv, "w", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=existing_fields)
            writer.writeheader()
            for rkey in sorted(row_map.keys()):
                writer.writerow(row_map[rkey])

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
        # 1) Save per-client evaluation metrics
        self._write_eval_metrics_csv(rnd, results)

        # 2) Aggregate client evaluation (FedAvg returns (loss, metrics))
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is not None and isinstance(aggregated, tuple):
            agg_loss = aggregated[0]
            agg_acc = aggregated[1].get("accuracy") if aggregated[1] else None
        else:
            agg_loss, agg_acc = None, None

        # Compute per-client distribution statistics (unweighted)
        client_losses = [float(ev.loss) for _, ev in results]
        client_accs = [float(ev.metrics.get("accuracy")) for _, ev in results if "accuracy" in ev.metrics]
        if client_losses:
            loss_mean, loss_std, loss_ci_low, loss_ci_high = _mean_std_ci(client_losses)
        else:
            loss_mean = loss_std = loss_ci_low = loss_ci_high = float("nan")
        if client_accs:
            acc_mean, acc_std, acc_ci_low, acc_ci_high = _mean_std_ci(client_accs)
        else:
            acc_mean = acc_std = acc_ci_low = acc_ci_high = float("nan")

        # 3) Centralized evaluation on full test set
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

        # 4) Convergence metrics (delta from previous round)
        delta_avg_client_loss = None if (self.prev_avg_client_loss is None or agg_loss is None) else (agg_loss - self.prev_avg_client_loss)
        delta_avg_client_acc = None if (self.prev_avg_client_acc is None or agg_acc is None) else (agg_acc - self.prev_avg_client_acc)
        
        # Update previous values for next round
        self.prev_avg_client_loss = agg_loss
        self.prev_avg_client_acc = agg_acc

        # 5) Standardized server-level metrics CSV → metrics/servers/rounds.csv
        std_metrics_dir = self.base_dir / "metrics"
        srv_dir = std_metrics_dir / "servers"
        srv_dir.mkdir(parents=True, exist_ok=True)
        std_rounds_csv = srv_dir / "rounds.csv"
        std_fields = [
            "global_round","local_round","server_id",
            "client_test_loss_mean","client_test_accuracy_mean",
            "client_test_loss_std","client_test_loss_ci95_low","client_test_loss_ci95_high",
            "client_test_accuracy_std","client_test_accuracy_ci95_low","client_test_accuracy_ci95_high",
            "server_partition_test_loss","server_partition_test_accuracy",
            "generalization_loss_gap","generalization_accuracy_gap",
            "conv_loss_delta","conv_acc_delta",
            "bytes_up_total","bytes_down_total","comp_time_sec_mean","round_time_sec",
        ]
        std_write_header = not std_rounds_csv.exists()
        with open(std_rounds_csv, "a", newline="") as f:
            std_writer = csv.DictWriter(f, fieldnames=std_fields)
            if std_write_header:
                std_writer.writeheader()
            # Generalization gaps w.r.t. participating client mean
            base_acc = agg_acc if agg_acc is not None else acc_mean
            base_loss = agg_loss if agg_loss is not None else loss_mean
            gen_gap_acc_partitions_minus_clients = None if (base_acc is None or server_partition_acc is None) else (server_partition_acc - base_acc)
            gen_gap_loss_partitions_minus_clients = None if (base_loss is None or server_partition_loss is None) else (server_partition_loss - base_loss)
            std_writer.writerow({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "server_id": self.server_id,
                "client_test_loss_mean": agg_loss,
                "client_test_accuracy_mean": agg_acc,
                "client_test_loss_std": loss_std,
                "client_test_loss_ci95_low": loss_ci_low,
                "client_test_loss_ci95_high": loss_ci_high,
                "client_test_accuracy_std": acc_std,
                "client_test_accuracy_ci95_low": acc_ci_low,
                "client_test_accuracy_ci95_high": acc_ci_high,
                "server_partition_test_loss": server_partition_loss,
                "server_partition_test_accuracy": server_partition_acc,
                "generalization_loss_gap": gen_gap_loss_partitions_minus_clients,
                "generalization_accuracy_gap": gen_gap_acc_partitions_minus_clients,
                "conv_loss_delta": delta_avg_client_loss if delta_avg_client_loss is not None else "",
                "conv_acc_delta": delta_avg_client_acc if delta_avg_client_acc is not None else "",
                "bytes_up_total": getattr(self, "server_bytes_up", 0),
                "bytes_down_total": getattr(self, "server_bytes_down", 0),
                "comp_time_sec_mean": getattr(self, "server_comp_time", 0.0),
                "round_time_sec": getattr(self, "server_round_time", 0.0),
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
