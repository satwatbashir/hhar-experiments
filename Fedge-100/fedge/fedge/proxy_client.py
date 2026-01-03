# proxy_client.py
import pickle
import argparse
import os
import time
import warnings
import sys
import signal
from pathlib import Path
from flwr.client import NumPyClient, start_client
# ClientConfig is only available in newer Flower versions; we REQUIRE it for strict node_id.
try:
    from flwr.client import ClientConfig  # type: ignore
except ImportError:  # pragma: no cover
    ClientConfig = None

# Suppress Flower deprecation warnings during client startup
import contextlib
import io
from flwr.common import NDArrays
from typing import Tuple, Dict, Optional
import atexit
import torch
import grpc
import logging
from fedge.task import Net, set_weights, test, get_weights, load_data, NUM_CLASSES
import toml
from fedge.utils import fs
from fedge.utils.bytes_helper import raw_bytes
from fedge.utils.fs_optimized import get_model_path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Suppress Python deprecation warnings in flwr module
import sys, logging, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
# Elevate Flower, ECE, gRPC logger levels to ERROR to hide warning logs
for name in ("flwr", "ece", "grpc"): logging.getLogger(name).setLevel(logging.ERROR)
# Drop printed 'DEPRECATED FEATURE' messages from stdout/stderr
class _DropDeprecated:
    def __init__(self, out): self._out = out
    def write(self, txt):
        if "DEPRECATED FEATURE" not in txt: self._out.write(txt)
    def flush(self): self._out.flush()
sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)

class ProxyClient(NumPyClient):
    """Proxy client that uploads leaf server models to cloud server."""
    
    def __init__(self, server_id):
        self.server_id = server_id
        self.proxy_id = os.environ.get("PROXY_ID", f"proxy_{server_id}")
        self._sent_complete = False  # prevent duplicates

        # Use absolute path to project root (consistent with orchestrator)
        self.project_root = Path(__file__).resolve().parent.parent

        # Current global round (1-indexed for directory structure)
        self.global_round = int(os.environ.get("GLOBAL_ROUND", "1"))

        # Standard directory structure: rounds/round_X/leaf/server_Y/
        self.base_dir = self.project_root / "rounds" / f"round_{self.global_round}" / "leaf" / f"server_{server_id}"
        
        # Use canonical model path helper for consistency
        model_path = get_model_path(self.project_root, server_id, self.global_round)
        if not model_path.exists():
            # Wait (up to 300 s) for the leaf server to save the model
            wait_sec, max_wait = 0, 300
            logger.info(f"[{self.proxy_id}] Waiting for model file {model_path} (max {max_wait}s)")
            while not model_path.exists() and wait_sec < max_wait:
                time.sleep(2)
                wait_sec += 2
            if not model_path.exists():
                logger.error(f"[{self.proxy_id}] Model file not found after {max_wait}s – aborting")
                sys.exit(1)
        logger.info(f"[{self.proxy_id}] Loading model from {model_path}")

        # Try to load the model with error handling for corrupted files
        loaded_data = None
        try:
            # Check if file is not empty
            if model_path.stat().st_size == 0:
                raise ValueError(f"Model file {model_path} is empty")
                
            with open(model_path, "rb") as f:
                # Load both NDArrays list and total samples from pickle
                loaded_data = pickle.load(f)

                # Strict format: expect (ndarrays, total_examples)
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    self.ndarrays, self.total_examples = loaded_data
                    logger.info(f"[{self.proxy_id}] Loaded model with {self.total_examples} total training examples")
                else:
                    raise ValueError("Unexpected model file format; expected (parameters, total_examples)")
                    
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            logger.error(f"[{self.proxy_id}] Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Cannot proceed with corrupted/missing model file {model_path}: {e}") from e

        # Initialize model for evaluation on this server's local shard (union of its clients)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            cfg = toml.load(self.project_root / "pyproject.toml")
            hier = cfg["tool"]["flwr"]["hierarchy"]
            cps_cfg = hier.get("clients_per_server", 3)
            if isinstance(cps_cfg, list):
                self.clients_per_server = int(cps_cfg[self.server_id])
                self.num_servers = int(hier.get("num_servers", len(cps_cfg)))
            else:
                self.clients_per_server = int(cps_cfg)
                self.num_servers = int(hier.get("num_servers", 3))
        except Exception as e:
            logger.warning(f"[{self.proxy_id}] Failed to read clients_per_server from TOML: {e}; defaulting to 3")
            self.clients_per_server = 3
            self.num_servers = 3

        # Build a sample loader from client 0 to infer model shape
        try:
            _, tmp_valloader, _ = load_data(
                "hhar",
                partition_id=0,
                num_partitions=self.num_servers * self.clients_per_server,
                batch_size=128,
                server_id=self.server_id,
            )
            sample, _ = next(iter(tmp_valloader))
            if not isinstance(sample, torch.Tensor):
                raise ValueError(f"Sample is not a tensor: {type(sample)}")
            if sample.ndim == 3:
                _, in_ch, T = sample.shape
                self.net = Net(in_ch=in_ch, n_class=NUM_CLASSES, seq_len=T).to(self.device)
            elif sample.ndim == 4:
                _, in_ch, T, _ = sample.shape
                self.net = Net(in_ch=in_ch, n_class=NUM_CLASSES, seq_len=T).to(self.device)
            else:
                raise ValueError(f"Unsupported input shape for HHAR: {tuple(sample.shape)}")
            set_weights(self.net, self.ndarrays)
        except StopIteration:
            raise RuntimeError("Empty validation loader for local client 0; cannot infer model shape")

        # Ensure completion CSV row on exit
        atexit.register(self._write_completion_signal)

        # Proxy signals CSV in standard leaf server directory
        # Ensure directory exists before writing
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_signals_csv = self.base_dir / "proxy_signals.csv"
        write_header = not self.proxy_signals_csv.exists()
        with open(self.proxy_signals_csv, "a", newline="") as fcsv:
            import csv
            writer = csv.DictWriter(fcsv, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": self.global_round,
                "proxy_id": self.proxy_id,
                "server_id": server_id,
                "signal_type": "started",
                "timestamp": time.time(),
            })
        logger.debug(f"[{self.proxy_id}] Wrote start row to {self.proxy_signals_csv}")


    def get_parameters(self, config) -> NDArrays:
        logger.debug(f"[{self.proxy_id}] Providing leaf server parameters to cloud")
        return self.ndarrays

    def fit(self, parameters, config) -> Tuple[NDArrays, int, dict]:
        import time
        t0 = time.time()
        logger.debug(f"[{self.proxy_id}] Received fit request, re-sending edge model with {self.total_examples} samples")
        bytes_up = raw_bytes(self.ndarrays)
        bytes_down = raw_bytes(parameters)  # size of global model downloaded from cloud
        round_time = time.time() - t0  # negligible but keeps schema consistent
        # Add server metadata for stable clustering
        dataset_flag = "hhar"
        
        return self.ndarrays, self.total_examples, {
            "server_id": self.server_id,
            "total_examples": self.total_examples,
            "bytes_down": bytes_down,
            "round_time": round_time,
            # Server metadata for clustering (flattened to avoid nested dict issues)
            "dataset_flag": dataset_flag,
            "view_type": "hhar_server",
            "specialization": "hhar_dynamic"
        }

    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        # OPTIMIZATION: Use cached metrics from leaf server instead of re-evaluating
        # This avoids redundant computation since leaf server already evaluated on same data
        try:
            import json
            from fedge.utils.fs_optimized import get_model_path
            model_path = get_model_path(self.project_root, self.server_id, self.global_round)
            metrics_path = model_path.with_suffix('.metrics.json')

            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    cached = json.load(f)
                loss = cached.get("loss", float("nan"))
                accuracy = cached.get("accuracy", 0.0)
                num_examples = cached.get("num_examples", 0)
                logger.info(f"[{self.proxy_id}] Using cached metrics from leaf server (saved computation)")
            else:
                # Fallback: evaluate if cache not available
                logger.warning(f"[{self.proxy_id}] No cached metrics found, falling back to evaluation")
                set_weights(self.net, parameters)
                total_n = 0
                loss_sum = 0.0
                acc_sum = 0.0
                for cid in range(self.clients_per_server):
                    _, valloader, _ = load_data(
                        "hhar",
                        partition_id=cid,
                        num_partitions=self.num_servers * self.clients_per_server,
                        batch_size=128,
                        server_id=self.server_id,
                    )
                    l, a = test(self.net, valloader, self.device)
                    n = len(valloader.dataset)
                    total_n += n
                    loss_sum += l * n
                    acc_sum += a * n
                loss = (loss_sum / total_n) if total_n > 0 else float("nan")
                accuracy = (acc_sum / total_n) if total_n > 0 else 0.0
                num_examples = total_n
        except Exception as e:
            logger.error(f"[{self.proxy_id}] Error reading cached metrics: {e}, falling back to evaluation")
            set_weights(self.net, parameters)
            total_n = 0
            loss_sum = 0.0
            acc_sum = 0.0
            for cid in range(self.clients_per_server):
                _, valloader, _ = load_data(
                    "hhar",
                    partition_id=cid,
                    num_partitions=self.num_servers * self.clients_per_server,
                    batch_size=128,
                    server_id=self.server_id,
                )
                l, a = test(self.net, valloader, self.device)
                n = len(valloader.dataset)
                total_n += n
                loss_sum += l * n
                acc_sum += a * n
            loss = (loss_sum / total_n) if total_n > 0 else float("nan")
            accuracy = (acc_sum / total_n) if total_n > 0 else 0.0
            num_examples = total_n

        logger.debug(f"[{self.proxy_id}] Eval -> loss: {loss}, samples: {num_examples}, accuracy: {accuracy}")

        # Return evaluation results first
        bytes_down = raw_bytes(parameters)
        metrics = {"accuracy": accuracy, "bytes_down_eval": bytes_down}
        results = (loss, num_examples, metrics)

        # Decide whether this evaluation corresponds to the *final* server
        # round executed by the current cloud instance. Flower passes the
        # current round index in `config` (1-indexed).  The orchestrator
        # injects the env-var TOTAL_SERVER_ROUNDS_THIS_CLOUD so that the
        # proxy can compare and only exit after the very last round.

        total_rounds_env = os.environ.get("TOTAL_SERVER_ROUNDS_THIS_CLOUD")
        if total_rounds_env is None:
            logger.error(f"[{self.proxy_id}] TOTAL_SERVER_ROUNDS_THIS_CLOUD not set – aborting")
            self._write_completion_signal()
            sys.exit(1)

        try:
            total_rounds = int(total_rounds_env)
        except ValueError:
            logger.error(f"[{self.proxy_id}] TOTAL_SERVER_ROUNDS_THIS_CLOUD invalid: {total_rounds_env}")
            self._write_completion_signal()
            sys.exit(1)

        current_round = int(config.get("server_round", 0))
        
        # Debug: Show exit decision values
        logger.debug(f"[{self.proxy_id}] Round {current_round}/{total_rounds}, should_exit={current_round >= total_rounds}")

        should_exit = current_round >= total_rounds

        if should_exit:
            # Write completion row and exit AFTER returning results
            import threading

            def exit_after_delay():
                # Avoid duplicate signals
                if self._sent_complete:
                    return
                time.sleep(1)  # Ensure response is sent
                self._write_completion_signal()
                import os
                logger.debug(f"[{self.proxy_id}] Exiting proxy client after evaluation...")
                os._exit(0)

            # Start exit thread so we return results first
            threading.Thread(target=exit_after_delay, daemon=True).start()

        # Return results before the thread exits
        return results

    def _write_completion_signal(self):
        import csv
        global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
        write_header = not self.proxy_signals_csv.exists()
        with open(self.proxy_signals_csv, "a", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": global_round,
                "proxy_id": self.proxy_id,
                "server_id": self.server_id,
                "signal_type": "complete",
                "timestamp": time.time(),
            })
        self._sent_complete = True
        logger.debug(f"[{self.proxy_id}] Wrote completion row to {self.proxy_signals_csv}")

def handle_signal(sig, frame):
    """Handle termination signals gracefully"""
    proxy_id = os.environ.get("PROXY_ID", "proxy")
    logger.info(f"[{proxy_id}] Received signal {sig}, shutting down gracefully...")
    # Append completion row on signal
    server_id = int(os.environ.get("SERVER_ID", "0"))
    global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    proxy_signals_csv = fs.leaf_server_dir(project_root, server_id, global_round) / "proxy_signals.csv"
    import csv
    with open(proxy_signals_csv, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
        if fp.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "global_round": global_round,
            "proxy_id": proxy_id,
            "server_id": server_id,
            "signal_type": "complete",
            "timestamp": time.time(),
        })
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--cloud_address", default=os.getenv("CLOUD_ADDRESS", "127.0.0.1:6000"))
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum connection retry attempts")
    parser.add_argument("--retry_delay", type=int, default=2, help="Seconds to wait between retries")
    parser.add_argument("--global_round", type=int, default=0, help="Current global round number (0-indexed)")
    parser.add_argument("--dir_round", type=int, help="Round number for directory structure (1-indexed)")

    args = parser.parse_args()

    # Store server_id in environment for signal handlers
    os.environ["SERVER_ID"] = str(args.server_id)

    # Store global round and dir round if provided via command line
    if args.global_round is not None:
        os.environ["GLOBAL_ROUND"] = str(args.global_round)
    if args.dir_round is not None:
        os.environ["DIR_ROUND"] = str(args.dir_round)

    # Extract ID from environment or use default
    proxy_id = os.environ.get("PROXY_ID", f"proxy_{args.server_id}")
    os.environ["PROXY_ID"] = proxy_id

    # Use the already configured logger from the top of the file
    # (logging is already set up with basicConfig)

    # ------------------------------------------------------------------ #
    # Tell Flower what the deterministic client id should be
    #   (must be a str, NOT an int)
    # ------------------------------------------------------------------ #
    client_node_id = str(args.server_id)
    os.environ["FLWR_CLIENT_NODE_ID"] = client_node_id
    
    logger.info(f"[{proxy_id}] Proxy start: server_id={args.server_id}, cloud={args.cloud_address}, node_id={client_node_id}")

    # Pre-load model and dataset once
    client = ProxyClient(args.server_id)
    server_address = args.cloud_address

    # Wait for cloud to be ready: look for cloud_started.signal
    # Use absolute path to project root for consistency
    project_root = Path(__file__).resolve().parent.parent
    start_signal = project_root / "signals" / "cloud_started.signal"
    wait_secs = 0
    while not start_signal.exists() and wait_secs < args.retry_delay * args.max_retries:
        logger.debug(f"[{proxy_id}] Waiting for cloud start signal...")
        time.sleep(1)
        wait_secs += 1

    # Run the client with retry logic (only RPC loop)
    for attempt in range(1, args.max_retries + 1):
        try:
            # Set strict node_id
            os.environ["FLWR_CLIENT_NODE_ID"] = client_node_id
            
            
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                if ClientConfig is None:
                    logger.warning(f"[{proxy_id}] flwr.client.ClientConfig not available; proceeding without client_config (node_id unenforced by Flower)")
                    start_client(
                        server_address=server_address,
                        client=client,
                    )
                else:
                    start_client(
                        server_address=server_address,
                        client=client,
                        client_config=ClientConfig(node_id=client_node_id)
                    )
            logger.info(f"[{proxy_id}] Proxy finished as Flower client {args.server_id}")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                if attempt < args.max_retries:
                    logger.warning(f"[{proxy_id}] Connection failed (attempt {attempt}/{args.max_retries}): {e.details()}")
                    logger.debug(f"[{proxy_id}] Retrying in {args.retry_delay} seconds...")
                    time.sleep(args.retry_delay)
                    continue
                else:
                    logger.error(f"[{proxy_id}] Connection failed after {args.max_retries} attempts: {e.details()}")
                    client._write_completion_signal()
                    sys.exit(1)
            else:
                logger.error(f"[{proxy_id}] Unexpected gRPC error: {e.details()}")
                client._write_completion_signal()
                sys.exit(1)
        except Exception as e:
            logger.error(f"[{proxy_id}] Unexpected error: {e}")
            client._write_completion_signal()
            sys.exit(1)
    else:
        # This else runs if the loop didn't break
        logger.error(f"[{proxy_id}] Failed to connect after {args.max_retries} attempts")
        client._write_completion_signal()
        sys.exit(1)
