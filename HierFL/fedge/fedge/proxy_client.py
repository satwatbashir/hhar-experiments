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
from flwr.common import NDArrays
from typing import Tuple, Dict, Optional
import atexit
import torch
import toml
import grpc
import logging
from fedge.task import Net, load_data, set_weights, test, get_weights
from fedge.utils import fs

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
    def __init__(self, server_id):
        self.server_id = server_id
        self.proxy_id = os.environ.get("PROXY_ID", f"proxy_{server_id}")
        self._sent_complete = False  # prevent duplicates

        # Locate project root and flat per-server directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        # Current global round (0-indexed)
        global_round = int(os.environ.get("GLOBAL_ROUND", "0"))

        # Flat directory shared with leaf server
        self.base_dir = fs.leaf_server_dir(project_root, server_id)
        models_dir = self.base_dir / "models"

        # Path to model for this global round; fallback to latest model if not present
        model_path = models_dir / f"model_s{server_id}_g{global_round}.pkl"
        if not model_path.exists():
            # find newest pkl in models_dir
            pkl_files = sorted(models_dir.glob("*.pkl"))
            if pkl_files:
                model_path = pkl_files[-1]
        print(f"[{self.proxy_id}] Loading model from {model_path}")

        with open(model_path, "rb") as f:
            # Load both NDArrays list and total samples from pickle
            loaded_data = pickle.load(f)

            # Handle both formats (for backward compatibility)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                self.ndarrays, self.total_examples = loaded_data
                print(f"[{self.proxy_id}] Loaded model and {self.total_examples} total training examples")
            else:
                # Fall back to old format if needed
                self.ndarrays = loaded_data
                self.total_examples = 1  # Default to 1 for backward compatibility
                print(f"[{self.proxy_id}] Loaded model (no sample count available)")

        # Initialize model and validation data for evaluation
        cfg = toml.load(project_root / "pyproject.toml")
        num_servers = cfg["tool"]["flwr"]["hierarchy"]["num_servers"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        set_weights(self.net, self.ndarrays)
        
        # Load server-specific test indices from partition JSON (REQUIRED - no fallback)
        import json
        parts_path = os.environ.get("PARTITIONS_JSON")
        if not parts_path or not Path(parts_path).exists():
            raise RuntimeError(
                "PARTITIONS_JSON is required for proxy client evaluation. "
                "Run orchestrator.py to generate user-based partitions first."
            )
        
        with open(parts_path, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
        
        if str(server_id) not in mapping:
            raise RuntimeError(f"Server {server_id} not found in partitions")
        
        # Flatten all client shards under this server
        server_map = mapping[str(server_id)]  # {"0":[...], "1":[...], ...}
        indices = [idx for lst in server_map.values() for idx in lst]
        _, self.valloader, _ = load_data("hhar", self.server_id, num_servers, indices=indices)

        # Ensure completion CSV row on exit
        atexit.register(self._write_completion_signal)

        # Proxy signals CSV shared with leaf server directory
        self.proxy_signals_csv = self.base_dir / "proxy_signals.csv"
        write_header = not self.proxy_signals_csv.exists()
        with open(self.proxy_signals_csv, "a", newline="") as fcsv:
            import csv
            writer = csv.DictWriter(fcsv, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": global_round,
                "proxy_id": self.proxy_id,
                "server_id": server_id,
                "signal_type": "started",
                "timestamp": time.time(),
            })
        print(f"[{self.proxy_id}] Wrote start row to {self.proxy_signals_csv}")

    def get_parameters(self, config) -> NDArrays:
        print(f"[{self.proxy_id}] Providing leaf server parameters to cloud")
        return self.ndarrays

    def fit(self, parameters, config) -> Tuple[NDArrays, int, dict]:
        # IMPORTANT: Do NOT overwrite the edge-server model with the incoming
        # global parameters. HierFL expects each proxy to *re-send* its current
        # edge model unchanged, along with the local sample count.  This restores
        # correct hierarchical averaging at the cloud layer.
        print(f"[{self.proxy_id}] Received fit request, re-sending edge model with {self.total_examples} samples")
        return self.ndarrays, self.total_examples, {}

    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        # Update model weights from server and run evaluation
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        num_examples = len(self.valloader.dataset)
        print(f"[{self.proxy_id}] Eval -> loss: {loss}, samples: {num_examples}, accuracy: {accuracy}")

        # Return evaluation results first
        metrics = {"accuracy": accuracy}
        results = (loss, num_examples, metrics)

        # Decide whether this evaluation corresponds to the *final* server
        # round executed by the current cloud instance. Flower passes the
        # current round index in `config` (1-indexed).  The orchestrator
        # injects the env-var TOTAL_SERVER_ROUNDS_THIS_CLOUD so that the
        # proxy can compare and only exit after the very last round.

        total_rounds_env = os.environ.get("TOTAL_SERVER_ROUNDS_THIS_CLOUD")
        # Fallback for backward compatibility
        if total_rounds_env is None:
            total_rounds_env = os.environ.get("SERVER_ROUNDS_PER_GLOBAL", "1")

        try:
            total_rounds = int(total_rounds_env)
        except ValueError:
            total_rounds = 1

        current_round = int(config.get("server_round", 0))

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
                print(f"[{self.proxy_id}] Exiting proxy client after evaluation...")
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
        print(f"[{self.proxy_id}] Wrote completion row to {self.proxy_signals_csv}")

def handle_signal(sig, frame):
    """Handle termination signals gracefully"""
    proxy_id = os.environ.get("PROXY_ID", "proxy")
    logger.info(f"[{proxy_id}] Received signal {sig}, shutting down gracefully...")
    # Append completion row on signal
    server_id = int(os.environ.get("SERVER_ID", "0"))
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    proxy_signals_csv = fs.leaf_server_dir(project_root, server_id) / "proxy_signals.csv"
    import csv
    global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
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

    logger.info(f"[{proxy_id}] Starting proxy client for Leaf Server {args.server_id}")
    logger.info(f"[{proxy_id}] Will connect to cloud server at {args.cloud_address}")

    # Pre-load model and dataset once
    client = ProxyClient(args.server_id)
    server_address = args.cloud_address

    # Wait for cloud to be ready: look for cloud_started.signal
    signals_dir = fs.get_signals_dir(Path(__file__).resolve().parent.parent)
    start_signal = signals_dir / "cloud_started.signal"
    wait_secs = 0
    while not start_signal.exists() and wait_secs < args.retry_delay * args.max_retries:
        logger.info(f"[{proxy_id}] Waiting for cloud start signal...")
        time.sleep(1)
        wait_secs += 1

    # Run the client with retry logic (only RPC loop)
    for attempt in range(1, args.max_retries + 1):
        try:
            start_client(
                server_address=server_address,
                client=client.to_client(),
            )
            logger.info(f"[{proxy_id}] Completed communication with cloud server")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                if attempt < args.max_retries:
                    logger.warning(f"[{proxy_id}] Connection failed (attempt {attempt}/{args.max_retries}): {e.details()}")
                    logger.info(f"[{proxy_id}] Retrying in {args.retry_delay} seconds...")
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
