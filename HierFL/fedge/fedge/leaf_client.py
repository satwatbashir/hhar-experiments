#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import signal
import warnings
import logging
from pathlib import Path

import torch
import grpc
from flwr.client import NumPyClient, start_client

# Import your task utilities:
from fedge.task import Net, load_data, set_weights, train, test, get_weights

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"):
    logging.getLogger(name).setLevel(logging.ERROR)

# =============================================================================
# Flower NumPyClient implementation
# =============================================================================
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_steps = local_epochs  # Keep internal variable name for compatibility
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_properties(self, config):
        from flwr.common import Properties
        cid = os.environ.get("CLIENT_ID", "")
        return Properties(other={"client_id": cid})

    def get_parameters(self, config):
        # Return the model parameters as numpy arrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """Fit model, record communication and computation metrics"""
        cid = os.environ.get("CLIENT_ID", "")
        # Communication: bytes received
        try:
            bytes_down = sum(arr.nbytes for arr in parameters)
        except:
            bytes_down = 0
        start_time = time.perf_counter()
        # Set model weights
        set_weights(self.net, parameters)
        set_end = time.perf_counter()
        # Local training
        train_loss = train(self.net, self.trainloader, self.local_steps, self.device)
        train_end = time.perf_counter()
        # Get updated weights and compute bytes sent
        weights = get_weights(self.net)
        try:
            bytes_up = sum(arr.nbytes for arr in weights)
        except:
            bytes_up = 0
        end_time = time.perf_counter()
        # Compute times
        comp_time = train_end - set_end
        round_time = end_time - start_time
        # Evaluate on validation set
        eval_loss, accuracy = test(self.net, self.valloader, self.device)
        # Collect metrics
        metrics = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "accuracy": accuracy,
            "bytes_down": bytes_down,
            "bytes_up": bytes_up,
            "comp_time": comp_time,
            "round_time": round_time,
            "client_id": cid
        }
        return weights, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        # Update local model, evaluate on validation set
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        cid = os.environ.get("CLIENT_ID", "")
        metrics = {"accuracy": accuracy, "client_id": cid}
        return loss, len(self.valloader.dataset), metrics

# =============================================================================
# Signal handler for graceful shutdown
# =============================================================================
def handle_signal(sig, frame):
    client_id = os.environ.get("CLIENT_ID", "leaf_client")
    logger.info(f"[{client_id}] Received signal {sig}, shutting down gracefully...")
    sys.exit(0)

# =============================================================================
# Main: parse arguments, load HHAR shard, start Flower client
# =============================================================================
def main():
    # 1) Catch SIGINT/SIGTERM so we can clean up nicely
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", type=int, required=True)
    parser.add_argument("--num_partitions", type=int, required=True)
    # Dataset flag for HHAR
    parser.add_argument(
        "--dataset_flag",
        type=str,
        choices=["hhar"],
        required=True,
        help="Dataset to load (hhar)",
    )
    parser.add_argument("--local_epochs", type=int, required=True)
    parser.add_argument("--server_addr", type=str, default=os.getenv("LEAF_ADDRESS", "127.0.0.1:6100"))
    parser.add_argument(
        "--max_retries", type=int, default=5, help="Max gRPC connection retries"
    )
    parser.add_argument(
        "--retry_delay", type=int, default=2, help="Seconds between retry attempts"
    )
    args = parser.parse_args()

    # 2) Build a human-readable CLIENT_ID for logging (e.g. "leaf_0_client_3")
    client_id = f"leaf_{os.environ.get('SERVER_ID','?')}_client_{args.partition_id}"
    os.environ["CLIENT_ID"] = client_id

    # 3) Determine indices from PARTITIONS_JSON (if provided) else fallback to local Dirichlet
    indices = None
    parts_path = os.environ.get("PARTITIONS_JSON")
    if parts_path and Path(parts_path).exists():
        with open(parts_path, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
        sid = int(os.environ.get("SERVER_ID", "0"))
        indices = mapping[str(sid)][str(args.partition_id)]

    trainloader, valloader, n_classes = load_data(
        args.dataset_flag,
        args.partition_id,
        args.num_partitions,
        indices=indices,
    )

    # 4) Instantiate Net for 1D HHAR inputs (B, C, T)
    sample, _ = next(iter(trainloader))
    # Optional guard in case a singleton dim sneaks in:
    if sample.ndim == 4 and sample.shape[1] == 1:
        sample = sample.squeeze(1)
    _, in_ch, T = sample.shape
    net = Net(in_ch=in_ch, seq_len=T, num_classes=n_classes)

    # 5) Wrap in the FlowerClient
    client = FlowerClient(net, trainloader, valloader, args.local_epochs)

    # 6) Connect (with retry logic) to the leaf server’s Flower endpoint
    retries = 0
    while retries < args.max_retries:
        try:
            logger.info(f"[{client_id}] Connecting to server at {args.server_addr}")
            start_client(server_address=args.server_addr, client=client.to_client())
            logger.info(f"[{client_id}] Client session completed successfully")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE and retries < args.max_retries:
                retries += 1
                logger.warning(
                    f"[{client_id}] Connection failed (attempt {retries}/{args.max_retries}): {e.details()}"
                )
                logger.info(f"[{client_id}] Retrying in {args.retry_delay}s …")
                time.sleep(args.retry_delay)
            else:
                logger.error(f"[{client_id}] Unexpected gRPC error: {e.details()}")
                return
        except Exception as e:
            logger.error(f"[{client_id}] Unexpected error: {e}")
            return

if __name__ == "__main__":
    main()
