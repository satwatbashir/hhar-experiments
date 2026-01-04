"""
Fedge-Simulation: Hierarchical FL with Clustering - In-Memory Simulation Mode

This orchestrator runs the entire Fedge pipeline in a single process with shared memory,
achieving ~7x speedup over the process-based Fedge-100 implementation.

Architecture:
- 3-level hierarchy: clients -> leaf servers -> cloud
- Dynamic clustering at cloud level using cosine similarity
- SCAFFOLD + FedProx optimization
- Accuracy gate for parameter acceptance

Usage:
    cd Fedge-Simulation/fedge
    SEED=42 python simulation_orchestrator.py
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import toml

# Add fedge module to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedge.task import (
    Net, load_hhar_data, HHARDataset, train, test, get_weights, set_weights,
    NUM_CLASSES, GLOBAL_SEED
)
from fedge.cluster_utils import weight_clustering, flatten_last_layer
from fedge.scaffold_utils import create_scaffold_manager, aggregate_scaffold_controls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration Loading
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = toml.load(PROJECT_ROOT / "pyproject.toml")

# Hierarchy config
HIER_CFG = CONFIG["tool"]["flwr"]["hierarchy"]
NUM_SERVERS = int(HIER_CFG["num_servers"])
CLIENTS_PER_SERVER = list(HIER_CFG["clients_per_server"])
GLOBAL_ROUNDS = int(HIER_CFG["global_rounds"])
SERVER_ROUNDS_PER_GLOBAL = int(HIER_CFG["server_rounds_per_global"])

# Learning parameters
LR_INIT = float(HIER_CFG["lr_init"])
SERVER_LR = float(HIER_CFG["server_lr"])
GLOBAL_LR = float(HIER_CFG["global_lr"])
LOCAL_EPOCHS = int(HIER_CFG["local_epochs"])
BATCH_SIZE = int(HIER_CFG["batch_size"])
EVAL_BATCH_SIZE = int(HIER_CFG["eval_batch_size"])

# Optimization parameters
WEIGHT_DECAY = float(HIER_CFG["weight_decay"])
CLIP_NORM = float(HIER_CFG["clip_norm"])
MOMENTUM = float(HIER_CFG["momentum"])
LR_GAMMA = float(HIER_CFG["lr_gamma"])
PROX_MU = float(HIER_CFG["prox_mu"])
SCAFFOLD_ENABLED = bool(HIER_CFG["scaffold_enabled"])

# Clustering config
CLUSTER_CFG = CONFIG["tool"]["flwr"]["cloud_cluster"]
CLUSTER_ENABLED = bool(CLUSTER_CFG["enable"])
CLUSTER_START_ROUND = int(CLUSTER_CFG["start_round"])
CLUSTER_FREQUENCY = int(CLUSTER_CFG["frequency"])
CLUSTER_TAU = float(CLUSTER_CFG["tau"])

# Accuracy gate
CLUSTER_BETTER_DELTA = float(HIER_CFG["cluster_better_delta"])

# SCAFFOLD warmup: Don't apply SCAFFOLD until model stabilizes
# This prevents NaN with imbalanced data (e.g., HHAR where Server 0 has 5x more data)
# Increased to 5 rounds after NaN explosion observed at round 4 with SEED=43
SCAFFOLD_WARMUP_ROUNDS = 5  # Start SCAFFOLD from round 6

# Seed
SEED = int(os.environ.get("SEED", CONFIG["tool"]["flwr"]["app"]["config"].get("seed", 42)))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logger.info(f"[CONFIG] SEED={SEED}, NUM_SERVERS={NUM_SERVERS}, CLIENTS_PER_SERVER={CLIENTS_PER_SERVER}")
logger.info(f"[CONFIG] GLOBAL_ROUNDS={GLOBAL_ROUNDS}, LOCAL_EPOCHS={LOCAL_EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
logger.info(f"[CONFIG] SCAFFOLD={SCAFFOLD_ENABLED}, PROX_MU={PROX_MU}, CLUSTER_TAU={CLUSTER_TAU}")

# ==============================================================================
# Data Loading and Partitioning
# ==============================================================================

def create_partitions(X: np.ndarray, y: np.ndarray) -> Dict[int, Dict[int, List[int]]]:
    """Create user-based partitions for hierarchical FL.

    Returns:
        Dict mapping server_id -> client_id -> list of sample indices
    """
    from fedge.task import load_hhar_windows_with_users

    # Check for existing partition file
    parts_json = PROJECT_ROOT / "rounds" / f"partitions_seed{SEED}.json"
    if parts_json.exists():
        logger.info(f"Loading existing partitions from {parts_json}")
        with open(parts_json) as f:
            return json.load(f)

    logger.info("Creating new partitions based on user IDs...")

    # Load with user info for partitioning
    _, _, users = load_hhar_windows_with_users()

    # Group indices by user
    user_to_indices = defaultdict(list)
    for idx, user in enumerate(users):
        user_to_indices[user].append(idx)

    unique_users = list(user_to_indices.keys())
    logger.info(f"Found {len(unique_users)} unique users")

    # Assign users to servers and clients
    total_clients = sum(CLIENTS_PER_SERVER)
    users_per_client = len(unique_users) // total_clients

    partitions = {}
    user_idx = 0

    for server_id in range(NUM_SERVERS):
        partitions[str(server_id)] = {}
        for client_id in range(CLIENTS_PER_SERVER[server_id]):
            # Assign users to this client
            client_users = unique_users[user_idx:user_idx + users_per_client]
            user_idx += users_per_client

            # Collect all indices for these users
            client_indices = []
            for user in client_users:
                client_indices.extend(user_to_indices[user])

            partitions[str(server_id)][str(client_id)] = client_indices
            logger.info(f"Server {server_id}, Client {client_id}: {len(client_indices)} samples")

    # Handle remaining users by distributing to last clients
    remaining_users = unique_users[user_idx:]
    if remaining_users:
        for i, user in enumerate(remaining_users):
            server_id = i % NUM_SERVERS
            client_id = i % CLIENTS_PER_SERVER[server_id]
            partitions[str(server_id)][str(client_id)].extend(user_to_indices[user])

    # Save partitions
    parts_json.parent.mkdir(parents=True, exist_ok=True)
    with open(parts_json, "w") as f:
        json.dump(partitions, f)
    logger.info(f"Saved partitions to {parts_json}")

    return partitions


# ==============================================================================
# Simulation Components
# ==============================================================================

class SimulatedClient:
    """In-memory client for simulation mode."""

    def __init__(
        self,
        client_id: str,
        server_id: int,
        train_indices: List[int],
        test_indices: List[int],
        full_dataset: HHARDataset,
        device: torch.device
    ):
        self.client_id = client_id
        self.server_id = server_id
        self.device = device

        # Create data loaders using subset of full dataset
        from torch.utils.data import Subset, DataLoader

        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)

        self.trainloader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        self.testloader = DataLoader(
            test_subset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

        self.num_train = len(train_indices)
        self.num_test = len(test_indices)

        # SCAFFOLD state
        self.scaffold_manager = None
        if SCAFFOLD_ENABLED:
            net = Net().to(self.device)
            self.scaffold_manager = create_scaffold_manager(net)
            # Move control variates to the correct device
            for name in self.scaffold_manager.client_control:
                self.scaffold_manager.client_control[name] = self.scaffold_manager.client_control[name].to(self.device)
            for name in self.scaffold_manager.server_control:
                self.scaffold_manager.server_control[name] = self.scaffold_manager.server_control[name].to(self.device)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
        ref_weights: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train locally and return updated weights."""
        # Create global model (before training) for SCAFFOLD
        global_net = Net()
        set_weights(global_net, parameters)
        global_net.to(self.device)  # Need to be on same device for SCAFFOLD

        # Create local model for training
        net = Net()
        set_weights(net, parameters)
        net.to(self.device)

        # Apply SCAFFOLD if enabled
        if SCAFFOLD_ENABLED and self.scaffold_manager:
            net._scaffold_manager = self.scaffold_manager
            if "server_control" in config and config["server_control"] is not None:
                # Move server control variates to the correct device
                server_control = config["server_control"]
                for name in server_control:
                    server_control[name] = server_control[name].to(self.device)
                self.scaffold_manager.server_control = server_control
            # Also move client control variates to device
            for name in self.scaffold_manager.client_control:
                self.scaffold_manager.client_control[name] = self.scaffold_manager.client_control[name].to(self.device)

        lr = config.get("lr", LR_INIT)
        epochs = config.get("epochs", LOCAL_EPOCHS)
        global_round = config.get("global_round", 0)

        # SCAFFOLD warmup: Only enable after warmup rounds to let model stabilize
        scaffold_active = SCAFFOLD_ENABLED and (global_round > SCAFFOLD_WARMUP_ROUNDS)

        # Train
        loss = train(
            net=net,
            loader=self.trainloader,
            epochs=epochs,
            device=self.device,
            lr=lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            clip_norm=CLIP_NORM,
            prox_mu=config.get("prox_mu", PROX_MU),
            ref_weights=ref_weights,
            global_round=global_round,
            scaffold_enabled=scaffold_active
        )

        # Evaluate
        eval_loss, accuracy = test(net, self.testloader, self.device)

        # Get updated weights
        new_weights = get_weights(net)

        # SCAFFOLD: update client control variate using proper API (only after warmup)
        scaffold_delta = None
        if scaffold_active and self.scaffold_manager:
            # Update client control: c_i = c_i - c_server + (1/K*lr) * (global - local)
            self.scaffold_manager.update_client_control(
                local_model=net,
                global_model=global_net,
                learning_rate=lr,
                local_epochs=epochs,
                clip_value=1.0  # Clip to prevent explosion with imbalanced data
            )
            scaffold_delta = self.scaffold_manager.get_client_control()

        metrics = {
            "train_loss": loss,
            "eval_loss": eval_loss,
            "accuracy": accuracy,
            "num_examples": self.num_train,
            "scaffold_delta": scaffold_delta
        }

        return new_weights, self.num_train, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate on local test data."""
        net = Net()
        set_weights(net, parameters)
        net.to(self.device)

        loss, accuracy = test(net, self.testloader, self.device)

        return loss, self.num_test, {"accuracy": accuracy}


class SimulatedLeafServer:
    """In-memory leaf server for simulation mode."""

    def __init__(
        self,
        server_id: int,
        clients: List[SimulatedClient],
        device: torch.device
    ):
        self.server_id = server_id
        self.clients = clients
        self.device = device

        # Server state
        self.latest_parameters: Optional[List[np.ndarray]] = None
        self.c_global: Optional[Dict[str, torch.Tensor]] = None  # SCAFFOLD server control
        self.prev_accuracy = 0.0

        # Metrics
        self.round_metrics = []

    def run_round(
        self,
        global_round: int,
        initial_parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], Dict]:
        """Run one round of local FL with all clients."""

        # Apply accuracy gate: compare new vs old parameters
        if self.latest_parameters is not None and CLUSTER_BETTER_DELTA > 0:
            # Evaluate incoming parameters
            net = Net()
            set_weights(net, initial_parameters)
            net.to(self.device)

            # Combine all client test data for server-level eval
            total_loss, total_acc, total_samples = 0.0, 0.0, 0
            for client in self.clients:
                loss, num, metrics = client.evaluate(initial_parameters, config)
                total_loss += loss * num
                total_acc += metrics["accuracy"] * num
                total_samples += num

            new_acc = total_acc / total_samples if total_samples > 0 else 0.0

            # Reject if new parameters are worse
            if new_acc < self.prev_accuracy - CLUSTER_BETTER_DELTA:
                logger.info(f"[Server {self.server_id}] Accuracy gate REJECTED: {new_acc:.4f} < {self.prev_accuracy:.4f}")
                initial_parameters = self.latest_parameters
            else:
                self.prev_accuracy = new_acc

        # Prepare config with SCAFFOLD control variates
        client_config = {
            "epochs": LOCAL_EPOCHS,
            "lr": config.get("lr", LR_INIT),
            "prox_mu": config.get("prox_mu", PROX_MU),
            "global_round": global_round,
            "server_control": self.c_global
        }

        # Collect client updates
        client_results = []
        for client in self.clients:
            weights, num_examples, metrics = client.fit(
                parameters=initial_parameters,
                config=client_config,
                ref_weights=initial_parameters
            )
            client_results.append((weights, num_examples, metrics))

        # Aggregate client weights (FedAvg)
        total_examples = sum(num for _, num, _ in client_results)
        aggregated_weights = []

        for layer_idx in range(len(initial_parameters)):
            weighted_sum = np.zeros_like(initial_parameters[layer_idx])
            for weights, num_examples, _ in client_results:
                weighted_sum += weights[layer_idx] * num_examples
            aggregated_weights.append(weighted_sum / total_examples)

        # SCAFFOLD: aggregate control variates using Dict[str, torch.Tensor] format
        if SCAFFOLD_ENABLED:
            scaffold_deltas = [m["scaffold_delta"] for _, _, m in client_results if m.get("scaffold_delta")]
            if scaffold_deltas:
                # Client weights for aggregation
                client_weights = [float(num) / total_examples for _, num, _ in client_results]

                # Aggregate using scaffold_utils function
                self.c_global = aggregate_scaffold_controls(scaffold_deltas, client_weights)

        # Update latest parameters
        self.latest_parameters = aggregated_weights

        # Compute aggregated metrics
        avg_loss = np.mean([m["train_loss"] for _, _, m in client_results])
        avg_accuracy = np.mean([m["accuracy"] for _, _, m in client_results])

        server_metrics = {
            "server_id": self.server_id,
            "global_round": global_round,
            "avg_train_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "num_clients": len(self.clients),
            "total_samples": total_examples
        }

        self.round_metrics.append(server_metrics)
        logger.info(f"[Server {self.server_id}] Round {global_round}: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")

        return aggregated_weights, server_metrics


class CloudAggregator:
    """Cloud-level aggregation with dynamic clustering."""

    def __init__(self, num_servers: int, device: torch.device):
        self.num_servers = num_servers
        self.device = device

        # Clustering state
        self.cluster_map: Dict[int, int] = {}  # server_id -> cluster_id
        self.cluster_parameters: Dict[int, List[np.ndarray]] = {}  # cluster_id -> parameters

        # Metrics
        self.round_metrics = []
        self.cluster_history = []

    def aggregate_and_cluster(
        self,
        server_models: List[Tuple[int, List[np.ndarray], int]],  # (server_id, weights, num_samples)
        global_round: int
    ) -> Dict[int, List[np.ndarray]]:
        """Aggregate server models and perform clustering."""

        # Extract weights and sample counts
        server_ids = [sid for sid, _, _ in server_models]
        weights_list = [w for _, w, _ in server_models]
        sample_counts = [n for _, _, n in server_models]
        total_samples = sum(sample_counts)

        # Step 1: Global aggregation (FedAvg across all servers)
        global_weights = []
        for layer_idx in range(len(weights_list[0])):
            weighted_sum = np.zeros_like(weights_list[0][layer_idx])
            for weights, num_samples in zip(weights_list, sample_counts):
                weighted_sum += weights[layer_idx] * num_samples
            global_weights.append(weighted_sum / total_samples)

        # Step 2: Clustering (if enabled and conditions met)
        should_cluster = (
            CLUSTER_ENABLED and
            global_round >= CLUSTER_START_ROUND and
            (global_round - CLUSTER_START_ROUND) % CLUSTER_FREQUENCY == 0
        )

        if should_cluster:
            logger.info(f"[Cloud] Running clustering at round {global_round}")

            # Run weight-based clustering
            labels, similarity_matrix, tau = weight_clustering(
                server_weights_list=weights_list,
                global_weights=global_weights,
                reference_imgs=None,
                round_num=global_round,
                tau=CLUSTER_TAU
            )

            # Update cluster map
            self.cluster_map = {sid: int(labels[i]) for i, sid in enumerate(server_ids)}

            # Compute cluster-specific averages
            unique_clusters = np.unique(labels)
            self.cluster_parameters = {}

            for cluster_id in unique_clusters:
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_samples = sum(sample_counts[i] for i in cluster_indices)

                cluster_weights = []
                for layer_idx in range(len(weights_list[0])):
                    weighted_sum = np.zeros_like(weights_list[0][layer_idx])
                    for i in cluster_indices:
                        weighted_sum += weights_list[i][layer_idx] * sample_counts[i]
                    cluster_weights.append(weighted_sum / cluster_samples)

                self.cluster_parameters[int(cluster_id)] = cluster_weights

            # Save clustering artifacts
            self._save_cluster_artifacts(global_round, labels, similarity_matrix, server_ids)

            logger.info(f"[Cloud] Clustering result: {len(unique_clusters)} clusters, map={self.cluster_map}")
        else:
            # No clustering: all servers get global model
            self.cluster_map = {sid: 0 for sid in server_ids}
            self.cluster_parameters = {0: global_weights}

        # Compute and log metrics
        avg_accuracy = np.mean([
            m.get("avg_accuracy", 0) for _, _, m in server_models if isinstance(m, dict)
        ]) if any(isinstance(m, dict) for _, _, m in server_models) else 0.0

        metrics = {
            "global_round": global_round,
            "num_clusters": len(self.cluster_parameters),
            "cluster_map": self.cluster_map.copy()
        }
        self.round_metrics.append(metrics)

        return self.cluster_parameters

    def get_cluster_model(self, server_id: int) -> List[np.ndarray]:
        """Get the cluster-specific model for a server."""
        cluster_id = self.cluster_map.get(server_id, 0)
        return self.cluster_parameters.get(cluster_id, list(self.cluster_parameters.values())[0])

    def _save_cluster_artifacts(
        self,
        global_round: int,
        labels: np.ndarray,
        similarity_matrix: np.ndarray,
        server_ids: List[int]
    ):
        """Save clustering artifacts to disk."""
        clusters_dir = PROJECT_ROOT / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster assignments
        assignments = {str(sid): int(labels[i]) for i, sid in enumerate(server_ids)}
        with open(clusters_dir / f"clusters_g{global_round}.json", "w") as f:
            json.dump(assignments, f, indent=2)

        # Save similarity matrix
        sim_df = pd.DataFrame(
            similarity_matrix,
            index=[f"s{i}" for i in server_ids],
            columns=[f"s{i}" for i in server_ids]
        )
        sim_df.to_csv(clusters_dir / f"similarity_matrix_g{global_round}.csv")

        self.cluster_history.append({
            "round": global_round,
            "labels": labels.tolist(),
            "cluster_map": assignments
        })


# ==============================================================================
# Main Orchestrator
# ==============================================================================

class SimulationOrchestrator:
    """Main orchestrator for in-memory hierarchical FL simulation."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Orchestrator] Using device: {self.device}")

        # Load data ONCE
        logger.info("[Orchestrator] Loading HHAR dataset (one-time)...")
        X, y, means, stds = load_hhar_data()
        self.full_dataset = HHARDataset(X, y, normalize=True, means=means, stds=stds)
        logger.info(f"[Orchestrator] Dataset loaded: {len(self.full_dataset)} samples")

        # Create partitions
        self.partitions = create_partitions(X, y)

        # Initialize components
        self._initialize_components()

        # Initialize global model
        self.global_model = Net()
        self.global_weights = get_weights(self.global_model)

        # Metrics collection
        self.run_dir = PROJECT_ROOT / "runs" / f"seed{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []

    def _initialize_components(self):
        """Initialize leaf servers and clients."""
        logger.info("[Orchestrator] Initializing leaf servers and clients...")

        self.leaf_servers: List[SimulatedLeafServer] = []

        for server_id in range(NUM_SERVERS):
            clients = []

            for client_id in range(CLIENTS_PER_SERVER[server_id]):
                # Get partition indices
                indices = self.partitions[str(server_id)][str(client_id)]

                # Split into train/test (80/20)
                n_train = int(0.8 * len(indices))
                np.random.seed(SEED + server_id * 100 + client_id)
                shuffled = np.random.permutation(indices)
                train_idx = shuffled[:n_train].tolist()
                test_idx = shuffled[n_train:].tolist()

                client = SimulatedClient(
                    client_id=f"{server_id}_{client_id}",
                    server_id=server_id,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    full_dataset=self.full_dataset,
                    device=self.device
                )
                clients.append(client)

            server = SimulatedLeafServer(
                server_id=server_id,
                clients=clients,
                device=self.device
            )
            self.leaf_servers.append(server)

            total_train = sum(c.num_train for c in clients)
            total_test = sum(c.num_test for c in clients)
            logger.info(f"[Orchestrator] Server {server_id}: {len(clients)} clients, "
                       f"{total_train} train / {total_test} test samples")

        # Initialize cloud aggregator
        self.cloud = CloudAggregator(NUM_SERVERS, self.device)

    def run_global_round(self, global_round: int) -> Dict:
        """Run one global round of hierarchical FL."""
        round_start = time.time()

        # Step 1: Distribute cluster-specific models to leaf servers
        server_models = []

        for server in self.leaf_servers:
            # Get cluster-specific parameters (or global if no clustering yet)
            if self.cloud.cluster_parameters:
                initial_params = self.cloud.get_cluster_model(server.server_id)
            else:
                initial_params = self.global_weights

            # Run leaf server round
            config = {
                "lr": LR_INIT * (LR_GAMMA ** (global_round - 1)),
                "prox_mu": PROX_MU
            }

            aggregated_weights, metrics = server.run_round(
                global_round=global_round,
                initial_parameters=initial_params,
                config=config
            )

            total_samples = sum(c.num_train for c in server.clients)
            server_models.append((server.server_id, aggregated_weights, total_samples))

        # Step 2: Cloud aggregation + clustering
        self.cloud.aggregate_and_cluster(server_models, global_round)

        # Step 3: Update global weights (for reference)
        total_samples = sum(n for _, _, n in server_models)
        new_global = []
        for layer_idx in range(len(self.global_weights)):
            weighted_sum = np.zeros_like(self.global_weights[layer_idx], dtype=np.float32)
            for _, weights, num_samples in server_models:
                weighted_sum += weights[layer_idx].astype(np.float32) * num_samples
            new_global.append(weighted_sum / total_samples)
        self.global_weights = new_global

        # Compute round metrics
        round_time = time.time() - round_start
        avg_accuracy = np.mean([s.round_metrics[-1]["avg_accuracy"] for s in self.leaf_servers])
        avg_loss = np.mean([s.round_metrics[-1]["avg_train_loss"] for s in self.leaf_servers])

        metrics = {
            "global_round": global_round,
            "avg_accuracy": avg_accuracy,
            "avg_loss": avg_loss,
            "num_clusters": len(self.cloud.cluster_parameters),
            "cluster_map": self.cloud.cluster_map.copy(),
            "round_time": round_time
        }

        self.metrics_history.append(metrics)

        logger.info(f"[Round {global_round}] acc={avg_accuracy:.4f}, loss={avg_loss:.4f}, "
                   f"clusters={len(self.cloud.cluster_parameters)}, time={round_time:.1f}s")

        return metrics

    def run(self):
        """Run the full simulation."""
        logger.info(f"[Orchestrator] Starting simulation: {GLOBAL_ROUNDS} global rounds")
        start_time = time.time()

        for global_round in range(1, GLOBAL_ROUNDS + 1):
            self.run_global_round(global_round)

            # Periodic garbage collection and incremental metrics save
            if global_round % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Save metrics incrementally (prevents data loss on crash)
                self._save_metrics()
                logger.info(f"[Checkpoint] Metrics saved at round {global_round}")

        total_time = time.time() - start_time

        # Save final metrics
        self._save_metrics()

        logger.info(f"[Orchestrator] Simulation complete!")
        logger.info(f"[Orchestrator] Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        logger.info(f"[Orchestrator] Average time per round: {total_time/GLOBAL_ROUNDS:.1f}s")

        final_acc = self.metrics_history[-1]["avg_accuracy"]
        logger.info(f"[Orchestrator] Final accuracy: {final_acc:.4f}")

    def _save_metrics(self):
        """Save all metrics to CSV files."""
        metrics_dir = self.run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save global metrics
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(metrics_dir / "global_rounds.csv", index=False)

        # Save per-server metrics
        for server in self.leaf_servers:
            server_df = pd.DataFrame(server.round_metrics)
            server_df.to_csv(metrics_dir / f"server_{server.server_id}.csv", index=False)

        # Save cluster history
        if self.cloud.cluster_history:
            with open(metrics_dir / "cluster_history.json", "w") as f:
                json.dump(self.cloud.cluster_history, f, indent=2)

        logger.info(f"[Orchestrator] Metrics saved to {metrics_dir}")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Fedge-Simulation: Hierarchical FL with Clustering")
    logger.info("=" * 60)

    orchestrator = SimulationOrchestrator()
    orchestrator.run()
