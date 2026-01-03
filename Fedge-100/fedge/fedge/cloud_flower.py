# cloud_flower.py
"""
Cloud server script for hierarchical federated learning using Flower ≥1.13.
This script starts a Flower ServerApp with communication-cost modifiers,
writes communication metrics to CSV after completion, and handles
dynamic clustering and signal files.
"""

# Suppress warnings
import warnings
import sys
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="flwr")
warnings.simplefilter("ignore")

# Set environment variable to suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import time
import json
import pickle
import logging
import shutil
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import toml
import numpy as np
import argparse

from flwr.server import start_server, ServerConfig
from flwr.common import Metrics, Parameters, NDArrays, FitIns
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from fedge.task import Net, load_data, set_weights, test, get_weights
from fedge.cluster_utils import weight_clustering
from fedge.utils.cloud_metrics import get_cloud_metrics_collector

def create_signal_file(signal_path: Path, message: str) -> None:
    """Create a signal file with the given message."""
    try:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(signal_path, "w") as f:
            f.write(f"{message}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"[Cloud Server] Failed to create signal file {signal_path}: {e}")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class CloudFedAvg(FedAvg):
    """FedAvg strategy with cloud-tier clustering and logging."""
    
    def __init__(self, **kwargs):
        # Extract cloud clustering configuration
        cloud_cluster_config = kwargs.pop('cloud_cluster_config', None)
        # Extract optional model reference for advanced algorithms
        model_ref = kwargs.pop('model', None)
        # Extract clustering configuration dict if provided
        cluster_cfg = kwargs.pop('clustering_config', None)
        
        # Do NOT set evaluate_fn: cloud will not run centralized HHAR evaluation
        super().__init__(**kwargs)
        
        # Cloud clustering configuration - enable clustering from round 1
        ccfg = cloud_cluster_config or {}
        self.cloud_cluster_config = ccfg
        self._cluster_tau = float(ccfg.get("tau", 0.7))
        self._cloud_cluster_enable = bool(ccfg.get("enable", True))  # Default enabled
        self._start_round = int(ccfg.get("start_round", 1))  # Start from round 1
        self._frequency = max(1, int(ccfg.get("frequency", 1)))  # Every round
        self._model = model_ref  # torch.nn.Module instance
        
        # Cloud clustering state
        self._current_round = 0
        self._cluster_log_data = []
        self._cluster_assignments = {}
        self._reference_set = None
        # FEDGE: Server to cluster params mapping for distribution
        self._server_to_cluster_params_map: Dict[int, Parameters] = {}
        self._latest_cluster_map: Dict[int, int] = {}  # server_id -> cluster_id
        # Convergence trackers (previous round values)
        self._prev_avg_srv_loss: float | None = None
        self._prev_avg_srv_acc: float | None = None
        
        if self._cloud_cluster_enable:
            logger.info(f"[Cloud] Clustering enabled: start_round={self._start_round}, frequency={self._frequency}")
        else:
            logger.info(f"[Cloud] Clustering DISABLED in config: {ccfg}")
    
    def _get_server_id_from_fit_result(self, fit_result) -> Optional[int]:
        """Extract server ID from fit result metrics."""
        try:
            metrics = fit_result.metrics
            # Try multiple property names that the proxy client sends
            for prop_name in ["server_id", "sid", "node_id", "client_id", "partition_id"]:
                if prop_name in metrics:
                    return int(metrics[prop_name])
            return None
        except Exception as exc:
            logger.error(f"[Cloud Server] Failed to get server_id from fit result: {exc}")
            return None

    def _get_server_id_from_client_proxy(self, client_proxy) -> Optional[int]:
        """Extract server ID from client proxy's cid."""
        try:
            cid = str(client_proxy.cid)
            # Common patterns: "server_0", "proxy_0", just "0", "s0", etc.
            import re
            match = re.search(r'(\d+)', cid)
            if match:
                return int(match.group(1))
            return None
        except Exception as exc:
            logger.error(f"[Cloud Server] Failed to get server_id from client proxy: {exc}")
            return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """
        FEDGE CRITICAL: Override to distribute cluster-specific parameters.

        Instead of sending the same aggregated params to all servers,
        each server receives the model from its assigned cluster.
        """
        # Get base configuration from parent FedAvg
        fit_configs = super().configure_fit(server_round, parameters, client_manager)

        # Check if we have cluster params to distribute
        if not self._server_to_cluster_params_map:
            logger.info(f"[Cloud] configure_fit: No cluster params available (round {server_round}), using standard params")
            return fit_configs

        # FEDGE: Replace parameters with cluster-specific parameters for each server
        updated_configs = []
        for client_proxy, fit_ins in fit_configs:
            server_id = self._get_server_id_from_client_proxy(client_proxy)

            if server_id is not None and server_id in self._server_to_cluster_params_map:
                # Replace with cluster-specific parameters
                cluster_params = self._server_to_cluster_params_map[server_id]
                cluster_id = self._latest_cluster_map.get(server_id, -1)

                # Create new FitIns with cluster params but keep config
                new_fit_ins = FitIns(parameters=cluster_params, config=fit_ins.config)

                logger.info(f"[Cloud] FEDGE: Server {server_id} receives Cluster {cluster_id} params")
                updated_configs.append((client_proxy, new_fit_ins))
            else:
                # Fallback: use aggregated params (should not happen after round 1)
                logger.warning(f"[Cloud] FEDGE: Server {server_id} not in cluster map, using fallback params")
                updated_configs.append((client_proxy, fit_ins))

        logger.info(f"[Cloud] FEDGE configure_fit: Distributed cluster params to {len(updated_configs)} servers")
        return updated_configs

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results; optionally perform cloud-tier clustering; save models."""
        # Use server_round directly (ignore GLOBAL_ROUND env var)
        global_round = server_round
        
        logger.info(f"[Cloud] aggregate_fit called: server_round={server_round}, global_round={global_round}, results={len(results)}, failures={len(failures)}")
        
        if failures:
            logger.warning(f"[Cloud Server] {len(failures)} servers failed during training")
        
        if not results:
            logger.warning("[Cloud Server] No results to aggregate")
            return super().aggregate_fit(server_round, results, failures)
        
        # 1) Standard FedAvg aggregation across all servers
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Convert Flower Parameters to list[ndarray] for saving/processing
        agg_list = parameters_to_ndarrays(aggregated_params) if aggregated_params is not None else None
        
        # CRITICAL: Validate aggregated parameters for NaN/inf corruption
        if agg_list is not None:
            has_nan = any(np.isnan(arr).any() for arr in agg_list)
            has_inf = any(np.isinf(arr).any() for arr in agg_list)
            
            if has_nan or has_inf:
                logger.error(f"[Cloud] CRITICAL: Aggregated model contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'inf' if has_inf else ''} parameters!")
                logger.error(f"[Cloud] FedAvg aggregation corrupted - using fallback weighted average")
                
                # Fallback: Simple weighted average of server models
                total_samples = sum(fit_result.num_examples for _, fit_result in results)
                if total_samples > 0:
                    weighted_sum = None
                    for _, fit_result in results:
                        weight = fit_result.num_examples / total_samples
                        server_params = parameters_to_ndarrays(fit_result.parameters)
                        
                        if weighted_sum is None:
                            weighted_sum = [weight * arr for arr in server_params]
                        else:
                            for i, arr in enumerate(server_params):
                                weighted_sum[i] += weight * arr
                    
                    # Validate fallback parameters
                    if weighted_sum is not None:
                        fallback_has_nan = any(np.isnan(arr).any() for arr in weighted_sum)
                        fallback_has_inf = any(np.isinf(arr).any() for arr in weighted_sum)
                        
                        if not fallback_has_nan and not fallback_has_inf:
                            logger.info(f"[Cloud] Fallback weighted average is clean - using as replacement")
                            agg_list = weighted_sum
                            aggregated_params = ndarrays_to_parameters(weighted_sum)
                        else:
                            logger.error(f"[Cloud] FATAL: Even fallback parameters are corrupted!")
                            return None, {}
                else:
                    logger.error(f"[Cloud] FATAL: No samples to compute fallback!")
                    return None, {}
            else:
                logger.debug(f"[Cloud] Aggregated parameters validated - no NaN/inf detected")

        # FEDGE: NO global model saving - only cluster models are used
        # The global aggregation is only used as a fallback when clustering fails
        # or for round 1 initialization
        logger.info(f"[Cloud] FEDGE architecture: No global model saved (cluster-only distribution)")

        # 2) Cloud-tier clustering (weight-based) if enabled for this round
        should_cluster = (
            self._cloud_cluster_enable
            and global_round >= self._start_round
            and (global_round - self._start_round) % self._frequency == 0
        )
        
        logger.info(f"[Cloud] Clustering check: enable={self._cloud_cluster_enable}, "
                   f"round={global_round}, start={self._start_round}, freq={self._frequency}, "
                   f"should_cluster={should_cluster}")
        
        if should_cluster:
            logger.info(f"[Cloud] *** CLUSTERING TRIGGERED *** at global round {global_round} "
                        f"(start={self._start_round}, every={self._frequency})")
            
            # Extract server models and weights
            server_models = {}
            server_weights = {}
            
            for client_proxy, fit_result in results:
                server_id = self._get_server_id_from_fit_result(fit_result)
                logger.info(f"[Cloud] Processing result from server_id={server_id}, samples={fit_result.num_examples}")
                if server_id is not None:
                    server_models[server_id] = parameters_to_ndarrays(fit_result.parameters)
                    server_weights[server_id] = fit_result.num_examples
                    logger.info(f"[Cloud] Server {server_id}: {fit_result.num_examples} samples, model params={len(server_models[server_id])}")
                else:
                    logger.warning(f"[Cloud] Failed to extract server_id from fit result")
            
            logger.info(f"[Cloud] Total server models collected: {len(server_models)} (need >=2 for clustering)")
            logger.info(f"[Cloud] Server IDs collected: {list(server_models.keys())}")
            
            if len(server_models) >= 2:
                # Perform clustering using cluster_utils
                try:
                    # Convert server_models dict to list for clustering function
                    server_ids = sorted(server_models.keys())
                    server_weights_list = [server_models[sid] for sid in server_ids]
                    
                    # Read tau from cloud cluster config
                    tau = getattr(self, "_cluster_tau", 0.7)
                    logger.info(f"[Cloud] Using tau = {tau} from config for clustering")
                    
                    # Convert aggregated_params to list if it's Parameters type
                    if aggregated_params is not None:
                        global_weights = parameters_to_ndarrays(aggregated_params)
                    else:
                        global_weights = server_weights_list[0]  # Use first as fallback
                    
                    labels, S, tau_sim = weight_clustering(
                        server_weights_list,
                        global_weights,
                        None,  # reference_imgs not needed for weight-based clustering
                        global_round,  # Use global_round consistently
                        tau
                    )
                    cluster_labels_array = labels
                    
                    # Surface pairwise similarities in cloud logs for debugging
                    for i in range(S.shape[0]):
                        logger.info(f"[Cloud] S[{i}]: " + " ".join(f"{v:.3f}" for v in S[i]))
                    logger.info(f"[Cloud] similarity_threshold used: {tau_sim:.3f}")
                    
                    # Convert cluster labels array to dict mapping server_id -> cluster_label
                    cluster_map = {server_ids[i]: int(cluster_labels_array[i]) for i in range(len(server_ids))}
                    unique_labels = set(cluster_map.values())
                    
                    logger.info(f"[Cloud] Clustering results: {len(unique_labels)} clusters")
                    logger.info(f"[Cloud] Cluster assignments: {cluster_map}")
                    
                    # Gate artifact writes with environment flag
                    write_artifacts = bool(os.getenv("CLOUD_WRITE_ARTIFACTS", "0") == "1")
                    
                    # Define cluster directory for artifact writes
                    cl_dir = Path().resolve() / "clusters"
                    cl_dir.mkdir(parents=True, exist_ok=True)
                        
                    if write_artifacts:
                        # Save clustering metrics
                        metrics_dir = Path().resolve() / "metrics" / "cloud"
                        metrics_dir.mkdir(parents=True, exist_ok=True)
                        cluster_metrics_path = metrics_dir / f"clustering_round_{global_round}.json"
                        with open(cluster_metrics_path, "w") as f:
                            json.dump({
                                "round": global_round,
                                "tau": tau,
                                "num_clusters": len(unique_labels),
                                "assignments": cluster_map,
                                "server_ids": server_ids
                            }, f, indent=2)
                        logger.info(f"[Cloud] Saved clustering metrics to {cluster_metrics_path}")
                            
                        # Save cluster mapping in canonical location and mirror to models/
                        cluster_json_path = cl_dir / f"clusters_g{global_round}.json"
                        with open(cluster_json_path, "w", encoding="utf-8") as fp:
                            json.dump({"round": global_round, "assignments": cluster_map}, fp, indent=2)
                        logger.info(f"[Cloud] Saved cluster mapping to {cluster_json_path}")
                            
                        # Save similarity matrix to file for analysis
                        sim_matrix_path = cl_dir / f"similarity_matrix_g{global_round}.csv"
                        np.savetxt(sim_matrix_path, S, delimiter=',', fmt='%.6f')
                        logger.info(f"[Cloud] Saved similarity matrix to {sim_matrix_path}")
                            
                        # Save pairwise similarities with threshold info
                        pairwise_csv_path = cl_dir / f"pairwise_similarities_g{global_round}.csv"
                        with open(pairwise_csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['server_i', 'server_j', 'similarity', 'above_threshold'])
                            for i in range(len(server_ids)):
                                for j in range(i+1, len(server_ids)):
                                    sim_val = S[i, j]
                                    above_thresh = sim_val >= tau
                                    writer.writerow([server_ids[i], server_ids[j], f'{sim_val:.6f}', above_thresh])
                        logger.info(f"[Cloud] Saved pairwise similarities to {pairwise_csv_path}")
                            
                        # Mirror to models/ directory for compatibility
                        models_dir = Path("models")
                        models_dir.mkdir(exist_ok=True)
                        shutil.copy2(cluster_json_path, models_dir / f"clusters_g{global_round}.json")
                    else:
                        logger.debug(f"[Cloud] Skipping artifact writes (CLOUD_WRITE_ARTIFACTS=0)")
                    
                    # FEDGE CRITICAL: Clear previous cluster params and store new ones
                    self._server_to_cluster_params_map.clear()
                    self._latest_cluster_map = cluster_map.copy()

                    # Weighted average within cluster
                    for lab in unique_labels:
                        cluster_servers = [sid for sid, cluster_id in cluster_map.items() if cluster_id == lab]
                        if not cluster_servers:
                            continue

                        total_weight = sum(server_weights.get(sid, 1) for sid in cluster_servers)
                        sums = None

                        for sid in cluster_servers:
                            if sid in server_models:
                                weight = server_weights.get(sid, 1) / total_weight
                                model_arrays = server_models[sid]

                                if sums is None:
                                    sums = [weight * arr for arr in model_arrays]
                                else:
                                    for i, arr in enumerate(model_arrays):
                                        sums[i] += weight * arr

                        if sums is not None:
                            # FEDGE CRITICAL: Convert to Parameters and store for distribution
                            cluster_params = ndarrays_to_parameters(sums)

                            # Map ALL servers in this cluster to receive this cluster's params
                            for sid in cluster_servers:
                                self._server_to_cluster_params_map[sid] = cluster_params
                                logger.info(f"[Cloud] FEDGE: Server {sid} -> Cluster {lab} params stored for distribution")

                            # Save as plain list for evaluator compatibility
                            cluster_path = cl_dir / f"model_cluster{lab}_g{global_round}.pkl"
                            with open(cluster_path, "wb") as f:
                                pickle.dump(sums, f, protocol=pickle.HIGHEST_PROTOCOL)
                            logger.info(f"[Cloud] Saved cluster {lab} model to {cluster_path}")

                            if write_artifacts:
                                # Mirror to models/ directory
                                models_dir = Path("models")
                                models_dir.mkdir(exist_ok=True)
                                shutil.copy2(cluster_path, models_dir / cluster_path.name)

                    logger.info(f"[Cloud] FEDGE: Cluster params stored for {len(self._server_to_cluster_params_map)} servers")

                    logger.info(f"[Cloud] *** CLUSTERING COMPLETED *** @ global round {global_round}: K={len(unique_labels)} labels={cluster_map}")
                    
                    # Log clustering results for verification
                    for lab in unique_labels:
                        cluster_servers = [sid for sid, label in cluster_map.items() if label == lab]
                        logger.info(f"[Cloud] Cluster {lab}: servers {cluster_servers}")
                    
                except Exception as e:
                    logger.error(f"[Cloud] Clustering failed: {e}, using standard aggregation")
                    import traceback
                    logger.error(f"[Cloud] Clustering error traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[Cloud] Not enough servers for clustering: {len(server_models)} < 2")
        else:
            logger.info(f"[Cloud] Clustering skipped for round {global_round}")

        logger.info(f"[Cloud] Aggregation completed for global round {global_round}")
        
        # Collect cloud-level metrics if clustering was performed
        if should_cluster and len(server_models) >= 2:
            try:
                cloud_metrics_collector = get_cloud_metrics_collector(Path().resolve())
                
                # Save cluster composition
                cloud_metrics_collector.save_cluster_composition(server_round, cluster_map)
                
                # Evaluate cluster performance based on server-reported local-shard metrics only
                cluster_performance = cloud_metrics_collector.evaluate_cluster_performance(
                    server_round, cluster_map, server_models, server_weights, None, None
                )
                # No centralized/global full-dataset evaluation
                global_accuracy = float('nan')
                global_loss = float('nan')
                
                # Calculate communication cost (approximate)
                total_comm_cost = sum(fit_result.num_examples * 4594000 for _, fit_result in results)  # Approximate model size
                
                # Save comprehensive cloud metrics
                cloud_metrics_collector.save_cloud_round_metrics(
                    server_round, cluster_map, cluster_performance, 
                    global_accuracy, global_loss, total_comm_cost
                )
                
                logger.info(f"[Cloud] Saved comprehensive cloud metrics for round {server_round}")
                
            except Exception as e:
                logger.error(f"[Cloud] Failed to collect cloud metrics: {e}")
        
        # Create per-round completion signal for orchestrator synchronization
        signal_dir = Path().resolve() / "signals"
        signal_dir.mkdir(exist_ok=True)
        completion_signal = signal_dir / f"cloud_round_{server_round}_completed.signal"
        create_signal_file(completion_signal, f"Cloud round {server_round} completed")
        logger.info(f"[Cloud] Created completion signal: {completion_signal}")
        
        return aggregated_params, metrics
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and compute agg_acc correctly."""
        try:
            if failures:
                logger.warning(f"[Cloud] Evaluation failures: {len(failures)}")
                return None, {"agg_acc": 0.0}
            
            if not results:
                logger.warning(f"[Cloud] No evaluation results to aggregate")
                return None, {"agg_acc": 0.0}
            
            # Compute weighted average accuracy
            total_examples = 0
            weighted_accuracy_sum = 0.0
            
            for _, eval_res in results:
                num_examples = getattr(eval_res, "num_examples", 0)
                accuracy = (eval_res.metrics or {}).get("accuracy", 0.0)
                
                weighted_accuracy_sum += accuracy * num_examples
                total_examples += num_examples
            
            agg_acc = (weighted_accuracy_sum / total_examples) if total_examples > 0 else 0.0
            
            # Call parent aggregate_evaluate for loss aggregation
            parent_result = super().aggregate_evaluate(server_round, results, failures)
            if parent_result is not None:
                loss, parent_metrics = parent_result
                parent_metrics = parent_metrics or {}
                parent_metrics["agg_acc"] = agg_acc
                # --- Standardized cloud rounds.csv ---
                try:
                    import numpy as _np
                    from scipy import stats as _stats
                except Exception:
                    import numpy as _np
                    _stats = None
                # Arrays across servers
                srv_losses = _np.array([float(ev.loss) for _, ev in results], dtype=float)
                srv_accs   = _np.array([float((ev.metrics or {}).get("accuracy", 0.0)) for _, ev in results], dtype=float)
                # Means/STD
                loss_mean = float(_np.mean(srv_losses)) if srv_losses.size else float("nan")
                acc_mean  = float(_np.mean(srv_accs))   if srv_accs.size   else float("nan")
                loss_std  = float(_np.std(srv_losses, ddof=1)) if srv_losses.size > 1 else 0.0
                acc_std   = float(_np.std(srv_accs, ddof=1))   if srv_accs.size   > 1 else 0.0
                # 95% CI via Student-t when available
                def _ci_pair(mu: float, sd: float, n: int) -> tuple[float, float]:
                    if n <= 1:
                        return float("nan"), float("nan")
                    if _stats is None:
                        # Normal approx
                        half = 1.96 * (sd / (_np.sqrt(n)))
                        return mu - half, mu + half
                    tcrit = _stats.t.ppf(0.975, df=n-1)
                    half = float(tcrit) * (sd / (_np.sqrt(n)))
                    return mu - half, mu + half
                loss_ci_lo, loss_ci_hi = _ci_pair(loss_mean, loss_std, int(srv_losses.size))
                acc_ci_lo,  acc_ci_hi  = _ci_pair(acc_mean,  acc_std,  int(srv_accs.size))

                # FEDGE: No central/global evaluation - only server-local metrics
                # Convergence deltas from previous round (server averages only)
                delta_avg_srv_loss = (
                    loss_mean - self._prev_avg_srv_loss if (self._prev_avg_srv_loss is not None and _np.isfinite(loss_mean)) else float("nan")
                )
                delta_avg_srv_acc = (
                    acc_mean - self._prev_avg_srv_acc if (self._prev_avg_srv_acc is not None and _np.isfinite(acc_mean)) else float("nan")
                )

                # Update prev trackers (server-level only)
                self._prev_avg_srv_loss = loss_mean if _np.isfinite(loss_mean) else None
                self._prev_avg_srv_acc  = acc_mean  if _np.isfinite(acc_mean)  else None

                # Communication placeholders (no fine-grained accounting here)
                bytes_up_total = 0
                bytes_down_total = 0

                cloud_metrics_dir = Path().resolve() / "metrics" / "cloud"
                cloud_metrics_dir.mkdir(parents=True, exist_ok=True)
                rounds_csv = cloud_metrics_dir / "rounds.csv"
                fields = [
                    "global_round",
                    "avg_test_loss_across_servers","std_test_loss_across_servers","ci95_low_test_loss_across_servers","ci95_high_test_loss_across_servers",
                    "avg_test_accuracy_across_servers","std_test_accuracy_across_servers","ci95_low_test_accuracy_across_servers","ci95_high_test_accuracy_across_servers",
                    "delta_avg_loss_across_servers","delta_avg_accuracy_across_servers",
                    "bytes_up_total","bytes_down_total"
                ]
                write_header = not rounds_csv.exists()
                with open(rounds_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    if write_header:
                        w.writeheader()
                    w.writerow({
                        "global_round": server_round,
                        "avg_test_loss_across_servers": loss_mean,
                        "std_test_loss_across_servers": loss_std,
                        "ci95_low_test_loss_across_servers": loss_ci_lo,
                        "ci95_high_test_loss_across_servers": loss_ci_hi,
                        "avg_test_accuracy_across_servers": acc_mean,
                        "std_test_accuracy_across_servers": acc_std,
                        "ci95_low_test_accuracy_across_servers": acc_ci_lo,
                        "ci95_high_test_accuracy_across_servers": acc_ci_hi,
                        "delta_avg_loss_across_servers": delta_avg_srv_loss,
                        "delta_avg_accuracy_across_servers": delta_avg_srv_acc,
                        "bytes_up_total": bytes_up_total,
                        "bytes_down_total": bytes_down_total,
                    })
                return loss, parent_metrics
            else:
                return None, {"agg_acc": agg_acc}
                
        except Exception as e:
            logger.error(f"[Cloud Eval] Evaluation failed: {e}")
            return None, {"agg_acc": 0.0}

def run_server():
    """Run long-running cloud server for all global rounds with dynamic clustering."""
    logger.info("[Cloud Server] Starting long-running cloud aggregation server")
    
    # Create signal directory
    signal_dir = Path().resolve() / "signals"
    signal_dir.mkdir(exist_ok=True)
    
    # Get total number of global rounds from environment
    total_rounds = int(os.getenv('TOTAL_GLOBAL_ROUNDS', '3'))
    logger.info(f"[Cloud Server] Will handle {total_rounds} global rounds")
    
    try:
        # Log environment variables set by orchestrator
        num_servers = os.getenv('NUM_SERVERS', '3')
        logger.info(f"[Cloud Server] Starting for {total_rounds} rounds with {num_servers} servers")
        
        # Load configuration
        config_path = Path("pyproject.toml")
        if not config_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        # Log working directory and models path for path verification
        cwd = Path().resolve()
        logger.info(f"[Cloud Server] Working directory: {cwd}")
        logger.info(f"[Cloud Server] Models directory: {cwd / 'models'}")

        config = toml.load(config_path)
        
        # Extract configuration
        hierarchy_config = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        optimizer_config = config.get("tool", {}).get("flwr", {}).get("optimizer", {})
        cluster_config = config.get("tool", {}).get("flwr", {}).get("cluster", {})
        cloud_cluster_cfg = config.get("tool", {}).get("flwr", {}).get("cloud_cluster", {})
        
        # FEDGE: For round 1, use fresh model. For rounds > 1, cluster params
        # are already stored from previous aggregate_fit() and distributed via configure_fit()
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        init_model = Net()
        initial_parameters = ndarrays_to_parameters(get_weights(init_model))
        logger.info(f"[Cloud Server] FEDGE: Using fresh model for initialization (cluster params managed internally)")

        # Get number of servers from environment (set by orchestrator)
        num_servers = int(os.getenv('NUM_SERVERS', '3'))
        
        # Use CloudFedAvg for HHAR with optional cloud clustering
        strategy = CloudFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_servers,
            min_evaluate_clients=num_servers,
            min_available_clients=num_servers,
            initial_parameters=initial_parameters,
            clustering_config=cluster_config,
            cloud_cluster_config=cloud_cluster_cfg,
            model=init_model,
        )
        # Enforce "all servers must participate"
        strategy.accept_failures = False
        
        # Start server to handle all global rounds
        logger.info(f"[Cloud Server] Starting on port {hierarchy_config.get('cloud_port', 6000)}")
        logger.info(f"[Cloud Server] Expecting {num_servers} proxy clients per round")
        
        # Create start signal file BEFORE starting server
        create_signal_file(signal_dir / "cloud_started.signal", "Cloud server started")
        logger.info("[Cloud Server] Start signal created")
        
        # Start server with proper configuration - handle ALL rounds in one go
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            logger.info(f"[Cloud Server] Starting Flower server for {total_rounds} rounds")
            history = start_server(
                server_address=f"0.0.0.0:{hierarchy_config.get('cloud_port', 6000)}",
                config=ServerConfig(
                    num_rounds=total_rounds,  # Handle ALL rounds in one server instance
                    round_timeout=300.0,  # 5 minutes timeout per round
                ),
                strategy=strategy
            )
        
        logger.info(f"[Cloud Server] All {total_rounds} rounds completed")
        return history
        
    except Exception as e:
        logger.exception(f"[Cloud Server] Unhandled error: {e}")
        create_signal_file(signal_dir / "cloud_error.signal", f"{e!r}")
        raise
    finally:
        # Create final completion signal after all rounds
        create_signal_file(signal_dir / "cloud_all_rounds_completed.signal", "Cloud server completed all rounds")

if __name__ == "__main__":
    run_server()
