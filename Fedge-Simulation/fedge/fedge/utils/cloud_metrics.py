"""
Cloud-level metrics collection for hierarchical federated learning.
Tracks cluster composition, performance, and statistical measures.
"""
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
import torch

class CloudMetricsCollector:
    """Collects and stores cloud-level clustering and performance metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.cloud_metrics_dir = self.project_root / "metrics" / "cloud"
        self.cloud_metrics_dir.mkdir(parents=True, exist_ok=True)
        
    def save_cluster_composition(self, global_round: int, cluster_assignments: Dict[str, int]):
        """Save cluster composition for the round."""
        composition_file = self.cloud_metrics_dir / f"cluster_composition_round_{global_round}.json"
        
        # Count servers per cluster
        cluster_counts = {}
        for server_id, cluster_id in cluster_assignments.items():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        composition_data = {
            "global_round": global_round,
            "num_clusters": len(set(cluster_assignments.values())),
            "cluster_assignments": cluster_assignments,
            "servers_per_cluster": cluster_counts,
            "total_servers": len(cluster_assignments)
        }
        
        with open(composition_file, 'w') as f:
            json.dump(composition_data, f, indent=2)
    
    def evaluate_cluster_performance(self, global_round: int, cluster_assignments: Dict[str, int], 
                                   server_models: Dict[int, Any], server_weights: Dict[int, int],
                                   test_loader, device) -> Dict[str, Any]:
        """Evaluate performance of each cluster using server-reported metrics."""
        # In CFL, we don't need centralized evaluation - we use server-reported metrics
        # This is more aligned with the original CFL design
        
        cluster_metrics = {}
        unique_clusters = set(cluster_assignments.values())
        
        # Get the server metrics from the latest round
        for cluster_id in unique_clusters:
            # Get servers in this cluster
            cluster_servers = [int(sid) for sid, cid in cluster_assignments.items() if cid == cluster_id]
            
            # Aggregate metrics from all servers in this cluster
            server_accuracies = []
            server_losses = []
            total_samples = 0
            
            # Read standardized server metrics files
            from pathlib import Path as _P
            for server_id in cluster_servers:
                rounds_csv = (
                    self.project_root
                    / "rounds" / "leaf" / f"server_{server_id}"
                    / "metrics" / "servers" / "rounds.csv"
                )
                if rounds_csv.exists():
                    import pandas as pd
                    try:
                        df = pd.read_csv(rounds_csv)
                        latest = df[df['global_round'] == global_round]
                        if not latest.empty:
                            # Use server's partition evaluation (its own data shard)
                            accuracy = float(latest['server_partition_test_accuracy'].values[0])
                            loss = float(latest['server_partition_test_loss'].values[0])
                            samples = server_weights.get(server_id, 0)
                            server_accuracies.append(accuracy)
                            server_losses.append(loss)
                            total_samples += samples
                    except Exception as e:
                        print(f"Error reading standardized server metrics for server {server_id}: {e}")
            
            # Calculate cluster metrics if we have data
            if server_accuracies:
                cluster_accuracy = sum(server_accuracies) / len(server_accuracies)
                cluster_loss = sum(server_losses) / len(server_losses)
                
                cluster_metrics[cluster_id] = {
                    "servers": cluster_servers,
                    "num_servers": len(cluster_servers),
                    "test_accuracy": cluster_accuracy,  # This is aggregated accuracy from servers
                    "test_loss": cluster_loss,
                    "total_samples": total_samples,
                    "test_samples": total_samples
                }
        
        return cluster_metrics
    
    def calculate_cluster_statistics(self, cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical measures across clusters."""
        if not cluster_metrics:
            return {}
        
        accuracies = [metrics["test_accuracy"] for metrics in cluster_metrics.values()]
        losses = [metrics["test_loss"] for metrics in cluster_metrics.values()]
        
        stats_data = {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracy_ci_lower": np.percentile(accuracies, 2.5),
            "accuracy_ci_upper": np.percentile(accuracies, 97.5),
            "loss_mean": np.mean(losses),
            "loss_std": np.std(losses),
            "loss_ci_lower": np.percentile(losses, 2.5),
            "loss_ci_upper": np.percentile(losses, 97.5),
            "num_clusters": len(cluster_metrics)
        }
        
        return stats_data
    
    def save_cloud_round_metrics(self, global_round: int, cluster_assignments: Dict[str, int],
                                cluster_metrics: Dict[str, Any], global_accuracy: float, 
                                global_loss: float, communication_cost: int = 0):
        """Deprecated CSV writer kept for backward compatibility: now stores only detailed JSON.
        Standardized cloud metrics are written by cloud_flower.py to metrics/cloud/rounds.csv.
        """
        
        # Calculate statistics
        cluster_stats = self.calculate_cluster_statistics(cluster_metrics)
        
        # Calculate generalization gaps (cluster vs global performance)
        cluster_global_gaps = {}
        for cluster_id, metrics in cluster_metrics.items():
            cluster_global_gaps[cluster_id] = {
                "accuracy_gap": metrics["test_accuracy"] - global_accuracy,
                "loss_gap": metrics["test_loss"] - global_loss
            }
        
        # CSV suppressed to avoid duplication; standardized cloud metrics are elsewhere.
        
        # Save detailed cluster metrics to JSON
        detailed_metrics = {
            "global_round": global_round,
            "cluster_assignments": cluster_assignments,
            "cluster_metrics": cluster_metrics,
            "cluster_statistics": cluster_stats,
            "generalization_gaps": cluster_global_gaps,
            "global_performance": {
                "accuracy": global_accuracy,
                "loss": global_loss
            }
        }
        
        detailed_file = self.cloud_metrics_dir / f"detailed_metrics_round_{global_round}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)

def get_cloud_metrics_collector(project_root: Path) -> CloudMetricsCollector:
    """Get or create a cloud metrics collector instance."""
    return CloudMetricsCollector(project_root)
