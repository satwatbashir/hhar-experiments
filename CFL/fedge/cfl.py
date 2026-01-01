import os
import csv
import numpy as np
import torch
import time


from flwr.server.strategy import FedAvg
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
    FitIns,
    EvaluateIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager




class CFL(FedAvg):
    """
    Recursive Clustered Federated Learning (CFL) strategy.

    - Maintains per-cluster models and assigns clients by CID.
    - Applies sign-corrected, size-weighted updates to cluster models.
    - Splits clusters via minimax bipartition with single-round ε₁/ε₂ criteria.
    - Returns (None, {}) from aggregate_fit (no centralized global model).
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        eps_1: float = 0.4,
        eps_2: float = 1.6,
        min_cluster_size: int = 2,
        gamma_max: float = 0.05,

        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.min_cluster_size = min_cluster_size

        # --- Persistent state ---
        self.cid_to_cluster: dict[str, int] = {}        # e.g., {"cid-001": 0, "cid-007": 1, ...}
        self.clusters: dict[int, set[str]] = {0: set()} # start with one cluster (0)
        self.cluster_models: dict[int, list[torch.Tensor]] = {}  # θ per cluster
        self.client_models: dict[str, list[torch.Tensor]] = {}   # optional per-client copy
        self.gamma_max = gamma_max
        self.client_num_examples: dict[str, int] = {}
        self._round_t0 = None
        self._num_splits_this_round = 0
        self._last_num_clients_trained: int = 0
        self._last_num_clients_evaluated: int = 0

    def _new_cluster_stats(self):
        # Deprecated: kept for backward-compatibility; no smoothing state needed.
        return {}

    def _cluster_signal(self, triplets):
        vecs = [torch.cat([d.flatten() for d in delta]) for _, _, delta in triplets]
        if not vecs:
            return 0.0, 0.0
        N = sum(n for _, n, _ in triplets)
        wmean = torch.zeros_like(vecs[0])
        for (v, (_, n, _)) in zip(vecs, triplets):
            wmean += (n / N) * v
        mean_norm = wmean.norm().item()
        max_norm = max(v.norm().item() for v in vecs)
        return mean_norm, max_norm

    def _vectors_from_triplets(self, triplets):
        return [torch.cat([d.flatten() for d in delta]) for _, _, delta in triplets]

    def _cosine_sim_matrix(self, vecs: list[torch.Tensor]) -> np.ndarray:
        X = torch.stack(vecs)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        S = (X @ X.t()).cpu().numpy()
        np.fill_diagonal(S, 1.0)
        return S

    def _optimal_bipartition(self, S: np.ndarray):
        """
        Paper-faithful O(M^3) bipartition (Algorithm 1).
        Input: cosine similarity matrix S in [-1,1], diagonal=1.
        Returns: (A_idx, B_idx, alpha_max_cross)
        """
        M = S.shape[0]
        # Start with singletons
        clusters: list[set[int]] = [set([i]) for i in range(M)]
        # Precompute pairwise similarities for complete-link updates
        def inter_sim(a: set[int], b: set[int]) -> float:
            aa = list(a)
            bb = list(b)
            if not aa or not bb:
                return -1.0
            return float(S[np.ix_(aa, bb)].max())

        # While more than two clusters, merge the most similar pair (complete-linkage)
        while len(clusters) > 2:
            best = None
            best_pair = (-1, -1)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = inter_sim(clusters[i], clusters[j])
                    if (best is None) or (sim > best):
                        best = sim
                        best_pair = (i, j)
            i, j = best_pair
            merged = clusters[i] | clusters[j]
            # Remove j first (higher index), then i
            for idx in sorted([i, j], reverse=True):
                clusters.pop(idx)
            clusters.append(merged)

        A, B = sorted(list(clusters[0])), sorted(list(clusters[1]))
        alpha_max_cross = S[np.ix_(A, B)].max() if A and B else -1.0
        return A, B, float(alpha_max_cross)

    def _try_split_cluster(self, c: int, triplets):
        # Strict: require full participation in this round
        members = list(self.clusters[c])
        round_cids = [cid for cid, _, _ in triplets]
        if set(round_cids) != set(members):
            return None
        # Build S from current round only
        vecs = [torch.cat([d.flatten() for d in delta]) for _, _, delta in triplets]
        S = self._cosine_sim_matrix(vecs)
        A_idx, B_idx, amax = self._optimal_bipartition(S)
        def mean_offdiag(M):
            if len(M) <= 1:
                return 1.0
            mask = ~np.eye(len(M), dtype=bool)
            return M[mask].mean() if mask.any() else 1.0
        gap = min(mean_offdiag(S[np.ix_(A_idx, A_idx)]), mean_offdiag(S[np.ix_(B_idx, B_idx)])) - amax
        idx_to_cid = [cid for cid, _, _ in triplets]
        A_cids = {idx_to_cid[i] for i in A_idx}
        B_cids = {idx_to_cid[i] for i in B_idx}
        return A_cids, B_cids, amax, gap



    def _perform_split_full(self, c_parent: int, A_cids: set[str], B_cids: set[str]):
        new_c1 = max(self.clusters.keys()) + 1
        new_c2 = new_c1 + 1
        parent_model = self.cluster_models[c_parent]
        self.cluster_models[new_c1] = [t.clone() for t in parent_model]
        self.cluster_models[new_c2] = [t.clone() for t in parent_model]
        self.clusters[new_c1] = set(A_cids)
        self.clusters[new_c2] = set(B_cids)
        del self.clusters[c_parent]
        del self.cluster_models[c_parent]
        for cid in A_cids:
            self.cid_to_cluster[cid] = new_c1
            self.client_models[cid] = [t.clone() for t in self.cluster_models[new_c1]]
        for cid in B_cids:
            self.cid_to_cluster[cid] = new_c2
            self.client_models[cid] = [t.clone() for t in self.cluster_models[new_c2]]
        return new_c1, new_c2

    def _log_split_event(self, server_round, parent_cluster, A_cids, B_cids, amax, gamma_ok, mean_norm, max_norm, child1, child2):
        """Log split events to cluster_events.csv"""
        os.makedirs("metrics", exist_ok=True)
        path = os.path.join("metrics", "cluster_events.csv")
        write_header = not os.path.exists(path)
        
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "round", "parent_cluster", "child1_cluster", "child2_cluster",
                "parent_size", "child1_size", "child2_size", "alpha_max_cross",
                "gamma_threshold", "gamma_ok", "mean_norm", "max_norm"
            ])
            if write_header:
                w.writeheader()
            w.writerow({
                "round": server_round,
                "parent_cluster": parent_cluster,
                "child1_cluster": child1,
                "child2_cluster": child2,
                "parent_size": len(A_cids) + len(B_cids),
                "child1_size": len(A_cids),
                "child2_size": len(B_cids),
                "alpha_max_cross": amax,
                "gamma_threshold": (1.0 - amax) / 2.0,
                "gamma_ok": gamma_ok,
                "mean_norm": mean_norm,
                "max_norm": max_norm,
            })

    def _log_cluster_stats(self, server_round, c, mean_norm, max_norm, amax, gap, sizeA, sizeB, accept):
        # Detailed cluster stats logging
        os.makedirs("metrics", exist_ok=True)
        path = os.path.join("metrics", "cluster_metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "round","cluster_id","size","mean_norm","max_norm",
                "alpha_max_cross","gap","gamma_threshold","gamma_max","sizeA","sizeB","split_accepted"
            ])
            if write_header:
                w.writeheader()
            w.writerow({
                "round": server_round,
                "cluster_id": c,
                "size": len(self.clusters.get(c, set())),
                "mean_norm": mean_norm,
                "max_norm": max_norm,
                "alpha_max_cross": (amax if amax is not None else ""),
                "gap": (gap if gap is not None else ""),
                "gamma_threshold": (((1.0 - amax) / 2.0) if amax is not None else ""),
                "gamma_max": self.gamma_max,
                "sizeA": sizeA,
                "sizeB": sizeB,
                "split_accepted": int(bool(accept)),
            })

    def configure_fit(self, server_round, parameters, client_manager: ClientManager, **kwargs):
        # Start round timer
        self._round_t0 = time.perf_counter()
        self._num_splits_this_round = 0
        
        # Initialize cluster 0 on first call
        if not self.clusters[0]:
            self.clusters[0] = set()
        if 0 not in self.cluster_models:
            nds = parameters_to_ndarrays(parameters)
            self.cluster_models[0] = [torch.tensor(x, dtype=torch.float32) for x in nds]
        

        # Respect fraction_fit
        try:
            num_available = client_manager.num_available()
        except Exception:
            num_available = len(list(client_manager.all()))
        target = max(self.min_fit_clients, int(round(self.fraction_fit * num_available)))
        sample = client_manager.sample(num_clients=target)
        fit_ins_by_client = []
        for client in sample:
            cid = client.cid
            # ensure membership
            if cid not in self.cid_to_cluster:
                self.cid_to_cluster[cid] = 0
                self.clusters[0].add(cid)
            c = self.cid_to_cluster[cid]
            # send the cluster model of this client's cluster
            nds = [t.cpu().numpy() for t in self.cluster_models[c]]
            params = ndarrays_to_parameters(nds)
            cfg = {"cid": cid, "cluster_id": c}
            fit_ins_by_client.append((client, FitIns(parameters=params, config=cfg)))
        return fit_ins_by_client

    def configure_evaluate(self, server_round, parameters, client_manager: ClientManager, **kwargs):
        """Send each selected client its cluster-specific model for evaluation."""
        num_avail = client_manager.num_available()
        sample = client_manager.sample(num_clients=num_avail)

        eval_ins_by_client = []
        for client in sample:
            cid = client.cid
            c = self.cid_to_cluster.get(cid, 0)
            nds = [t.cpu().numpy() for t in self.cluster_models[c]]
            params = ndarrays_to_parameters(nds)
            cfg = {"cid": cid, "cluster_id": c}
            eval_ins_by_client.append((client, EvaluateIns(parameters=params, config=cfg)))
        return eval_ins_by_client

    def aggregate_evaluate(self, server_round, results, failures, **kwargs):
        # Track how many clients returned evaluation results this round
        try:
            self._last_num_clients_evaluated = len(results) if results is not None else 0
        except Exception:
            self._last_num_clients_evaluated = 0
        # Delegate to base implementation (we ignore centralized eval metrics in strict CFL)
        return super().aggregate_evaluate(server_round, results, failures, **kwargs)

    def aggregate_fit(self, server_round, results, failures, **kwargs):
        """Aggregate fit results cluster-wise and decide on splits (strict CFL)."""
        # Group updates by cluster
        by_cluster: dict[int, list[tuple[str,int,list[torch.Tensor]]]] = {}
        for client, fit_res in results:
            cid = client.cid
            c = int(fit_res.metrics["cluster_id"])
            n = int(fit_res.num_examples)
            self.client_num_examples[cid] = n
            delta = [torch.tensor(x) for x in parameters_to_ndarrays(fit_res.parameters)]
            by_cluster.setdefault(c, []).append((cid, n, delta))

        # Per-cluster aggregation and strict split decision (single round)
        for c, triplets in by_cluster.items():
            if not triplets:
                continue
            N = sum(n for _, n, _ in triplets)
            avg_delta = []
            for p in range(len(triplets[0][2])):
                s = torch.zeros_like(triplets[0][2][p])
                for _, n, delta in triplets:
                    s += (n / N) * delta[p]
                avg_delta.append(s)
            # update cluster model
            for p, d in enumerate(avg_delta):
                self.cluster_models[c][p] -= d
            # reset client copies
            for cid, _, _ in triplets:
                self.client_models[cid] = [t.clone() for t in self.cluster_models[c]]
            # single-round split criteria
            mean_norm, max_norm = self._cluster_signal(triplets)
            ready_to_consider_split = (mean_norm < self.eps_1) and (max_norm > self.eps_2)
            if ready_to_consider_split:
                cand = self._try_split_cluster(c, triplets)
                if cand is not None:
                    A_cids, B_cids, amax, gap = cand
                    gamma_ok = ((1.0 - amax) / 2.0) > self.gamma_max
                    accept = (gamma_ok and
                              gap is not None and gap > 0.0 and         # ← add this line
                              len(A_cids) >= self.min_cluster_size and
                              len(B_cids) >= self.min_cluster_size)
                    self._log_cluster_stats(server_round, c, mean_norm, max_norm, amax, gap,
                                            len(A_cids), len(B_cids), accept)
                    if accept:
                        child1, child2 = self._perform_split_full(c, A_cids, B_cids)
                        self._num_splits_this_round += 1
                        # Log split event
                        self._log_split_event(server_round, c, A_cids, B_cids, amax, gamma_ok, mean_norm, max_norm, child1, child2)
                else:
                    # not enough participation (strict mode requires all members)
                    self._log_cluster_stats(server_round, c, mean_norm, max_norm,
                                            amax=None, gap=None,
                                            sizeA=len(self.clusters[c]), sizeB=0,
                                            accept=False)

        # --- NEW: call fit metrics aggregator so comm/time get cached ---
        if getattr(self, "fit_metrics_aggregation_fn", None) is not None:
            try:
                metrics_list = [(fit_res.num_examples, fit_res.metrics) for (_client, fit_res) in results]
                _ = self.fit_metrics_aggregation_fn(metrics_list)
            except Exception as e:
                print(f"[warn] fit_metrics_aggregation_fn failed: {e}")

        # Log system metrics
        round_wall = time.perf_counter() - self._round_t0 if self._round_t0 else 0.0
        os.makedirs("metrics", exist_ok=True)
        # Number of clients trained this round
        self._last_num_clients_trained = len(results) if results is not None else 0
        with open(os.path.join("metrics", "system_metrics.csv"), "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "round_wall_time_sec",
                    "num_clusters",
                    "num_splits_this_round",
                    "num_clients_trained",
                    "num_clients_evaluated",
                ],
            )
            if f.tell() == 0:
                w.writeheader()
            w.writerow({
                "round": server_round,
                "round_wall_time_sec": round_wall,
                "num_clusters": len(self.clusters),
                "num_splits_this_round": self._num_splits_this_round,
                "num_clients_trained": self._last_num_clients_trained,
                "num_clients_evaluated": self._last_num_clients_evaluated,
            })
        
        # Strict CFL: no centralized representative global model
        return None, {}
        
@torch.no_grad()
def compute_max_diff_norm(diffs: list[list[torch.Tensor]]) -> float:
    if not diffs:
        return 0.0
    def _vec(lst: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.flatten() for t in lst], dim=0) if lst else torch.tensor([])
    return max(_vec(d).norm().item() for d in diffs)


@torch.no_grad()
def compute_mean_diff_norm(diffs: list[list[torch.Tensor]]) -> float:
    if not diffs:
        return 0.0
    def _vec(lst: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.flatten() for t in lst], dim=0) if lst else torch.tensor([])
    stacked = torch.stack([_vec(d) for d in diffs], dim=0)
    return stacked.mean(dim=0).norm().item()
