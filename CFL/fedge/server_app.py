"""fedge: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# Removed unused imports: FedAvg, FedProx
from fedge.task import Net, get_weights, load_data, set_weights, test
from typing import List, Tuple
import os, csv
from statistics import mean
import numpy as np
from scipy import stats
import torch
from .cfl import CFL
# from .scaffold import Scaffold  # unused

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)
strategy: CFL  # will be set in server_fn below

# For tracking distributed metrics rounds (evaluate)
dist_round_counter = {"value": 1}
# For tracking fit metrics rounds
fit_round_counter = {"value": 1}
# Cache of last fit metrics by CID for rollups
last_fit: dict[str, dict] = {}

# Strict schema enforcement during development
def require(m: Metrics, k: str):
    if k not in m:
        raise KeyError(f"Missing '{k}' in client metrics for this round")
    return m[k]

def _vectorize_control(ctrl: List[torch.Tensor]) -> torch.Tensor:
    """
    Safely flatten and concatenate a list of control‐variate tensors.
    Uses torch.flatten to avoid dtype/view issues.
    """
    flat_tensors = [torch.flatten(c) for c in ctrl]
    return torch.cat(flat_tensors, dim=0)

def _compute_ci95(values: list, confidence: float = 0.95):
    """Compute 95% confidence interval using Student-t distribution."""
    if len(values) < 2:
        return None, None
    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # sample std
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_critical * std_val / np.sqrt(n)
    return mean_val - margin, mean_val + margin

def aggregate_evaluate_metrics(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Write per-client, per-cluster, and global CSVs with std dev and 95% CI.

    Flower passes results as List[(num_examples, metrics_dict)]. We must extract
    the client ID from metrics_dict["cid"], not from the first tuple element.
    """
    global dist_round_counter, ctracker

    round_num = dist_round_counter["value"]

    # Build cid -> metrics
    by_cid: dict[str, Metrics] = {}
    for _num, m in metrics_list:
        cid = str(require(m, "cid"))
        _ = int(require(m, "cluster_id"))
        by_cid[cid] = m

    # 1) Write per-client metrics
    os.makedirs("metrics", exist_ok=True)
    client_path = os.path.join("metrics", "clients_metrics.csv")
    client_fields = [
        "round","cid","cluster_id",
        "test_acc","test_loss","train_acc","train_loss","acc_gap","loss_gap",
        "num_test_samples",
        "download_bytes_eval","upload_bytes_eval",
        "download_bytes_fit","upload_bytes_fit","comp_time_fit_sec","comp_time_eval_sec",
    ]
    client_write_header = not os.path.exists(client_path)
    with open(client_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=client_fields)
        if client_write_header:
            w.writeheader()
        for cid, m in sorted(by_cid.items()):
            lf = last_fit.get(cid, {})
            w.writerow({
                "round": round_num,
                "cid": cid,
                "cluster_id": int(require(m, "cluster_id")),
                "test_acc": float(require(m, "test_accuracy")),
                "test_loss": float(require(m, "test_loss")),
                "train_acc": float(require(m, "train_accuracy")),
                "train_loss": float(require(m, "train_loss")),
                "acc_gap": float(require(m, "accuracy_gap")),
                "loss_gap": float(require(m, "loss_gap")),
                "num_test_samples": int(require(m, "num_test_samples")),
                "download_bytes_eval": int(require(m, "download_bytes_eval")),
                "upload_bytes_eval": int(require(m, "upload_bytes_eval")),
                "download_bytes_fit": int(lf.get("download_bytes_fit", 0)),
                "upload_bytes_fit": int(lf.get("upload_bytes_fit", 0)),
                "comp_time_fit_sec": float(lf.get("comp_time_fit_sec", 0.0)),
                "comp_time_eval_sec": float(m.get("comp_time_eval_sec", 0.0)),
            })

    # 2) Write per-cluster rollups with 95% CI
    clusters: dict[int, list[tuple[str, Metrics]]] = {}
    for cid, m in by_cid.items():
        c = int(require(m, "cluster_id"))
        clusters.setdefault(c, []).append((cid, m))

    cluster_path = os.path.join("metrics", "clusters_metrics.csv")
    cluster_fields = [
        "round","cluster_id","size",
        "test_acc_mean","test_acc_std","test_acc_ci95_low","test_acc_ci95_high",
        "test_loss_mean","test_loss_std","test_loss_ci95_low","test_loss_ci95_high",
        "train_acc_mean","train_acc_std","train_loss_mean","train_loss_std",
        "acc_gap_mean","acc_gap_std","loss_gap_mean","loss_gap_std",
        "download_MB_eval_sum","upload_MB_eval_sum",
        "download_MB_fit_sum","upload_MB_fit_sum","comp_time_fit_sec_sum",
    ]
    cl_write_header = not os.path.exists(cluster_path)
    with open(cluster_path, "a", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=cluster_fields)
        if cl_write_header:
            w.writeheader()
        for c, items in sorted(clusters.items()):
            ms = [m for _, m in items]
            tacc  = [float(require(m, "test_accuracy"))  for m in ms]
            tloss = [float(require(m, "test_loss"))      for m in ms]
            tracc = [float(require(m, "train_accuracy")) for m in ms]
            trlos = [float(require(m, "train_loss"))     for m in ms]
            agap  = [float(require(m, "accuracy_gap"))   for m in ms]
            lgap  = [float(require(m, "loss_gap"))       for m in ms]
            dne   = [int(require(m, "download_bytes_eval")) for m in ms]
            upe   = [int(require(m, "upload_bytes_eval"))   for m in ms]
            members = [cid for cid, _ in items]
            dnf = [int(last_fit.get(cid, {}).get("download_bytes_fit", 0)) for cid in members]
            upf = [int(last_fit.get(cid, {}).get("upload_bytes_fit", 0))   for cid in members]
            ctf = [float(last_fit.get(cid, {}).get("comp_time_fit_sec", 0.0)) for cid in members]

            # Compute 95% CI for test accuracy and test loss
            tacc_ci_low, tacc_ci_high = _compute_ci95(tacc)
            tloss_ci_low, tloss_ci_high = _compute_ci95(tloss)

            w.writerow({
                "round": round_num, "cluster_id": c, "size": len(items),
                "test_acc_mean": (np.mean(tacc) if tacc else 0.0),
                "test_acc_std": (np.std(tacc, ddof=1) if len(tacc) > 1 else 0.0),
                "test_acc_ci95_low": (tacc_ci_low if tacc_ci_low is not None else 0.0),
                "test_acc_ci95_high": (tacc_ci_high if tacc_ci_high is not None else 0.0),
                "test_loss_mean": (np.mean(tloss) if tloss else 0.0),
                "test_loss_std": (np.std(tloss, ddof=1) if len(tloss) > 1 else 0.0),
                "test_loss_ci95_low": (tloss_ci_low if tloss_ci_low is not None else 0.0),
                "test_loss_ci95_high": (tloss_ci_high if tloss_ci_high is not None else 0.0),
                "train_acc_mean": (np.mean(tracc) if tracc else 0.0),
                "train_acc_std": (np.std(tracc, ddof=1) if len(tracc) > 1 else 0.0),
                "train_loss_mean": (np.mean(trlos) if trlos else 0.0),
                "train_loss_std": (np.std(trlos, ddof=1) if len(trlos) > 1 else 0.0),
                "acc_gap_mean": (np.mean(agap) if agap else 0.0),
                "acc_gap_std": (np.std(agap, ddof=1) if len(agap) > 1 else 0.0),
                "loss_gap_mean": (np.mean(lgap) if lgap else 0.0),
                "loss_gap_std": (np.std(lgap, ddof=1) if len(lgap) > 1 else 0.0),
                "download_MB_eval_sum": sum(dne)/(1024*1024),
                "upload_MB_eval_sum":   sum(upe)/(1024*1024),
                "download_MB_fit_sum":  sum(dnf)/(1024*1024),
                "upload_MB_fit_sum":    sum(upf)/(1024*1024),
                "comp_time_fit_sec_sum": sum(ctf),
            })

    # 3) Write global per-round summary with convergence tracking
    if by_cid:
        # Flatten all clients across clusters
        all_tacc  = [float(require(m, "test_accuracy"))  for m in by_cid.values()]
        all_tloss = [float(require(m, "test_loss"))      for m in by_cid.values()]
        all_tracc = [float(require(m, "train_accuracy")) for m in by_cid.values()]
        all_trlos = [float(require(m, "train_loss"))     for m in by_cid.values()]
        all_agap  = [float(require(m, "accuracy_gap"))   for m in by_cid.values()]
        all_lgap  = [float(require(m, "loss_gap"))       for m in by_cid.values()]

        # Global means for convergence tracking
        global_test_acc_mean = np.mean(all_tacc)
        global_test_loss_mean = np.mean(all_tloss)
        
        # Get convergence metrics
        conv_metrics = ctracker.update(round_num, global_test_loss_mean, global_test_acc_mean)

        # Compute 95% CI for global metrics
        tacc_ci_low, tacc_ci_high = _compute_ci95(all_tacc)
        tloss_ci_low, tloss_ci_high = _compute_ci95(all_tloss)

        rounds_path = os.path.join("metrics", "rounds_metrics.csv")
        rounds_fields = [
            "round","clients_evaluated",
            "test_acc_mean","test_acc_std","test_acc_ci95_low","test_acc_ci95_high",
            "test_loss_mean","test_loss_std","test_loss_ci95_low","test_loss_ci95_high",
            "train_acc_mean","train_acc_std","train_loss_mean","train_loss_std",
            "acc_gap_mean","acc_gap_std","loss_gap_mean","loss_gap_std",
            "conv_acc_rate","conv_loss_rate","conv_acc_stability","conv_loss_stability",
        ]
        rounds_write_header = not os.path.exists(rounds_path)
        with open(rounds_path, "a", newline="") as rf:
            w = csv.DictWriter(rf, fieldnames=rounds_fields)
            if rounds_write_header:
                w.writeheader()
            w.writerow({
                "round": round_num,
                "clients_evaluated": len(by_cid),
                "test_acc_mean": global_test_acc_mean,
                "test_acc_std": (np.std(all_tacc, ddof=1) if len(all_tacc) > 1 else 0.0),
                "test_acc_ci95_low": (tacc_ci_low if tacc_ci_low is not None else 0.0),
                "test_acc_ci95_high": (tacc_ci_high if tacc_ci_high is not None else 0.0),
                "test_loss_mean": global_test_loss_mean,
                "test_loss_std": (np.std(all_tloss, ddof=1) if len(all_tloss) > 1 else 0.0),
                "test_loss_ci95_low": (tloss_ci_low if tloss_ci_low is not None else 0.0),
                "test_loss_ci95_high": (tloss_ci_high if tloss_ci_high is not None else 0.0),
                "train_acc_mean": (np.mean(all_tracc) if all_tracc else 0.0),
                "train_acc_std": (np.std(all_tracc, ddof=1) if len(all_tracc) > 1 else 0.0),
                "train_loss_mean": (np.mean(all_trlos) if all_trlos else 0.0),
                "train_loss_std": (np.std(all_trlos, ddof=1) if len(all_trlos) > 1 else 0.0),
                "acc_gap_mean": (np.mean(all_agap) if all_agap else 0.0),
                "acc_gap_std": (np.std(all_agap, ddof=1) if len(all_agap) > 1 else 0.0),
                "loss_gap_mean": (np.mean(all_lgap) if all_lgap else 0.0),
                "loss_gap_std": (np.std(all_lgap, ddof=1) if len(all_lgap) > 1 else 0.0),
                "conv_acc_rate": conv_metrics.get("conv_acc_rate", 0.0),
                "conv_loss_rate": conv_metrics.get("conv_loss_rate", 0.0),
                "conv_acc_stability": conv_metrics.get("conv_acc_stability", 0.0),
                "conv_loss_stability": conv_metrics.get("conv_loss_stability", 0.0),
            })
        
        mean_acc, mean_loss = global_test_acc_mean, global_test_loss_mean
    else:
        mean_acc, mean_loss = 0.0, 0.0

    dist_round_counter["value"] += 1
    return {"avg_test_accuracy": mean_acc, "avg_test_loss": mean_loss, "round": round_num}


def aggregate_fit_metrics(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Cache last per-CID fit metrics for rollups and return a summary.

    Flower passes results as List[(num_train_samples, metrics_dict)]. We must
    extract the client ID from metrics_dict["cid"].
    """
    global fit_round_counter

    round_num = fit_round_counter["value"]
    # Cache last fit metrics by CID
    trained = 0
    for num_train, m in metrics_list:
        cid = str(require(m, "cid"))
        _ = int(require(m, "cluster_id"))
        last_fit[cid] = {
            "download_bytes_fit": int(require(m, "download_bytes_fit")),
            "upload_bytes_fit":   int(require(m, "upload_bytes_fit")),
            "comp_time_fit_sec":  float(m.get("comp_time_fit_sec", m.get("comp_time_sec", 0.0))),
            "num_train_samples":  int(m.get("num_train_samples", num_train)),
            "cluster_id":         int(m.get("cluster_id")),
        }
        trained += 1

    fit_round_counter["value"] += 1
    return {"trained_clients": trained, "round": round_num}

# Centralized convergence tracker
class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc = None
        self.loss_changes = []
        self.acc_changes  = []
    def update(self, round_num, loss, acc) -> dict:
        if self.prev_loss is None or round_num == 0:
            self.prev_loss, self.prev_acc = loss, acc
            return {}
        dl = loss - self.prev_loss
        da = acc  - self.prev_acc
        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc
        return {
            "conv_loss_rate":      float(dl),
            "conv_acc_rate":       float(da),
            "conv_loss_stability": float(np.var(self.loss_changes)),
            "conv_acc_stability":  float(np.var(self.acc_changes)),
        }
ctracker = ConvergenceTracker()
dataset_flag = "hhar"
# Removed stray print


# Centralized evaluation & CSV logging
# Removed evaluate_and_log_central for strict CFL (no centralized eval)
# Client-side evaluation already reflects cluster models

# Cluster metrics logger
def server_fn(context: Context):
    global strategy
    # inside server_fn(context: Context)
    num_rounds = int(context.run_config.get("num-server-rounds", 100))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    batch_size = int(context.run_config.get("batch_size", 64))
    data_root = context.run_config.get("data-root", "hhar")
    use_watches = context.run_config.get("use-watches", True)
    sample_rate_hz = int(context.run_config.get("sample-rate-hz", 50))
    window_seconds = int(context.run_config.get("window-seconds", 2))
    window_stride_seconds = int(context.run_config.get("window-stride-seconds", 1))
    num_classes = int(context.run_config.get("num-classes", 6))

    from fedge.task import load_data, Net, get_weights
    trainloader, _, n_classes = load_data(
        partition_id=0, num_partitions=1, batch_size=1,
        data_root=data_root, use_watches=use_watches, sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds, window_stride_seconds=window_stride_seconds,
        num_classes=num_classes
    )
    sample, _ = next(iter(trainloader))
    _, in_ch, T = sample.shape
    net = Net(in_ch=in_ch, seq_len=T, num_classes=n_classes)
    parameters = ndarrays_to_parameters(get_weights(net))


    strategy = CFL(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=1,
        min_available_clients=2,
        initial_parameters=parameters,
        # CFL-specific parameters

        eps_1=0.4,
        eps_2=1.6,
        min_cluster_size=1,
        # Callbacks
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        # evaluate_fn removed for strict CFL (no centralized eval)
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
