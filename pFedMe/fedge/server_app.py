"""fedge: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedge.task import Net, get_weights, load_data, set_weights, test
from typing import List, Tuple
import os, csv
import numpy as np
import torch
from .pfedme import pFedMe

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)
# Centralized metrics paths
METRICS_DIR = "metrics"
CLIENTS_CSV = os.path.join(METRICS_DIR, "clients.csv")
ROUNDS_CSV  = os.path.join(METRICS_DIR, "rounds.csv")
CENTRAL_CSV = os.path.join(METRICS_DIR, "centralized_metrics.csv")
strategy: pFedMe  # will be set in server_fn below

# Replace both counters with a single truth from Flower's server_round
CURRENT_ROUND = {"value": 0}

# ---------- CSV helper + round cache ----------
CLIENT_CACHE: dict[int, dict[int, dict]] = {}  # {round: {cid: fit_metrics}}

def _append_csv(path: str, row: dict, header: list[str]) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)
# ----------------------------------------------



def _mean_std_ci(values: list[float]):
    """Calculate mean, std, and 95% CI without SciPy dependency."""
    import math
    n = len(values)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    mu = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    # simple t* lookup for 95% (two-sided); normal approx if n>30
    t95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
           11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086}
    tcrit = t95.get(n-1, 1.96 if n >= 30 else 2.086)  # crude but fine
    half = tcrit * (sd / math.sqrt(n)) if n > 1 else 0.0
    return mu, sd, (mu - half, mu + half)

def aggregate_and_log(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Merge eval metrics with cached fit metrics and write clients.csv + rounds.csv."""
    # Use the real round last observed during aggregate_fit
    round_num = CURRENT_ROUND["value"]
    if round_num <= 0:
        # Fallback if called before fit aggregation
        round_num = max(CLIENT_CACHE.keys(), default=1)

    # Weighted sums for round aggregates
    total_examples = sum(int(n) for n, _ in metrics_list) or 1
    w_acc_sum, w_loss_sum = 0.0, 0.0

    # Pull fit cache for this round
    cache = CLIENT_CACHE.get(round_num, {})

    # Accumulators for comm/comp aggregates
    up_list, dn_list, comp_list = [], [], []

    # 1) Write per-client rows (clients.csv)
    for n, m in metrics_list:
        cid = int(m.get("cid", -1))
        w_acc_sum  += float(m["test_accuracy"]) * int(n)
        w_loss_sum += float(m["test_loss"]) * int(n)

        fit_m = cache.get(cid, {})
        row = {
            "round": round_num,
            "cid": cid,
            # Backward-compat: some plotters expect 'client_id'
            "client_id": cid,
            "num_examples": int(n),
            "train_loss_mean": float(fit_m.get("train_loss_mean", 0.0)),
            "train_accuracy_mean": float(fit_m.get("train_accuracy_mean", 0.0)),
            "upload_bytes": int(fit_m.get("upload_bytes", 0)),
            "download_bytes": int(fit_m.get("download_bytes", 0)),
            "comp_time_sec": float(fit_m.get("comp_time_sec", 0.0)),
            "y_delta_norm": float(fit_m.get("y_delta_norm", 0.0)),
            "c_delta_norm": float(fit_m.get("c_delta_norm", 0.0)),
            "test_loss": float(m["test_loss"]),
            "test_accuracy": float(m["test_accuracy"]),
        }
        _append_csv(
            CLIENTS_CSV,
            row,
            [
                "round","cid","client_id","num_examples",
                "train_loss_mean","train_accuracy_mean",
                "upload_bytes","download_bytes","comp_time_sec",
                "y_delta_norm","c_delta_norm",
                "test_loss","test_accuracy",
            ],
        )

        up_list.append(row["upload_bytes"])
        dn_list.append(row["download_bytes"])
        comp_list.append(row["comp_time_sec"])

    # 2) Write per-round aggregates (rounds.csv) with CI/std
    client_test_accuracy_mean = w_acc_sum / total_examples
    client_test_loss_mean = w_loss_sum / total_examples

    # Calculate CI and std for test metrics
    accs = [float(m["test_accuracy"]) for _, m in metrics_list]
    losses = [float(m["test_loss"]) for _, m in metrics_list]
    
    acc_mu, acc_sd, (acc_lo, acc_hi) = _mean_std_ci(accs)
    loss_mu, loss_sd, (loss_lo, loss_hi) = _mean_std_ci(losses)

    # Simple comm time model (adjust if you like)
    network_speed_mbps = 10.0
    comm_times = [
        (up + dn) / (network_speed_mbps * 1e6 / 8.0) for up, dn in zip(up_list, dn_list)
    ]
    wall_clock_sec = float(np.mean([c + t for c, t in zip(comp_list, comm_times)])) if comp_list else 0.0

    rounds_row = {
        "round": round_num,
        "fit_train_loss_mean": float(np.mean([cache[c]["train_loss_mean"] for c in cache])) if cache else 0.0,
        "fit_train_accuracy_mean": float(np.mean([cache[c]["train_accuracy_mean"] for c in cache])) if cache else 0.0,
        "bytes_up_total": int(np.sum(up_list)) if up_list else 0,
        "bytes_down_total": int(np.sum(dn_list)) if dn_list else 0,
        "comp_time_sec_mean": float(np.mean(comp_list)) if comp_list else 0.0,
        "wall_clock_sec": wall_clock_sec,
        "client_test_loss_mean": client_test_loss_mean,
        "client_test_accuracy_mean": client_test_accuracy_mean,
        "client_test_accuracy_std": acc_sd,
        # Legacy short keys
        "client_test_accuracy_ci95_lo": acc_lo,
        "client_test_accuracy_ci95_hi": acc_hi,
        # Standardized long keys (duplicates)
        "client_test_accuracy_ci95_low": acc_lo,
        "client_test_accuracy_ci95_high": acc_hi,
        "client_test_loss_std": loss_sd,
        # Legacy short keys
        "client_test_loss_ci95_lo": loss_lo,
        "client_test_loss_ci95_hi": loss_hi,
        # Standardized long keys (duplicates)
        "client_test_loss_ci95_low": loss_lo,
        "client_test_loss_ci95_high": loss_hi,
    }
    _append_csv(
        ROUNDS_CSV,
        rounds_row,
        [
            "round",
            "fit_train_loss_mean","fit_train_accuracy_mean",
            "bytes_up_total","bytes_down_total",
            "comp_time_sec_mean","wall_clock_sec",
            "client_test_loss_mean","client_test_accuracy_mean",
            "client_test_accuracy_std",
            # Legacy and standardized variants
            "client_test_accuracy_ci95_lo","client_test_accuracy_ci95_hi",
            "client_test_accuracy_ci95_low","client_test_accuracy_ci95_high",
            "client_test_loss_std",
            "client_test_loss_ci95_lo","client_test_loss_ci95_hi",
            "client_test_loss_ci95_low","client_test_loss_ci95_high",
        ],
    )

    # Return something small for Flower logs
    return {"client_test_accuracy_mean": client_test_accuracy_mean}


def aggregate_fit_metrics(*args, **kwargs) -> Metrics:
    """Cache client fit metrics per round and cid.

    Supports both Flower callback styles:
    1) (server_round, results, failures) with FitRes objects
    2) metrics_list: List[Tuple[int, Metrics]]   (fallback, uses CURRENT_ROUND)
    """
    # Style 1 (preferred): (server_round, results, failures)
    if len(args) == 3 and hasattr(args[1][0][1], "metrics"):
        server_round, results, _failures = args
        CURRENT_ROUND["value"] = int(server_round)
        round_idx = CURRENT_ROUND["value"]
        CLIENT_CACHE.setdefault(round_idx, {})
        for _, fitres in results:
            m = fitres.metrics or {}
            cid = int(m.get("cid", -1))
            if cid >= 0:
                CLIENT_CACHE[round_idx][cid] = {
                    "train_loss_mean": float(m.get("train_loss_mean", 0.0)),
                    "train_accuracy_mean": float(m.get("train_accuracy_mean", 0.0)),
                    "upload_bytes": int(m.get("upload_bytes", 0)),
                    "download_bytes": int(m.get("download_bytes", 0)),
                    "comp_time_sec": float(m.get("comp_time_sec", 0.0)),
                    "y_delta_norm": float(m.get("y_delta_norm", 0.0)),
                    "c_delta_norm": float(m.get("c_delta_norm", 0.0)),
                }
        return {"fit_cache_size": len(CLIENT_CACHE.get(round_idx, {}))}

    # Style 2 (legacy): (metrics_list,)
    elif len(args) == 1 and isinstance(args[0], list):
        round_idx = CURRENT_ROUND["value"]
        CLIENT_CACHE.setdefault(round_idx, {})
        metrics_list: List[Tuple[int, Metrics]] = args[0]
        for _, m in metrics_list:
            cid = int(m.get("cid", -1))
            if cid >= 0:
                CLIENT_CACHE[round_idx][cid] = {
                    "train_loss_mean": float(m.get("train_loss_mean", 0.0)),
                    "train_accuracy_mean": float(m.get("train_accuracy_mean", 0.0)),
                    "upload_bytes": int(m.get("upload_bytes", 0)),
                    "download_bytes": int(m.get("download_bytes", 0)),
                    "comp_time_sec": float(m.get("comp_time_sec", 0.0)),
                    "y_delta_norm": float(m.get("y_delta_norm", 0.0)),
                    "c_delta_norm": float(m.get("c_delta_norm", 0.0)),
                }
        return {"fit_cache_size": len(CLIENT_CACHE.get(round_idx, {}))}

    # Unknown calling convention
    return {"fit_cache_size": 0}

# Centralized convergence tracker
class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc = None
        self.loss_changes = []
        self.acc_changes  = []
    def update(self, round_num, loss, acc) -> dict:
        # Debug convergence inputs
        if np.isnan(loss) or np.isinf(loss):
            print(f"CONVERGENCE ERROR: loss={loss} at round {round_num}")
        if np.isnan(acc) or np.isinf(acc):
            print(f"CONVERGENCE ERROR: acc={acc} at round {round_num}")
            
        if self.prev_loss is None or round_num == 0:
            self.prev_loss, self.prev_acc = loss, acc
            # Return default values for first round
            return {
                "conv_loss_rate": 0.0,
                "conv_acc_rate": 0.0,
                "conv_loss_stability": 0.0,
                "conv_acc_stability": 0.0,
            }
            
        dl = loss - self.prev_loss
        da = acc  - self.prev_acc
        
        # Debug convergence calculations
        if np.isnan(dl) or np.isinf(dl):
            print(f"CONVERGENCE ERROR: dl={dl} (loss={loss}, prev_loss={self.prev_loss})")
        if np.isnan(da) or np.isinf(da):
            print(f"CONVERGENCE ERROR: da={da} (acc={acc}, prev_acc={self.prev_acc})")
            
        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc
        
        # Calculate variance
        loss_var = float(np.var(self.loss_changes)) if len(self.loss_changes) > 1 else 0.0
        acc_var = float(np.var(self.acc_changes)) if len(self.acc_changes) > 1 else 0.0
        
        # Debug variance calculations
        if np.isnan(loss_var) or np.isinf(loss_var):
            print(f"CONVERGENCE ERROR: loss_var={loss_var}, loss_changes={self.loss_changes[-5:]}")
        if np.isnan(acc_var) or np.isinf(acc_var):
            print(f"CONVERGENCE ERROR: acc_var={acc_var}, acc_changes={self.acc_changes[-5:]}")
            
        return {
            "conv_loss_rate":      float(dl),
            "conv_acc_rate":       float(da),
            "conv_loss_stability": loss_var,
            "conv_acc_stability":  acc_var,
        }
ctracker = ConvergenceTracker()


# Centralized evaluation & CSV logging
def evaluate_and_log_central(round_num: int, parameters, config):
    # Use Scaffold-compatible load_data signature
    trainloader, testloader, num_classes = load_data(
        partition_id=0,
        num_partitions=1,
        batch_size=64,
        data_root=config.get("data-root", "hhar"),
        use_watches=config.get("use-watches", True),
        sample_rate_hz=config.get("sample-rate-hz", 50),
        window_seconds=config.get("window-seconds", 2),
        window_stride_seconds=config.get("window-stride-seconds", 1),
        num_classes=config.get("num-classes", 6),
    )

    # 2) Infer channels & dims (supports images [B,C,H,W] and sequences [B,C,T])
    sample, _ = next(iter(trainloader))
    if sample.ndim == 4:
        _, in_ch, H, W = sample.shape
        net = Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes)
    elif sample.ndim == 3:
        _, in_ch, T = sample.shape
        net = Net(in_ch=in_ch, n_class=num_classes, seq_len=T)
    else:
        raise ValueError(f"Unsupported input shape: {tuple(sample.shape)}")
    # Accept either list of ndarrays or Flower Parameters
    try:
        nds = parameters if isinstance(parameters, list) else None
        if nds is None:
            from flwr.common import parameters_to_ndarrays as _p2n
            nds = _p2n(parameters)
    except Exception:
        from flwr.common import parameters_to_ndarrays as _p2n
        nds = _p2n(parameters)
    set_weights(net, nds)

    # 4) Send to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 5) Compute centralized train metrics with debugging
    # Centralized evaluation: compute train and test metrics
    train_loss, train_acc = test(net, trainloader, device)
    test_loss, test_acc = test(net, testloader, device)
    # Print summary for this round
    print(f"Round {round_num}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    # Build base record with gaps (with additional nan safety)
    rec = {
        "central_train_loss": float(train_loss),
        "central_train_accuracy": float(train_acc),
        "central_test_loss": float(test_loss),
        "central_test_accuracy": float(test_acc),
        "central_loss_gap": float(test_loss - train_loss),
        "central_accuracy_gap": float(train_acc - test_acc),
    }

    # Add convergence metrics (convergence tracker now handles nan internally)
    conv_metrics = ctracker.update(round_num, test_loss, test_acc)
    rec.update(conv_metrics)

    # Log to CSV with static headers
    fieldnames = [
        "round",
        "central_train_loss",
        "central_train_accuracy",
        "central_test_loss",
        "central_test_accuracy",
        "central_loss_gap",
        "central_accuracy_gap",
        "conv_loss_rate",
        "conv_acc_rate",
        "conv_loss_stability",
        "conv_acc_stability",
    ]
    path = CENTRAL_CSV
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({"round": round_num, **rec})
    return test_loss, rec

# Cluster metrics logger
def server_fn(context: Context):
    global strategy
    # Default now 'cifar10' instead of 'organmix'
    dataset_flag = context.node_config.get("dataset_flag", "hhar")

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_available_clients = context.run_config.get("min_available_clients", 2)
    
    # pFedMe hyperparameters
    lamda = context.run_config.get("lamda", 15.0)
    inner_steps = context.run_config.get("inner_steps", 5)
    outer_steps = context.run_config.get("outer_steps", 1)
    inner_lr = context.run_config.get("inner_lr", 0.01)
    outer_lr = context.run_config.get("outer_lr", 0.01)
    beta = context.run_config.get("beta", 1.0)

    trainloader, testloader, num_classes = load_data(
        partition_id=0,
        num_partitions=1,
        batch_size=1,
        data_root=context.run_config.get("data-root", "hhar"),
        use_watches=context.run_config.get("use-watches", True),
        sample_rate_hz=context.run_config.get("sample-rate-hz", 50),
        window_seconds=context.run_config.get("window-seconds", 2),
        window_stride_seconds=context.run_config.get("window-stride-seconds", 1),
        num_classes=context.run_config.get("num-classes", 6),
    )

    sample, _ = next(iter(trainloader))
    if isinstance(sample, torch.Tensor):
        if sample.ndim == 4:
            _, in_ch, H, W = sample.shape
            init_net = Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes)
        elif sample.ndim == 3:
            _, in_ch, T = sample.shape
            init_net = Net(in_ch=in_ch, n_class=num_classes, seq_len=T)
        else:
            raise ValueError(f"Unsupported input shape: {tuple(sample.shape)}")
    else:
        print("Sample is NOT a tensor! Type:", type(sample))
        print("Sample content:", sample)
        raise ValueError("Sample is not a tensor. Did your transform apply?")

    ndarrays   = get_weights(init_net)
    parameters = ndarrays_to_parameters(ndarrays)


    strategy = pFedMe(
        lamda=lamda,
        inner_steps=inner_steps,
        outer_steps=outer_steps,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        beta=beta,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_and_log,
        # Pass dataset_flag into central evaluate
        evaluate_fn=lambda rnd, params, cfg: evaluate_and_log_central(rnd, params, {**cfg, "dataset_flag": dataset_flag}),  # NEW
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
