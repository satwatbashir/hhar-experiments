"""fedge: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from fedge.task import Net, get_weights, load_data, load_full_data, set_weights, test
from typing import List, Tuple
import os, csv
import numpy as np
import torch
from scipy.stats import t
import math

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)
strategy: FedProx  # will be set in server_fn below

# For tracking distributed metrics rounds
dist_round_counter = {"value": 1}
# For tracking fit metrics rounds
fit_round_counter = {"value": 1}



def _ci_bounds(mean: float, std: float, n: int) -> tuple[float, float]:
    """95% confidence interval bounds (two-sided)."""
    if n <= 1:
        return (float("nan"), float("nan"))
    tcrit = t.ppf(0.975, df=n-1)   # 97.5% quantile, two-sided 95% CI
    half = tcrit * (std / math.sqrt(n))
    return (mean - half, mean + half)

def aggregate_and_log(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregation callback for distributed (per-client) evaluation.
    Logs per-client metrics to clients.csv and aggregated metrics.
    """
    global dist_round_counter
    round_num = dist_round_counter["value"]
    
    # Print distributed evaluation per-client metrics to console
    print(f"\n=== DISTRIBUTED EVAL (Round {round_num}) ===")
    for idx, (_, m) in enumerate(metrics_list):
        print(f"Client {idx+1}: test_loss={m['test_loss']:.4f}, test_acc={m['test_accuracy']:.4f}")

    # 1) Log per-client metrics to clients.csv
    os.makedirs("metrics", exist_ok=True)
    clients_csv = os.path.join("metrics", "clients.csv")
    write_header = not os.path.exists(clients_csv)
    
    with open(clients_csv, "a", newline="") as f:
        fieldnames = [
            "round", "cid", "client_id", "test_accuracy", "test_loss",
            "train_accuracy", "train_loss", "accuracy_gap", "loss_gap",
            "comp_time_sec", "download_bytes", "upload_bytes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        for idx, (_, m) in enumerate(metrics_list):
            writer.writerow({
                "round": round_num,
                # Standardized id column
                "cid": idx,
                # Backward-compat column name used previously in plots
                "client_id": idx,
                "test_accuracy": m.get("test_accuracy", 0.0),
                "test_loss": m.get("test_loss", 0.0),
                "train_accuracy": m.get("train_accuracy", 0.0),
                "train_loss": m.get("train_loss", 0.0),
                "accuracy_gap": m.get("accuracy_gap", 0.0),
                "loss_gap": m.get("loss_gap", 0.0),
                "comp_time_sec": m.get("comp_time_sec", 0.0),
                "download_bytes": m.get("download_bytes", 0),
                "upload_bytes": m.get("upload_bytes", 0),
            })

    # 2) Compute aggregated metrics with confidence intervals
    accs = [m["test_accuracy"] for _, m in metrics_list]
    losses = [m["test_loss"] for _, m in metrics_list]
    acc_gaps = [m.get("accuracy_gap", 0.0) for _, m in metrics_list]
    loss_gaps = [m.get("loss_gap", 0.0) for _, m in metrics_list]
    
    # Statistics
    acc_mean = float(np.mean(accs)) if accs else 0.0
    acc_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    loss_mean = float(np.mean(losses)) if losses else 0.0
    loss_std = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
    
    acc_ci_lo, acc_ci_hi = _ci_bounds(acc_mean, acc_std, len(accs))
    loss_ci_lo, loss_ci_hi = _ci_bounds(loss_mean, loss_std, len(losses))

    # Communication metrics
    comp_times = [m.get("comp_time_sec", 0.0) for _, m in metrics_list]
    up_bytes = [m.get("upload_bytes", 0) for _, m in metrics_list]
    dn_bytes = [m.get("download_bytes", 0) for _, m in metrics_list]

    result = {
        # Legacy aggregate keys
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "test_accuracy_ci95_low": acc_ci_lo,
        "test_accuracy_ci95_high": acc_ci_hi,
        "test_loss_mean": loss_mean,
        "test_loss_std": loss_std,
        "test_loss_ci95_low": loss_ci_lo,
        "test_loss_ci95_high": loss_ci_hi,
        "accuracy_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else 0.0,
        "loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
        "comp_time_sec_mean": float(np.mean(comp_times)) if comp_times else 0.0,
        "bytes_up_total": float(np.sum(up_bytes)) if up_bytes else 0.0,
        "bytes_down_total": float(np.sum(dn_bytes)) if dn_bytes else 0.0,
        # Standardized duplicate keys (client_*) for cross-project consistency
        "client_test_accuracy_mean": acc_mean,
        "client_test_accuracy_std": acc_std,
        "client_test_accuracy_ci95_low": acc_ci_lo,
        "client_test_accuracy_ci95_high": acc_ci_hi,
        "client_test_loss_mean": loss_mean,
        "client_test_loss_std": loss_std,
        "client_test_loss_ci95_low": loss_ci_lo,
        "client_test_loss_ci95_high": loss_ci_hi,
        "client_accuracy_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else 0.0,
        "client_loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
    }

    # 3) Update rounds.csv with distributed evaluation metrics
    rounds_csv = os.path.join("metrics", "rounds.csv")
    
    # Load existing rows if file exists
    existing = []
    if os.path.exists(rounds_csv):
        with open(rounds_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    else:
        fieldnames = [
            "round", "fit_train_loss_mean", "fit_train_accuracy_mean",
            "bytes_up_total", "bytes_down_total", "comp_time_sec_mean", "wall_clock_sec",
        ]

    # Ensure distributed evaluation columns exist
    for col in [
        # Legacy names
        "test_accuracy_mean", "test_accuracy_std", "test_accuracy_ci95_low", "test_accuracy_ci95_high",
        "test_loss_mean", "test_loss_std", "test_loss_ci95_low", "test_loss_ci95_high",
        "accuracy_gap_mean", "loss_gap_mean",
        # Standardized duplicates
        "client_test_accuracy_mean", "client_test_accuracy_std",
        "client_test_accuracy_ci95_low", "client_test_accuracy_ci95_high",
        "client_test_loss_mean", "client_test_loss_std",
        "client_test_loss_ci95_low", "client_test_loss_ci95_high",
        "client_accuracy_gap_mean", "client_loss_gap_mean",
    ]:
        if col not in fieldnames:
            fieldnames.append(col)

    # Find existing row for this round or create new one
    found = False
    for row in existing:
        if int(row.get("round", 0)) == round_num:
            # Update with distributed evaluation metrics
            row.update({
                # Legacy
                "test_accuracy_mean": acc_mean,
                "test_accuracy_std": acc_std,
                "test_accuracy_ci95_low": acc_ci_lo,
                "test_accuracy_ci95_high": acc_ci_hi,
                "test_loss_mean": loss_mean,
                "test_loss_std": loss_std,
                "test_loss_ci95_low": loss_ci_lo,
                "test_loss_ci95_high": loss_ci_hi,
                "accuracy_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else 0.0,
                "loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
                # Standardized duplicates
                "client_test_accuracy_mean": acc_mean,
                "client_test_accuracy_std": acc_std,
                "client_test_accuracy_ci95_low": acc_ci_lo,
                "client_test_accuracy_ci95_high": acc_ci_hi,
                "client_test_loss_mean": loss_mean,
                "client_test_loss_std": loss_std,
                "client_test_loss_ci95_low": loss_ci_lo,
                "client_test_loss_ci95_high": loss_ci_hi,
                "client_accuracy_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else 0.0,
                "client_loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
            })
            found = True
            break
    
    if not found:
        # Create new row with distributed metrics
        new_row = {
            "round": round_num,
            # Legacy
            "test_accuracy_mean": acc_mean,
            "test_accuracy_std": acc_std,
            "test_accuracy_ci95_low": acc_ci_lo,
            "test_accuracy_ci95_high": acc_ci_hi,
            "test_loss_mean": loss_mean,
            "test_loss_std": loss_std,
            "test_loss_ci95_low": loss_ci_lo,
            "test_loss_ci95_high": loss_ci_hi,
            "accuracy_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else 0.0,
            "loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
            # Standardized duplicates
            "client_test_accuracy_mean": acc_mean,
            "client_test_accuracy_std": acc_std,
            "client_test_accuracy_ci95_low": acc_ci_lo,
            "client_test_accuracy_ci95_high": acc_ci_hi,
            "client_test_loss_mean": loss_mean,
            "client_test_loss_std": loss_std,
            "client_test_loss_ci95_low": loss_ci_lo,
            "client_test_loss_ci95_high": loss_ci_hi,
            "client_accuracy_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else 0.0,
            "client_loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
        }
        existing.append(new_row)

    # Write back to file
    with open(rounds_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    # Increment for next round
    dist_round_counter["value"] += 1

    return result


def aggregate_fit_metrics(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate and log metrics returned by client `fit` to rounds.csv."""
    global fit_round_counter

    round_num = fit_round_counter["value"]

    comp_times = [m.get("comp_time_sec", 0.0) for _, m in metrics_list]
    up_bytes = [m.get("upload_bytes", 0) for _, m in metrics_list]
    dn_bytes = [m.get("download_bytes", 0) for _, m in metrics_list]
    train_losses = [m.get("train_loss_mean", 0.0) for _, m in metrics_list]
    train_accs = [m.get("train_accuracy_mean", 0.0) for _, m in metrics_list]
    train_samples = [m.get("total_train_samples", 0) for _, m in metrics_list]

    # Compute aggregated training metrics
    fit_train_loss_mean = float(np.mean(train_losses)) if train_losses else 0.0
    fit_train_accuracy_mean = float(np.mean(train_accs)) if train_accs else 0.0
    bytes_up_total = float(np.sum(up_bytes)) if up_bytes else 0.0
    bytes_down_total = float(np.sum(dn_bytes)) if dn_bytes else 0.0
    comp_time_sec_mean = float(np.mean(comp_times)) if comp_times else 0.0
    
    # Estimate wall clock time (computation + communication)
    network_speed_mbps = 10.0
    comm_time_sec = (bytes_up_total + bytes_down_total) / (network_speed_mbps * 1e6 / 8)
    wall_clock_sec = comp_time_sec_mean + comm_time_sec

    # Write/update rounds.csv
    os.makedirs("metrics", exist_ok=True)
    rounds_csv = os.path.join("metrics", "rounds.csv")
    
    # Load existing rows if file exists
    existing = []
    if os.path.exists(rounds_csv):
        with open(rounds_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    else:
        fieldnames = ["round", "fit_train_loss_mean", "fit_train_accuracy_mean",
                      "bytes_up_total", "bytes_down_total", "comp_time_sec_mean", "wall_clock_sec"]

    # Create or update row for this round
    row_data = {
        "round": round_num,
        "fit_train_loss_mean": fit_train_loss_mean,
        "fit_train_accuracy_mean": fit_train_accuracy_mean,
        "bytes_up_total": bytes_up_total,
        "bytes_down_total": bytes_down_total,
        "comp_time_sec_mean": comp_time_sec_mean,
        "wall_clock_sec": wall_clock_sec,
    }

    # Find existing row for this round or append new one
    found = False
    for row in existing:
        if int(row.get("round", 0)) == round_num:
            row.update(row_data)
            found = True
            break
    
    if not found:
        existing.append(row_data)

    # Write back to file
    with open(rounds_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    fit_round_counter["value"] += 1

    return {
        "fit_train_loss_mean": fit_train_loss_mean,
        "fit_train_accuracy_mean": fit_train_accuracy_mean,
        "bytes_up_total": bytes_up_total,
        "bytes_down_total": bytes_down_total,
        "comp_time_sec_mean": comp_time_sec_mean,
        "wall_clock_sec": wall_clock_sec,
    }

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


def evaluate_and_log_central_dataset(round_num: int, parameters, config, 
                                   data_root: str, use_watches: bool, sample_rate_hz: int,
                                   window_seconds: int, window_stride_seconds: int, num_classes: int):
    """Centralized evaluation using HHAR full dataset."""
    # 1) Load full HHAR dataset for centralized eval
    trainloader, testloader, n_classes = load_full_data(
        batch_size=64,
        data_root=data_root,
        use_watches=use_watches,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        window_stride_seconds=window_stride_seconds,
        num_classes=num_classes,
    )

    # 2) Infer input shape from HHAR data: (B, C, T)
    sample, _ = next(iter(trainloader))
    if sample.ndim == 4:  # (batch, channels, time, extra) - take first sample
        _, in_ch, T = sample[0].shape
    elif sample.ndim == 3:  # (batch, channels, time)
        _, in_ch, T = sample.shape
    else:
        raise ValueError(f"Unexpected sample shape: {sample.shape}")

    # 3) Build model & load global params
    net = Net(in_ch=in_ch, seq_len=T, num_classes=n_classes)
    # Handle both Parameters object and raw list of ndarrays
    if isinstance(parameters, list):
        nds = parameters
    else:
        nds = parameters_to_ndarrays(parameters)
    set_weights(net, nds)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 4) Centralized metrics
    train_loss, train_acc = test(net, trainloader, device)
    test_loss, test_acc = test(net, testloader, device)
    print(f"Round {round_num}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    rec = {
        "central_train_loss": float(train_loss),
        "central_train_accuracy": float(train_acc),
        "central_test_loss": float(test_loss),
        "central_test_accuracy": float(test_acc),
        "central_loss_gap": float(test_loss - train_loss),
        "central_accuracy_gap": float(train_acc - test_acc),
    }

    conv_metrics = ctracker.update(round_num, test_loss, test_acc)
    rec.update(conv_metrics)

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
    os.makedirs("metrics", exist_ok=True)
    path = os.path.join("metrics", "centralized_metrics.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({"round": round_num, **rec})
    return test_loss, rec

def server_fn(context: Context):
    global strategy

    # Read HHAR config parameters
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_available_clients = context.run_config.get("min_available_clients", 9)
    
    # FedProx hyperparameters
    proximal_mu = context.run_config.get("proximal_mu", 0.01)
    
    # HHAR dataset parameters
    data_root = context.run_config.get("data-root", "hhar")
    use_watches = context.run_config.get("use-watches", True)
    sample_rate_hz = context.run_config.get("sample-rate-hz", 50)
    window_seconds = context.run_config.get("window-seconds", 2)
    window_stride_seconds = context.run_config.get("window-stride-seconds", 1)
    num_classes = context.run_config.get("num-classes", 6)

    # Initialize model using HHAR data
    trainloader, testloader, n_classes = load_data(
        partition_id=0, 
        num_partitions=1, 
        batch_size=1,
        data_root=data_root,
        use_watches=use_watches,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        window_stride_seconds=window_stride_seconds,
        num_classes=num_classes,
    )

    sample, _ = next(iter(trainloader))
    if isinstance(sample, torch.Tensor):
        if sample.ndim == 4:  # (batch, channels, time, extra) - take first sample
            _, in_ch, T = sample[0].shape
        elif sample.ndim == 3:  # (batch, channels, time)
            _, in_ch, T = sample.shape
        else:
            raise ValueError(f"Unexpected sample shape: {sample.shape}")
    else:
        print("Sample is NOT a tensor! Type:", type(sample))
        print("Sample content:", sample)
        raise ValueError("Sample is not a tensor. Did your transform apply?")

    ndarrays = get_weights(Net(in_ch=in_ch, seq_len=T, num_classes=n_classes))
    parameters = ndarrays_to_parameters(ndarrays)

    def eval_fn(round_num, parameters, config):
        return evaluate_and_log_central_dataset(
            round_num, parameters, config,
            data_root, use_watches, sample_rate_hz,
            window_seconds, window_stride_seconds, num_classes
        )

    strategy = FedProx(
        proximal_mu=proximal_mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_and_log,
        evaluate_fn=eval_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
