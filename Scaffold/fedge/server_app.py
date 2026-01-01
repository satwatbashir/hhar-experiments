"""fedge: A Flower / PyTorch app (SCAFFOLD server)."""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import csv
import os

import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fedge.task import Net, get_weights, load_data, load_full_data, set_weights, test
from .scaffold import Scaffold


# ───────────────────────────── Utilities ─────────────────────────────

class ConvergenceTracker:
    """Tiny helper to report crude convergence rates/stability per round."""
    def __init__(self) -> None:
        self.prev_loss: float | None = None
        self.prev_acc: float | None = None

    def update(self, rnd: int, loss: float, acc: float) -> Dict[str, float]:
        if self.prev_loss is None:
            self.prev_loss, self.prev_acc = loss, acc
            return {
                "conv_loss_rate": 0.0,
                "conv_acc_rate": 0.0,
                "conv_loss_stability": 0.0,
                "conv_acc_stability": 0.0,
            }
        dl = self.prev_loss - loss      # positive if loss decreased
        da = acc - self.prev_acc        # positive if acc increased
        rec = {
            "conv_loss_rate": float(dl),
            "conv_acc_rate": float(da),
            "conv_loss_stability": float(abs(dl)),
            "conv_acc_stability": float(abs(da)),
        }
        self.prev_loss, self.prev_acc = loss, acc
        return rec


ctracker = ConvergenceTracker()

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)


# ───────────────────── Metrics aggregation/logging ─────────────────────

def _weighted_avg(values: List[float], weights: List[int]) -> float:
    tot = float(sum(weights))
    if tot <= 0:
        return float(np.mean(values)) if values else 0.0
    return float(sum(v * (w / tot) for v, w in zip(values, weights)))


# Optional: simple counters so we can log the round number in CSVs
_fit_round = {"v": 0}
_eval_round = {"v": 0}

def aggregate_fit_metrics(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Flower passes List[(num_examples, metrics_dict)] here.
    Aggregate weighted-by-num_examples and write one CSV row per round.
    """
    nums: List[int] = []
    store: Dict[str, List[Tuple[float, int]]] = {}
    for n, m in fit_metrics:
        if not m:
            continue
        n = int(n)
        nums.append(n)
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)):
                store.setdefault(k, []).append((float(v), n))

    # Weighted means for most metrics
    agg: Dict[str, float] = {
        k: _weighted_avg([v for v, w in lst], [w for v, w in lst])
        for k, lst in store.items()
    }

    # SUM communication fields if present
    bytes_up_total = 0
    bytes_down_total = 0
    for n, m in fit_metrics:
        if not m: 
            continue
        if "upload_bytes" in m:
            bytes_up_total += int(m["upload_bytes"])
        if "download_bytes" in m:
            bytes_down_total += int(m["download_bytes"])
    
    # If Scaffold added per-round totals (recommended above), prefer those:
    if "bytes_up_total" in agg or "bytes_down_total" in agg:
        pass  # keep Scaffold totals
    else:
        agg["bytes_up_total"] = bytes_up_total
        agg["bytes_down_total"] = bytes_down_total

    # Return aggregated metrics (no CSV writing - handled by strategy now)
    return agg


from scipy.stats import t
import math

def _ci_bounds(mean: float, std: float, n: int) -> tuple[float, float]:
    """95% confidence interval bounds (two-sided)."""
    if n <= 1:
        return (float("nan"), float("nan"))
    tcrit = t.ppf(0.975, df=n-1)   # 97.5% quantile, two-sided 95% CI
    half = tcrit * (std / math.sqrt(n))
    return (mean - half, mean + half)

def aggregate_eval_metrics(eval_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Same shape for evaluation metrics: List[(num_examples, metrics_dict)].
    """
    store: Dict[str, List[Tuple[float, int]]] = {}
    for n, m in eval_metrics:
        if not m:
            continue
        n = int(n)
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)):
                store.setdefault(k, []).append((float(v), n))

    # Weighted means for most metrics
    agg: Dict[str, float] = {
        k: _weighted_avg([v for v, w in lst], [w for v, w in lst])
        for k, lst in store.items()
    }

    # Add per-round std calculation for client test accuracy/loss
    acc_vals = [float(m.get("test_accuracy", 0)) for _, m in eval_metrics if m and "test_accuracy" in m]
    loss_vals = [float(m.get("test_loss", 0)) for _, m in eval_metrics if m and "test_loss" in m]
    # New: client gaps
    acc_gap_vals  = [float(m.get("accuracy_gap", 0)) for _, m in eval_metrics if m and "accuracy_gap" in m]
    loss_gap_vals = [float(m.get("loss_gap", 0))     for _, m in eval_metrics if m and "loss_gap" in m]
    
    if acc_vals:
        m_acc = float(np.mean(acc_vals))
        s_acc = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0
        agg["client_test_accuracy_mean"] = m_acc
        agg["client_test_accuracy_std"] = s_acc
        lo, hi = _ci_bounds(m_acc, s_acc, len(acc_vals))
        agg["client_test_accuracy_ci95_low"] = lo
        agg["client_test_accuracy_ci95_high"] = hi
        
    if loss_vals:
        m_loss = float(np.mean(loss_vals))
        s_loss = float(np.std(loss_vals, ddof=1)) if len(loss_vals) > 1 else 0.0
        agg["client_test_loss_mean"] = m_loss
        agg["client_test_loss_std"] = s_loss
        lo, hi = _ci_bounds(m_loss, s_loss, len(loss_vals))
        agg["client_test_loss_ci95_low"] = lo
        agg["client_test_loss_ci95_high"] = hi
    
    # New: means/std for gaps
    if acc_gap_vals:
        agg["client_accuracy_gap_mean"] = float(np.mean(acc_gap_vals))
        agg["client_accuracy_gap_std"]  = float(np.std(acc_gap_vals, ddof=1)) if len(acc_gap_vals) > 1 else 0.0
    if loss_gap_vals:
        agg["client_loss_gap_mean"] = float(np.mean(loss_gap_vals))
        agg["client_loss_gap_std"]  = float(np.std(loss_gap_vals, ddof=1)) if len(loss_gap_vals) > 1 else 0.0

    # SUM communication fields if present (though eval typically doesn't have bytes)
    bytes_up_total = 0
    bytes_down_total = 0
    for n, m in eval_metrics:
        if not m: 
            continue
        if "upload_bytes" in m:
            bytes_up_total += int(m["upload_bytes"])
        if "download_bytes" in m:
            bytes_down_total += int(m["download_bytes"])
    
    if bytes_up_total > 0 or bytes_down_total > 0:
        agg["bytes_up_total"] = bytes_up_total
        agg["bytes_down_total"] = bytes_down_total

    # ---- Update rounds.csv (append or update last row) ----
    rounds_csv = os.path.join("metrics", "rounds.csv")

    # Load existing rows (fit phase wrote/created the row for this round)
    existing = []
    if os.path.exists(rounds_csv):
        with open(rounds_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    else:
        fieldnames = ["round", "fit_train_loss_mean", "fit_train_accuracy_mean",
                      "bytes_up_total", "bytes_down_total", "comp_time_sec_mean", "wall_clock_sec"]

    # Ensure columns exist
    for col in ["client_test_loss_mean","client_test_accuracy_mean",
                "client_test_loss_std","client_test_accuracy_std",
                "client_test_accuracy_ci95_low","client_test_accuracy_ci95_high",
                "client_test_loss_ci95_low","client_test_loss_ci95_high",
                "client_accuracy_gap_mean","client_accuracy_gap_std",
                "client_loss_gap_mean","client_loss_gap_std"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # If fit didn't create a row (shouldn't happen), append one
    if not existing:
        existing = [{"round": 0}]

    # Update last row with aggregates we just computed
    last = existing[-1]
    for k in ["client_test_loss_mean","client_test_accuracy_mean",
              "client_test_loss_std","client_test_accuracy_std",
              "client_test_accuracy_ci95_low","client_test_accuracy_ci95_high",
              "client_test_loss_ci95_low","client_test_loss_ci95_high",
              "client_accuracy_gap_mean","client_accuracy_gap_std",
              "client_loss_gap_mean","client_loss_gap_std"]:
        if k in agg:
            last[k] = agg[k]

    with open(rounds_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    # Return aggregated metrics (no CSV writing - decluttered)
    return agg


# ───────────────────── Centralized evaluation (server) ─────────────────────

def make_eval_fn(batch_size: int, data_root: str, use_watches: bool, sample_rate_hz: int, 
                 window_seconds: int, window_stride_seconds: int, num_classes: int):
    """Bind dataset choice into a Flower evaluate_fn for centralized metrics."""
    def evaluate_and_log_central(server_round: int, parameters, config):
        # 1) Load centralized (all users) dataset for global evaluation
        trainloader, testloader, n_classes = load_full_data(
            batch_size=batch_size, data_root=data_root, use_watches=use_watches,
            sample_rate_hz=sample_rate_hz, window_seconds=window_seconds,
            window_stride_seconds=window_stride_seconds, num_classes=num_classes,
        )

        # 2) Infer input shape from a batch
        sample, _ = next(iter(trainloader))  # shape (B, C, T)
        _, in_ch, T = sample.shape

        # 3) Build model, load global weights, choose device
        net = Net(in_ch=in_ch, seq_len=T, num_classes=n_classes)
        set_weights(net, parameters)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # 4) Compute centralized metrics
        train_loss, train_acc = test(net, trainloader, device)
        test_loss, test_acc = test(net, testloader, device)

        rec = {
            "central_train_loss": float(train_loss),
            "central_train_accuracy": float(train_acc),
            "central_test_loss": float(test_loss),
            "central_test_accuracy": float(test_acc),
            "central_loss_gap": float(test_loss - train_loss),
            "central_accuracy_gap": float(train_acc - test_acc),
        }
        rec.update(ctracker.update(server_round, test_loss, test_acc))

        # 5) Append to CSV
        os.makedirs("metrics", exist_ok=True)
        path = os.path.join("metrics", "centralized_metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            fieldnames = [
                "round",
                "central_train_loss", "central_train_accuracy",
                "central_test_loss", "central_test_accuracy",
                "central_loss_gap", "central_accuracy_gap",
                "conv_loss_rate", "conv_acc_rate",
                "conv_loss_stability", "conv_acc_stability",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({"round": server_round, **rec})

        # Flower expects (loss, metrics) from evaluate_fn
        return float(test_loss), rec

    return evaluate_and_log_central


# ───────────────────────────── Server factory ─────────────────────────────

def server_fn(context: Context) -> ServerAppComponents:
    # Read config
    num_rounds = int(context.run_config.get("num-server-rounds", 100))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    batch_size = int(context.run_config.get("batch_size", 64))
    data_root = context.run_config.get("data-root", "hhar")
    use_watches = context.run_config.get("use-watches", True)
    sample_rate_hz = int(context.run_config.get("sample-rate-hz", 50))
    window_seconds = int(context.run_config.get("window-seconds", 2))
    window_stride_seconds = int(context.run_config.get("window-stride-seconds", 1))
    num_classes = int(context.run_config.get("num-classes", 6))
    # Build an init model using one batch to infer input shape
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

    # Centralized evaluation fn bound to the dataset choice
    eval_fn = make_eval_fn(batch_size, data_root, use_watches, sample_rate_hz,
                          window_seconds, window_stride_seconds, num_classes)

    # Strategy
    strategy = Scaffold(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
        evaluate_fn=eval_fn,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
