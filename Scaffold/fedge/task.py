# task.py — HHAR dataset processing and model definition

from __future__ import annotations
import os, glob
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

def ensure_hhar_dataset(base: str, use_uci_official: bool = True, use_watches: bool = True) -> None:
    """
    Verify HHAR CSVs exist under `base`. Never downloads.
    We intentionally ignore 'Still exp' if present; only 'Activity recognition exp' is required.
    """
    os.makedirs(base, exist_ok=True)
    phone_acc  = [p for p in glob.glob(os.path.join(base, "**", "Phones_accelerometer*.csv"), recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    phone_gyro = [p for p in glob.glob(os.path.join(base, "**", "Phones_gyroscope*.csv"),    recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    if not phone_acc or not phone_gyro:
        raise RuntimeError(
            "HHAR not found (need Phones_accelerometer*.csv and Phones_gyroscope*.csv). "
            f"Ensure 'Activity recognition exp' is extracted under: {os.path.abspath(base)}"
        )
    if use_watches:
        watch_acc  = glob.glob(os.path.join(base, "**", "Watch_accelerometer*.csv"), recursive=True)
        watch_gyro = glob.glob(os.path.join(base, "**", "Watch_gyroscope*.csv"),     recursive=True)
        if not watch_acc or not watch_gyro:
            raise RuntimeError("Watch CSVs missing but use_watches=True. Either extract them or set use-watches=false.")

# -----------------------------
# Model (kept name: Net)
# -----------------------------
class Net(nn.Module):
    """
    Tiny 1D-CNN for HHAR windows. Input: (B, C=6, T), Output: (B, num_classes)
    If you want ultra-basic: set USE_LOGREG=True to swap body to logistic regression.
    """
    USE_LOGREG = False  # toggle to True only if you want the ultra-basic model

    def __init__(self, in_ch: int = 6, seq_len: int = 100, num_classes: int = 6):
        super().__init__()
        if self.USE_LOGREG:
            self.body = nn.Identity()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(in_ch, num_classes)
        else:
            self.body = nn.Sequential(
                nn.Conv1d(in_ch, 64, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(64, num_classes)

    def forward(self, x):  # x: (B, C, T)
        if self.USE_LOGREG:
            # average over time -> (B, C)
            x = self.pool(x).squeeze(-1)
            return self.head(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)  # (B, 64)
        return self.head(x)

# -----------------------------
# Dataset & preprocessing
# -----------------------------
ACTIVITY_ORDER = [
    # ensure stable mapping to 0..(num_classes-1)
    "walking", "sitting", "standing", "biking", "stairsup", "stairsdown"
]

def _label_to_int(lbl: str) -> int:
    s = str(lbl).strip().lower().replace(" ", "")
    # Check for null/unknown labels first
    if s in ("null", "none", "", "nan"):
        return -1  # Mark for filtering
    if s not in ACTIVITY_ORDER:
        # fallback for dataset aliases
        alias = {
            "walk": "walking", "cycling": "biking", "bike": "biking",
            "sit": "sitting", "stand": "standing", "upstairs": "stairsup",
            "downstairs": "stairsdown", "stairup": "stairsup", "stairdown": "stairsdown",
        }
        s = alias.get(s, s)
    if s not in ACTIVITY_ORDER:
        return -1  # Mark for filtering instead of raising error
    return ACTIVITY_ORDER.index(s)

def _col(df: pd.DataFrame, names: List[str]) -> str:
    """Find a column by trying several likely names (case-insensitive)."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    raise KeyError(f"none of {names} found in columns={list(df.columns)}")

def _load_csvs(base: str, use_watches: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load phone/watch acc/gyro CSVs from UCI HHAR and return merged long tables:
      acc  : [timestamp, user, device, activity, x, y, z, src]
      gyro : [timestamp, user, device, activity, x, y, z, src]
    'src' marks 'phone' or 'watch'
    """
    # Try typical UCI filenames (adjust if yours differ)
    phone_acc  = [p for p in glob.glob(os.path.join(base, "**", "Phones_accelerometer*.csv"), recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    phone_gyro = [p for p in glob.glob(os.path.join(base, "**", "Phones_gyroscope*.csv"),    recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    watch_acc  = [p for p in glob.glob(os.path.join(base, "**", "Watch_accelerometer*.csv"),  recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    watch_gyro = [p for p in glob.glob(os.path.join(base, "**", "Watch_gyroscope*.csv"),      recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]

    def read_mark(paths: list[str], src: str) -> pd.DataFrame:
        usecols = ["User", "Device", "Model", "gt", "Arrival_Time", "Creation_Time", "x", "y", "z"]
        dtypes  = {
            "User": "string",          # small strings; will compress later
            "Device": "string",
            "Model": "string",
            "gt": "string",
            "x": "float32", "y": "float32", "z": "float32",
        }
        dfs = []
        for p in paths:
            df = pd.read_csv(
                p,
                usecols=[c for c in usecols if c in pd.read_csv(p, nrows=0).columns],
                dtype=dtypes,
                engine="c",
                memory_map=True,
            )
            df["src"] = src
            # Build timestamp from Arrival_Time or Creation_Time (ms)
            s = df["Arrival_Time"] if "Arrival_Time" in df.columns else df["Creation_Time"]
            s = pd.to_numeric(s, errors="coerce")
            ts = pd.to_datetime(s, unit="ms", errors="coerce")

            # Drop old time cols (avoid dtype clashes/copies) and rename to lean schema
            drop_cols = [c for c in ["Arrival_Time", "Creation_Time", "Model"] if c in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)

            # Replace, don't overwrite mismatched dtype
            df.drop(columns=[c for c in ["timestamp"] if c in df.columns], inplace=True, errors="ignore")
            df["timestamp"] = pd.DatetimeIndex(ts)

            df.rename(columns={"User": "user", "Device": "device", "gt": "activity"}, inplace=True)
            df = df[["user", "device", "src", "timestamp", "x", "y", "z", "activity"]]
            # compact strings to category (much smaller)
            df["user"] = df["user"].astype("category")
            df["device"] = df["device"].astype("category")
            df["activity"] = df["activity"].astype("category")
            dfs.append(df)

        if not dfs: return pd.DataFrame()
        out = pd.concat(dfs, ignore_index=True)
        return out

    acc  = read_mark(phone_acc, "phone")
    gyro = read_mark(phone_gyro, "phone")
    if use_watches:
        acc  = pd.concat([acc,  read_mark(watch_acc,  "watch")], ignore_index=True)
        gyro = pd.concat([gyro, read_mark(watch_gyro, "watch")], ignore_index=True)

    # Harmonize columns
    for df in (acc, gyro):
        if df.empty: continue
        tcol = _col(df, ["timestamp", "time", "creation_time", "TimeStamp"])
        ucol = _col(df, ["user", "subject", "subject_id"])
        dcol = _col(df, ["device", "model", "device_id", "Device"])
        lcol = _col(df, ["activity", "gt", "label"])
        xcol = _col(df, ["x", "X"])
        ycol = _col(df, ["y", "Y"])
        zcol = _col(df, ["z", "Z"])
        df.rename(columns={tcol:"timestamp", ucol:"user", dcol:"device",
                           lcol:"activity", xcol:"x", ycol:"y", zcol:"z"}, inplace=True)
        # unify label names
        df["activity"] = df["activity"].astype(str)

    return acc, gyro

def _prep_and_resample(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """
    Clean one sensor table and resample to target_hz in a memory-safe way:
      - work per identity (user, device, src) to avoid huge global groupby
      - deduplicate timestamps within each identity
      - resample to fixed rate with ffill/bfill
    """
    # Keep only what we need and ensure tight dtypes
    cols = ["user", "device", "src", "timestamp", "x", "y", "z", "activity"]
    df = df[cols].dropna(subset=["timestamp"]).copy()
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype("float32", copy=False)
    df = df.sort_values(["user", "device", "src", "timestamp"])

    rule = f"{int(round(1000 / target_hz))}ms"

    out_frames = []
    # observed=True prevents cartesian-product reindexing with categoricals
    for (u, d, s), g in df.groupby(["user", "device", "src"], observed=True):
        # Deduplicate exact timestamp collisions within this identity
        # (mean x/y/z over same-timestamp samples; keep last activity label)
        g = (
            g.groupby("timestamp", as_index=False, observed=True)
             .agg({"x": "mean", "y": "mean", "z": "mean", "activity": "last"})
        )

        g = g.set_index("timestamp")
        g = g[~g.index.duplicated(keep="last")].sort_index()

        # Resample numeric; carry activity with ffill
        num = g[["x", "y", "z"]].astype("float32").resample(rule).mean().ffill().bfill()
        act = g[["activity"]].reindex(num.index, method="ffill")

        out = num.copy()
        out["activity"] = act["activity"]
        out["user"] = u
        out["device"] = d
        out["src"] = s
        out_frames.append(out.reset_index())

    if not out_frames:
        raise RuntimeError("No resampled data produced (empty sensor table after filtering).")

    return pd.concat(out_frames, ignore_index=True)

def _resample_merge(acc: pd.DataFrame, gyro: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    if acc.empty or gyro.empty:
        raise RuntimeError("Missing accelerometer or gyroscope CSVs.")
    a = _prep_and_resample(acc,  target_hz).rename(columns={"x":"ax","y":"ay","z":"az"})
    g = _prep_and_resample(gyro, target_hz).rename(columns={"x":"gx","y":"gy","z":"gz"})
    m = pd.merge(
        a, g,
        on=["user","device","src","timestamp","activity"],
        how="inner"
    ).dropna()
    if m.empty:
        raise RuntimeError("Resampling produced no aligned rows. Check CSV columns and timestamps.")
    m["y"] = m["activity"].map(lambda s: _label_to_int(s))
    # Filter out unknown labels (y >= 6 or y == -1)
    m = m[m["y"] >= 0].copy()
    m = m[m["y"] < 6].copy()
    return m[["user","device","src","timestamp","ax","ay","az","gx","gy","gz","y"]]

def _windowize(df: pd.DataFrame, window_sec: int, stride_sec: int, hz: int) -> List[tuple[pd.DataFrame, str, str]]:
    """
    Group dataframe by (user, src) and return list of (dataframe, user, src) tuples
    for efficient windowing in the dataset class.
    """
    frames = []
    for (u, s), grp in df.groupby(["user", "src"], observed=True):
        frames.append((grp, str(u), str(s)))
    return frames

class WindowedSeq(Dataset):
    def __init__(self, frames: list[tuple[pd.DataFrame, str, str]], window_sec: int, stride_sec: int, hz: int, normalize=True):
        self.T = window_sec * hz
        self.S = stride_sec * hz
        self.samples = []   # (arr, labels, start)
        self.means = None
        self.stds = None

        # build once per (user, src) chunk
        for df, u, s in frames:
            arr = df[["ax","ay","az","gx","gy","gz"]].to_numpy(dtype=np.float32, copy=False)
            lab = df["y"].to_numpy(dtype=np.int64, copy=False)
            for start in range(0, len(arr) - self.T + 1, self.S):
                self.samples.append((arr, lab, start))

        if normalize:
            # compute global mean/std over a random subset to avoid huge scans
            idx = np.linspace(0, len(self.samples)-1, num=min(20000, len(self.samples)), dtype=int)
            cat = np.vstack([self.samples[i][0][self.samples[i][2]:self.samples[i][2]+self.T] for i in idx])
            self.means = cat.mean(axis=0)
            self.stds  = cat.std(axis=0) + 1e-8

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        arr, lab, start = self.samples[i]
        x = arr[start:start+self.T].T  # shape (6, T)
        if self.means is not None:
            x = (x - self.means[:, None]) / self.stds[:, None]
        # Filter out invalid labels before majority vote
        window_labels = lab[start:start+self.T]
        valid_labels = window_labels[(window_labels >= 0) & (window_labels < 6)]
        if len(valid_labels) == 0:
            # Fallback to first valid label in window, or 0 if none
            y = 0
        else:
            y = np.bincount(valid_labels).argmax()
        return torch.from_numpy(x), int(y)

# -----------------------------
# Public API expected by your apps
# -----------------------------
def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 64,
    data_root: str = "hhar",
    use_watches: bool = True,
    sample_rate_hz: int = 50,
    window_seconds: int = 2,
    window_stride_seconds: int = 1,
    num_classes: int = 6,
):
    # 0) Ensure HHAR dataset is available
    ensure_hhar_dataset(data_root, use_uci_official=True, use_watches=use_watches)
    
    # 1) Load CSVs and build merged stream
    acc, gyro = _load_csvs(data_root, use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)

    # 2) Group by user and src
    frames = _windowize(merged, window_seconds, window_stride_seconds, sample_rate_hz)

    # 3) Clients = users (9 users => 9 clients). No Dirichlet.
    user_frames = {}
    for df, user, src in frames:
        if user not in user_frames:
            user_frames[user] = []
        user_frames[user].append((df, user, src))
    
    users = sorted(list(user_frames.keys()))
    assert len(users) >= num_partitions, f"Need >= {num_partitions} users; found {len(users)}"

    # deterministically map partition_id -> user
    user_for_client = users[partition_id]
    client_frames = user_frames[user_for_client]

    # 4) Build datasets/loaders
    # Train/test split per client: 80/20 on that user's frames
    total_frames = len(client_frames)
    cut = int(0.8 * total_frames)
    train_frames = client_frames[:cut]
    test_frames = client_frames[cut:] if cut < total_frames else client_frames[-max(1, total_frames//5):]

    train_ds = WindowedSeq(train_frames, window_seconds, window_stride_seconds, sample_rate_hz, normalize=True)
    test_ds = WindowedSeq(test_frames, window_seconds, window_stride_seconds, sample_rate_hz, normalize=True)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader, num_classes

def load_full_data(
    batch_size: int = 64,
    data_root: str = "hhar",
    use_watches: bool = True,
    sample_rate_hz: int = 50,
    window_seconds: int = 2,
    window_stride_seconds: int = 1,
    num_classes: int = 6,
):
    """
    Build train/test over the ENTIRE population (all users/windows), for centralized evaluation.
    """
    ensure_hhar_dataset(data_root, use_uci_official=True, use_watches=use_watches)
    acc, gyro = _load_csvs(data_root, use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)
    frames = _windowize(merged, window_seconds, window_stride_seconds, sample_rate_hz)

    # Split frames into train/test (80/20)
    total_frames = len(frames)
    cut = int(0.8 * total_frames)
    train_frames = frames[:cut]
    test_frames = frames[cut:]
    
    train_ds = WindowedSeq(train_frames, window_seconds, window_stride_seconds, sample_rate_hz, normalize=True)
    test_ds = WindowedSeq(test_frames, window_seconds, window_stride_seconds, sample_rate_hz, normalize=True)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
        num_classes,
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Eval / weights helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.eval()
    ce = nn.CrossEntropyLoss(reduction="mean")
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            if isinstance(labels, torch.Tensor):
                if labels.ndim > 1:
                    labels = labels.squeeze()
                labels = labels.long()
                labels = labels.to(device)
            logits = net(sequences)
            loss = ce(logits, labels)
            loss_sum += float(loss.item()) * int(labels.shape[0])
            correct += int((logits.argmax(1) == labels).sum().item())
            total += int(labels.shape[0])
    return (loss_sum / max(1, total)), (correct / max(1, total))

def get_weights(net: nn.Module):
    return [v.detach().cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, weights):
    keys = list(net.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(w) for k, w in zip(keys, weights)})
    net.load_state_dict(state_dict, strict=True)
