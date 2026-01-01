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
import hashlib

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
def _get_cache_key(data_root: str, use_watches: bool, sample_rate_hz: int, 
                   window_seconds: int, window_stride_seconds: int) -> str:
    """Generate cache key for NPZ files based on processing parameters."""
    key_str = f"{data_root}_{use_watches}_{sample_rate_hz}_{window_seconds}_{window_stride_seconds}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]

def _load_or_cache_processed_data(data_root: str, use_watches: bool, sample_rate_hz: int,
                                  window_seconds: int, window_stride_seconds: int):
    """Load processed data from cache or create and cache it."""
    cache_key = _get_cache_key(data_root, use_watches, sample_rate_hz, window_seconds, window_stride_seconds)
    cache_dir = os.path.join(data_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"processed_{cache_key}.npz")
    
    if os.path.exists(cache_file):
        print(f"Loading cached processed data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['user_data'].item()  # dict of user -> list of (X, y) arrays
    
    print(f"Processing HHAR data (will cache to {cache_file})")
    # 0) Ensure HHAR dataset is available
    ensure_hhar_dataset(data_root, use_uci_official=True, use_watches=use_watches)
    
    # 1) Load CSVs and build merged stream
    acc, gyro = _load_csvs(data_root, use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)

    # 2) Group by user and src, then pre-process all windows
    frames = _windowize(merged, window_seconds, window_stride_seconds, sample_rate_hz)
    
    user_data = {}
    T = window_seconds * sample_rate_hz
    S = window_stride_seconds * sample_rate_hz
    
    for df, user, src in frames:
        if user not in user_data:
            user_data[user] = []
        
        # Pre-compute all windows for this (user, src)
        arr = df[["ax","ay","az","gx","gy","gz"]].to_numpy(dtype=np.float32, copy=False)
        lab = df["y"].to_numpy(dtype=np.int64, copy=False)
        
        windows_X = []
        windows_y = []
        
        for start in range(0, len(arr) - T + 1, S):
            x = arr[start:start+T].T  # shape (6, T)
            y = np.bincount(lab[start:start+T]).argmax()
            windows_X.append(x)
            windows_y.append(y)
        
        if windows_X:
            # Stack windows: (N, 6, T) where N is number of windows
            X_stack = np.stack(windows_X, axis=0)  # (N, 6, T)
            y_stack = np.array(windows_y)  # (N,)
            user_data[user].append((X_stack, y_stack))
    
    # Cache the processed data
    np.savez_compressed(cache_file, user_data=user_data)
    print(f"Cached processed data to {cache_file}")
    
    return user_data

class CachedWindowedSeq(Dataset):
    """Memory-efficient dataset using pre-cached windows.

    Optionally accept precomputed normalization statistics (means/stds) so that
    test/validation can reuse training normalization.
    """
    def __init__(self, user_windows_list, normalize=True, means=None, stds=None):
        self.samples = []  # (X_array, y_array, window_idx)
        self.means = None
        self.stds = None
        
        # Collect all windows from this user's (user, src) pairs
        all_X = []
        for X_arr, y_arr in user_windows_list:
            for i in range(len(X_arr)):
                self.samples.append((X_arr, y_arr, i))
                # Only gather samples to compute stats if we don't already have them
                if normalize and means is None and stds is None and len(all_X) < 20000:
                    all_X.append(X_arr[i])
        
        if normalize:
            if means is not None and stds is not None:
                # Use provided statistics
                self.means = np.array(means, dtype=np.float32)
                self.stds  = np.array(stds, dtype=np.float32)
            elif all_X:
                # Compute statistics from a subset
                cat = np.array(all_X)            # (N, 6, T)
                m = cat.mean(axis=(0, 2))        # (6,)
                s = cat.std(axis=(0, 2)) + 1e-8  # (6,)
                self.means = m[:, None]          # (6, 1)
                self.stds  = s[:, None]          # (6, 1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X_arr, y_arr, window_idx = self.samples[idx]
        x = X_arr[window_idx].copy()  # (6, T)
        y = y_arr[window_idx]
        
        # Ensure x has correct shape (6, T) - remove any extra dimensions
        if x.ndim > 2:
            x = x.squeeze()
        
        if self.means is not None:
            x = (x - self.means) / self.stds  # (6, 1) stats -> stays (6, T)
            if x.ndim != 2:
                x = np.squeeze(x)
        
        return torch.from_numpy(x), int(y)

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
    # Load or create cached processed data
    user_data = _load_or_cache_processed_data(data_root, use_watches, sample_rate_hz, 
                                              window_seconds, window_stride_seconds)
    
    # Map partition_id to user
    users = sorted(list(user_data.keys()))
    assert len(users) >= num_partitions, f"Need >= {num_partitions} users; found {len(users)}"
    
    user_for_client = users[partition_id]
    user_windows = user_data[user_for_client]
    
    # Split user's data into train/test within each (user, src) shard (80/20 per shard)
    train_windows = []
    test_windows = []
    for X_arr, y_arr in user_windows:
        n = len(X_arr)
        if n <= 1:
            # Not enough windows to split; keep for training
            train_windows.append((X_arr, y_arr))
            continue
        cut = max(1, int(0.8 * n))
        train_windows.append((X_arr[:cut], y_arr[:cut]))
        if n - cut > 0:
            test_windows.append((X_arr[cut:], y_arr[cut:]))
    # Fallback to ensure we have some test data
    if not test_windows and user_windows:
        X_arr, y_arr = user_windows[-1]
        n = len(X_arr)
        if n > 1:
            k = max(1, n // 5)
            test_windows = [(X_arr[-k:], y_arr[-k:])]
            train_windows[-1] = (X_arr[:-k], y_arr[:-k])
    
    # Datasets with consistent normalization (test uses train statistics)
    train_ds = CachedWindowedSeq(train_windows, normalize=True)
    test_ds = CachedWindowedSeq(test_windows, normalize=True, means=train_ds.means, stds=train_ds.stds)
    
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
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
    Uses caching to prevent OOM.
    """
    # Load cached processed data
    user_data = _load_or_cache_processed_data(data_root, use_watches, sample_rate_hz, 
                                              window_seconds, window_stride_seconds)
    
    # Combine all users' data and split within each (user, src) shard
    train_windows = []
    test_windows = []
    for _, user_windows in user_data.items():
        for X_arr, y_arr in user_windows:
            n = len(X_arr)
            if n <= 1:
                train_windows.append((X_arr, y_arr))
                continue
            cut = max(1, int(0.8 * n))
            train_windows.append((X_arr[:cut], y_arr[:cut]))
            if n - cut > 0:
                test_windows.append((X_arr[cut:], y_arr[cut:]))
    # Ensure we have some test data
    if not test_windows and train_windows:
        X_arr, y_arr = train_windows[-1]
        n = len(X_arr)
        if n > 1:
            k = max(1, n // 5)
            test_windows = [(X_arr[-k:], y_arr[-k:])]
            train_windows[-1] = (X_arr[:-k], y_arr[:-k])

    # Use training normalization for test set as well
    train_ds = CachedWindowedSeq(train_windows, normalize=True)
    test_ds = CachedWindowedSeq(test_windows, normalize=True, means=train_ds.means, stds=train_ds.stds)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        num_classes,
    )

# ───────────────────────────── Eval / weights helpers ─────────────────────────────
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
