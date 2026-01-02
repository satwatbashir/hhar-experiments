# fedge/task.py — HHAR dataset processing and model definition

from __future__ import annotations
import os, glob, zipfile, hashlib
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import toml
import gc

# Load HHAR configuration from pyproject.toml
task_script_dir = Path(__file__).resolve().parent
task_project_root = task_script_dir.parent
cfg = toml.load(task_project_root / "pyproject.toml")
app_config = cfg["tool"]["flwr"]["app"]["config"]

# HHAR Configuration
DATA_ROOT = app_config.get("data-root", "hhar/Activity recognition exp")
USE_WATCHES = app_config.get("use-watches", True)
SAMPLE_RATE_HZ = app_config.get("sample-rate-hz", 50)
WINDOW_SECONDS = app_config.get("window-seconds", 2)
WINDOW_STRIDE_SECONDS = app_config.get("window-stride-seconds", 1)
NUM_CLASSES = app_config.get("num-classes", 6)

# Global seed for reproducibility
# Priority: ENV variable SEED > config file > default (42)
import random
GLOBAL_SEED = int(os.environ.get("SEED", app_config.get("seed", 42)))

# Set global seeds for reproducibility
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
print(f"[SEED] Using seed: {GLOBAL_SEED}")

# ───────────────────────── HHAR Data Processing ──────────────────────────

# Activity labels in order (align with Scaffold)
ACTIVITY_ORDER = ["walking", "sitting", "standing", "biking", "stairsup", "stairsdown"]

def _activity_to_int(lbl: str) -> int:
    """Convert activity label to integer. Returns -1 for unknown labels to be filtered out."""
    s = str(lbl).lower().strip()
    if s in ("null", "none", "", "nan"):
        return -1  # Mark for filtering
    else:
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

def ensure_hhar_dataset(data_root: str = "hhar", use_uci_official: bool = True, use_watches: bool = True):
    """Download HHAR dataset if not present using legacy UCI endpoints."""
    root_path = Path(data_root)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    phone_acc = list(root_path.rglob("Phones_accelerometer*.csv"))
    phone_gyro = list(root_path.rglob("Phones_gyroscope*.csv"))
    
    if phone_acc and phone_gyro:
        if use_watches:
            watch_acc = list(root_path.rglob("Watch_accelerometer*.csv"))
            watch_gyro = list(root_path.rglob("Watch_gyroscope*.csv"))
            if watch_acc and watch_gyro:
                return  # All files present
        else:
            return  # Phone files sufficient
    
    # Download using legacy UCI endpoints (modern ones return 404)
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/"
    files_to_download = ["Activity recognition exp.zip"]
    if use_watches:
        files_to_download.append("Still exp.zip")
    
    for filename in files_to_download:
        zip_path = root_path / filename
        if not zip_path.exists():
            print(f"Downloading {filename}...")
            url = base_url + filename.replace(" ", "%20")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_path)

def _get_cache_key(data_root: str, use_watches: bool, sample_rate_hz: int, 
                  window_seconds: int, window_stride_seconds: int) -> str:
    """Generate MD5 cache key for dataset configuration."""
    config_str = f"{data_root}_{use_watches}_{sample_rate_hz}_{window_seconds}_{window_stride_seconds}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

class HHARDataset(Dataset):
    """HHAR Dataset with memory-mapped loading."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        
        if normalize:
            # Per-channel z-score normalization (computed over random subset for efficiency)
            n_samples = min(20000, len(self.X))
            indices = np.random.choice(len(self.X), n_samples, replace=False)
            subset = self.X[indices]
            
            # Compute stats over (N, T) for each channel
            means = subset.mean(axis=(0, 2))  # Shape: (6,)
            stds = subset.std(axis=(0, 2))   # Shape: (6,)
            stds = np.where(stds == 0, 1.0, stds)  # Avoid division by zero
            self.means = means.reshape(6, 1)  # Shape: (6, 1)
            self.stds = stds.reshape(6, 1)   # Shape: (6, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if hasattr(self, 'means'):
            x = (x - self.means) / self.stds
        return torch.from_numpy(x), torch.from_numpy(np.array(self.y[idx]))

# ───────────────────────── Net ──────────────────────────
class Net(nn.Module):
    """
    Tiny 1D-CNN for HHAR windows. Input: (B, C=6, T=100), Output: (B, num_classes=6)
    Identical architecture across all federated learning projects.
    """
    def __init__(self, in_ch: int = 6, seq_len: int = 100, num_classes: int = 6):
        super().__init__()
        # Two 1D convolutional layers with BatchNorm and ReLU
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Adaptive pooling to get fixed size output
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification layer
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (B, 6, 100)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, 100)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 100)
        x = self.pool(x)                     # (B, 64, 1)
        x = x.view(x.size(0), -1)           # (B, 64)
        x = self.fc(x)                      # (B, 6)
        return x

# ─────────────────────── HHAR Processing Functions ───────────────────────

def _load_csvs(data_root: str, use_watches: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load phone/watch acc/gyro CSVs from UCI HHAR."""
    def _load_one(pattern: str) -> pd.DataFrame:
        files = [p for p in glob.glob(os.path.join(data_root, "**", pattern), recursive=True)
                 if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
        if not files:
            raise RuntimeError(f"Missing HHAR files for pattern: {pattern}")
        dfs = []
        for p in files:
            chunk_size = 50000
            chunks = []
            for chunk in pd.read_csv(
                p,
                chunksize=chunk_size,
                usecols=lambda c: c.strip().lower() in
                    {"user", "device", "model", "gt", "creation_time", "arrival_time", "x", "y", "z"},
                dtype={
                    "User": "string", "Device": "string", "Model": "string", "gt": "string",
                    "x": "float32", "y": "float32", "z": "float32"
                },
                low_memory=False,
            ):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk["src"] = "phone" if pattern.startswith("Phones") else "watch"
                chunk = chunk.iloc[::5].copy()
                chunks.append(chunk)
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    phone_acc  = _load_one("Phones_accelerometer*.csv")
    phone_gyro = _load_one("Phones_gyroscope*.csv")
    if use_watches:
        watch_acc  = _load_one("Watch_accelerometer*.csv")
        watch_gyro = _load_one("Watch_gyroscope*.csv")
        acc = pd.concat([phone_acc, watch_acc], ignore_index=True)
        gyro = pd.concat([phone_gyro, watch_gyro], ignore_index=True)
    else:
        acc, gyro = phone_acc, phone_gyro
    
    def norm(df: pd.DataFrame) -> pd.DataFrame:
        t = _col(df, ["timestamp", "Time", "time", "Creation_Time", "Arrival_Time"])
        ux = _col(df, ["User", "user"])
        dv = _col(df, ["Device", "device", "Model", "model"])
        act = _col(df, ["gt", "GT", "Activity", "activity"])
        x  = _col(df, ["x", "X"])
        y  = _col(df, ["y", "Y"])
        z  = _col(df, ["z", "Z"])
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df[t], errors="coerce"),
            "user": df[ux], "device": df[dv], "activity": df[act],
            "x": df[x], "y": df[y], "z": df[z], "src": df["src"]
        })
        return out.dropna(subset=["timestamp"])

    return norm(acc), norm(gyro)

def _prep_and_resample(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """Clean one sensor table and resample to target_hz in a memory-safe way."""
    cols = ["user", "device", "src", "timestamp", "x", "y", "z", "activity"]
    df = df[cols].dropna(subset=["timestamp"]).copy()
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype("float32", copy=False)
    df = df.sort_values(["user", "device", "src", "timestamp"])

    rule = f"{int(round(1000 / target_hz))}ms"
    out_frames = []
    
    for (u, d, s), g in df.groupby(["user", "device", "src"], observed=True):
        g = (
            g.groupby("timestamp", as_index=False, observed=True)
             .agg({"x": "mean", "y": "mean", "z": "mean", "activity": "last"})
        )
        g = g.set_index("timestamp")
        g = g[~g.index.duplicated(keep="last")].sort_index()

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
    a = _prep_and_resample(acc, target_hz).rename(columns={"x":"ax","y":"ay","z":"az"})
    g = _prep_and_resample(gyro, target_hz).rename(columns={"x":"gx","y":"gy","z":"gz"})
    m = pd.merge(a, g, on=["user","device","src","timestamp","activity"], how="inner").dropna()
    if m.empty:
        raise RuntimeError("Resampling produced no aligned rows. Check CSV columns and timestamps.")
    m["y"] = m["activity"].map(lambda s: _activity_to_int(s))
    # Filter out unknown labels (y >= 6 or y == -1)
    m = m[m["y"] >= 0].copy()
    m = m[m["y"] < 6].copy()
    return m[["user","device","src","timestamp","ax","ay","az","gx","gy","gz","y","activity"]]

def _windowize_with_users(df: pd.DataFrame, window_seconds: int = 2, stride_seconds: int = 1,
                          sample_rate_hz: int = 50) -> Tuple[np.ndarray, np.ndarray, list]:
    """Windowize and return per-window user IDs for user-based partitioning."""
    window_size = window_seconds * sample_rate_hz
    stride_size = stride_seconds * sample_rate_hz
    sensor_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    Xs, ys, users = [], [], []
    for (user, device), group in df.groupby(["user", "device"]):
        if len(group) < window_size:
            continue
        sensor_data = group[sensor_cols].values.astype(np.float32)
        activities = group["activity"].values
        for i in range(0, len(sensor_data) - window_size + 1, stride_size):
            window = sensor_data[i:i + window_size]
            labels_segment = activities[i:i + window_size]
            lbl_ints = [ _activity_to_int(l) for l in labels_segment ]
            # Filter out invalid labels
            valid_lbls = [l for l in lbl_ints if 0 <= l < 6]
            if not valid_lbls:  # Skip windows with no valid labels
                continue
            maj = int(np.bincount(np.array(valid_lbls, dtype=np.int64)).argmax())
            if maj >= 6:  # Skip unknown majority labels
                continue
            Xs.append(window.T)
            ys.append(maj)
            users.append(str(user))
    if not Xs:
        raise RuntimeError("No windows could be created from the data")
    return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64), users

def _windowize(df: pd.DataFrame, window_seconds: int = 2, stride_seconds: int = 1, 
              sample_rate_hz: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series data with majority-vote labels."""
    window_size = window_seconds * sample_rate_hz
    stride_size = stride_seconds * sample_rate_hz
    sensor_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    
    all_windows = []
    all_labels = []
    
    for (user, device), group in df.groupby(["user", "device"]):
        if len(group) < window_size:
            continue
        sensor_data = group[sensor_cols].values.astype(np.float32)
        activities = group["activity"].values
        
        for i in range(0, len(sensor_data) - window_size + 1, stride_size):
            window = sensor_data[i:i + window_size]
            labels_segment = activities[i:i + window_size]
            lbl_ints = [ _activity_to_int(l) for l in labels_segment ]
            # Filter out invalid labels
            valid_lbls = [l for l in lbl_ints if 0 <= l < 6]
            if not valid_lbls:  # Skip windows with no valid labels
                continue
            maj = int(np.bincount(np.array(valid_lbls, dtype=np.int64)).argmax())
            if maj >= 6:  # Skip unknown majority labels
                continue
            all_windows.append(window.T)
            all_labels.append(maj)
    
    if not all_windows:
        raise RuntimeError("No windows could be created from the data")
    
    return np.stack(all_windows, axis=0), np.array(all_labels, dtype=np.int64)

def load_hhar_data(data_root: str = "hhar/Activity recognition exp",
                   use_watches: bool = True,
                   sample_rate_hz: int = 50,
                   window_seconds: int = 2,
                   window_stride_seconds: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load and process HHAR data with caching."""
    
    # Check cache first
    cache_key = _get_cache_key(data_root, use_watches, sample_rate_hz, window_seconds, window_stride_seconds)
    cache_dir = Path("data/hhar_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hhar_{cache_key}.npz"
    
    if cache_file.exists():
        print(f"Loading cached HHAR data: {cache_file}")
        data = np.load(cache_file, mmap_mode='r')
        return data['X'], data['y']
    
    print("Processing HHAR data from scratch...")
    
    # Ensure dataset is downloaded
    ensure_hhar_dataset(data_root, use_watches=use_watches)
    
    # Load and merge CSVs
    acc, gyro = _load_csvs(data_root, use_watches)
    full_data = _resample_merge(acc, gyro, sample_rate_hz)
    
    # Create windows
    X, y = _windowize(full_data, window_seconds, window_stride_seconds, sample_rate_hz)
    del full_data
    gc.collect()
    
    # Cache the processed data
    print(f"Caching processed data: {cache_file}")
    np.savez_compressed(cache_file, X=X, y=y)
    
    return X, y

def load_hhar_windows_with_users(data_root: str = "hhar/Activity recognition exp",
                                 use_watches: bool = True,
                                 sample_rate_hz: int = 50,
                                 window_seconds: int = 2,
                                 window_stride_seconds: int = 1) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load HHAR and return windows, labels, and per-window user IDs (no caching)."""
    # Ensure dataset is available
    ensure_hhar_dataset(data_root, use_watches=use_watches)
    acc, gyro = _load_csvs(data_root, use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)
    full_data = merged
    gc.collect()
    X, y, users = _windowize_with_users(full_data, window_seconds, window_stride_seconds, sample_rate_hz)
    del full_data
    gc.collect()
    return X, y, users

# ─────────────────────── User-based partitioning ───────────────────────
def _get_user_partitions_from_windows() -> dict[str, List[int]]:
    """
    Create true user-based partitions for HHAR dataset using per-window user IDs.
    Returns mapping from user_id -> list of window indices.
    """
    
    # Load windows with per-window user IDs
    X, y, users = load_hhar_windows_with_users(
        data_root=DATA_ROOT,
        use_watches=USE_WATCHES,
        sample_rate_hz=SAMPLE_RATE_HZ,
        window_seconds=WINDOW_SECONDS,
        window_stride_seconds=WINDOW_STRIDE_SECONDS,
    )
    
    # Group indices by user
    user_to_indices = defaultdict(list)
    for idx, user in enumerate(users):
        user_to_indices[user].append(idx)
    
    return dict(user_to_indices)

# ─────────────────────── load_data ───────────────────────
def load_data(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    *,
    batch_size: int = 20,
    indices: Optional[List[int]] = None,
):
    """
    Return (trainloader, testloader, n_classes) for user-based HHAR partitioning.
    
    - dataset_flag: must be "hhar"
    - partition_id: integer in [0..num_partitions-1]
    - num_partitions: how many clients per leaf server (should be 3 for HierFL)
    - batch_size: DataLoader batch size (default 20 to match HierFL)
    - indices: if provided by hierarchical partitioning, use these indices directly
    """
    if dataset_flag.lower() != "hhar":
        raise ValueError("This loader only supports dataset_flag='hhar'")
    
    # Load full HHAR dataset
    X, y = load_hhar_data(
        data_root=DATA_ROOT,
        use_watches=USE_WATCHES,
        sample_rate_hz=SAMPLE_RATE_HZ,
        window_seconds=WINDOW_SECONDS,
        window_stride_seconds=WINDOW_STRIDE_SECONDS
    )
    
    # Create full dataset
    full_dataset = HHARDataset(X, y, normalize=True)
    
    # If explicit indices provided (HierFL hierarchical split), use them directly
    if indices is not None:
        client_indices = indices
    else:
        # REQUIRE PARTITIONS_JSON for user-based partitioning - no fallback
        import os
        partitions_json = os.environ.get("PARTITIONS_JSON")
        if not partitions_json or not os.path.exists(partitions_json):
            raise RuntimeError(
                "No PARTITIONS_JSON found. HierFL requires user-based partitioning. "
                "Run orchestrator.py to generate partitions first."
            )
        
        # Load partitions and extract indices for this client
        import json
        with open(partitions_json, 'r') as f:
            mapping = json.load(f)
        
        # For standalone usage, assume server_id=0 
        server_id = os.environ.get("SERVER_ID", "0")
        if str(server_id) not in mapping:
            raise RuntimeError(f"Server {server_id} not found in partitions")
        if str(partition_id) not in mapping[str(server_id)]:
            raise RuntimeError(f"Client {partition_id} not found for server {server_id}")
        
        client_indices = mapping[str(server_id)][str(partition_id)]
    
    # Split client data into train/test (80/20)
    n_client_samples = len(client_indices)
    n_train = int(0.8 * n_client_samples)
    
    # Shuffle indices for random train/test split
    np.random.seed(GLOBAL_SEED + partition_id)  # Deterministic but different per client
    shuffled_indices = np.random.permutation(client_indices)
    
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    
    # Create data loaders
    trainloader = DataLoader(
        Subset(full_dataset, train_indices), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    testloader = DataLoader(
        Subset(full_dataset, test_indices), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return trainloader, testloader, NUM_CLASSES

# ─────────────────── train / test / weights ───────────────────
def train(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device):
    """
    Local training on *epochs* mini-batch updates (matches HierFL).
    
    In the original HierFL implementation the hyper-parameter `num_local_update`
    counts **optimizer steps**, not full passes over the dataset. This revised
    loop keeps the same semantics: each iteration consumes exactly one batch; if
    we reach the end of the loader we simply restart it.
    """
    net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    ce = nn.CrossEntropyLoss()
    net.train()

    data_iter = iter(loader)
    step = 0
    loss = 0.0
    while step < epochs:  # epochs == #updates
        try:
            signals, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            signals, labels = next(data_iter)
        
        signals, labels = signals.to(device), labels.to(device)
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        # Cast to required dtype for CrossEntropyLoss
        labels = labels.long()
        
        opt.zero_grad()
        loss = ce(net(signals), labels)
        loss.backward()
        opt.step()
        scheduler.step()  # decay LR every update, as in HierFL
        step += 1
    
    return float(loss)

def test(net: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluate model on test data."""
    net.to(device)
    ce = nn.CrossEntropyLoss()
    net.eval()
    total_samples, correct, loss_sum = 0, 0, 0.0
    
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = labels.squeeze(-1)
            # Cast to required dtype for CrossEntropyLoss
            labels = labels.long()
            
            out = net(signals)
            loss_sum += ce(out, labels).item()
            correct += (out.argmax(1) == labels).sum().item()
            total_samples += len(labels)
    
    return (loss_sum / len(loader), correct / total_samples)

def get_weights(net: nn.Module):
    """Extract model weights as numpy arrays."""
    return [v.cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, w):
    """Set model weights from numpy arrays."""
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), w)}))
