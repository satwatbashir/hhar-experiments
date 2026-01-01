# fedge/task.py — HHAR dataset pipeline and tiny 1D CNN (consistent across projects)
from __future__ import annotations
import logging
import os
import glob
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
from collections import OrderedDict

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import toml
import gc

logger = logging.getLogger(__name__)

# Config helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _get_hier_cfg() -> Dict:
    return toml.load(PROJECT_ROOT / "pyproject.toml")["tool"]["flwr"]["hierarchy"]

# Load HHAR app config
_cfg = toml.load(PROJECT_ROOT / "pyproject.toml")
_app_cfg = _cfg.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
DATA_ROOT = _app_cfg.get("data-root", "hhar/Activity recognition exp")
USE_WATCHES = bool(_app_cfg.get("use-watches", True))
SAMPLE_RATE_HZ = int(_app_cfg.get("sample-rate-hz", 50))
WINDOW_SECONDS = int(_app_cfg.get("window-seconds", 2))
WINDOW_STRIDE_SECONDS = int(_app_cfg.get("window-stride-seconds", 1))
NUM_CLASSES = int(_app_cfg.get("num-classes", 6))

# ------------------------------ HHAR dataset ------------------------------
def ensure_hhar_dataset(data_root: str = DATA_ROOT, use_watches: bool = USE_WATCHES) -> None:
    root_path = Path(data_root)
    root_path.mkdir(parents=True, exist_ok=True)
    phone_acc = list(root_path.rglob("Phones_accelerometer*.csv"))
    phone_gyro = list(root_path.rglob("Phones_gyroscope*.csv"))
    if phone_acc and phone_gyro:
        if use_watches:
            watch_acc = list(root_path.rglob("Watch_accelerometer*.csv"))
            watch_gyro = list(root_path.rglob("Watch_gyroscope*.csv"))
            if watch_acc and watch_gyro:
                return
        else:
            return
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/"
    files = ["Activity recognition exp.zip"]
    if use_watches:
        files.append("Still exp.zip")
    for filename in files:
        zip_path = root_path / filename
        if not zip_path.exists():
            logger.info(f"Downloading {filename}…")
            url = base_url + filename.replace(" ", "%20")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Extracting {filename}…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root_path)

def _col(df: pd.DataFrame, names: List[str]) -> str:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} found in columns={list(df.columns)}")

def _load_csvs(data_root: str, use_watches: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                usecols=lambda c: c.strip().lower() in {"user","device","model","gt","creation_time","arrival_time","x","y","z"},
                dtype={"User":"string","Device":"string","Model":"string","gt":"string","x":"float32","y":"float32","z":"float32"},
                low_memory=False,
            ):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk["src"] = "phone" if pattern.startswith("Phones") else "watch"
                chunk = chunk.iloc[::5].copy()
                chunks.append(chunk)
            if chunks:
                dfs.append(pd.concat(chunks, ignore_index=True))
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
        t = _col(df, ["timestamp","Time","time","Creation_Time","Arrival_Time"])
        ux = _col(df, ["User","user"])
        dv = _col(df, ["Device","device","Model","model"])
        act = _col(df, ["gt","GT","Activity","activity"])
        x  = _col(df, ["x","X"])
        y  = _col(df, ["y","Y"])
        z  = _col(df, ["z","Z"])
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df[t], errors="coerce"),
            "user": df[ux], "device": df[dv], "activity": df[act],
            "x": df[x], "y": df[y], "z": df[z], "src": df["src"]
        })
        return out.dropna(subset=["timestamp"])

    return norm(acc), norm(gyro)

def _prep_and_resample(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    cols = ["user","device","src","timestamp","x","y","z","activity"]
    df = df[cols].dropna(subset=["timestamp"]).copy()
    df[["x","y","z"]] = df[["x","y","z"]].astype("float32", copy=False)
    df = df.sort_values(["user","device","src","timestamp"])
    rule = f"{int(round(1000/target_hz))}ms"
    out_frames = []
    for (u,d,s), g in df.groupby(["user","device","src"], observed=True):
        g = g.groupby("timestamp", as_index=False, observed=True).agg({"x":"mean","y":"mean","z":"mean","activity":"last"})
        g = g.set_index("timestamp")
        g = g[~g.index.duplicated(keep="last")].sort_index()
        num = g[["x","y","z"]].astype("float32").resample(rule).mean().ffill().bfill()
        act = g[["activity"]].reindex(num.index, method="ffill")
        out = num.copy(); out["activity"] = act["activity"]; out["user"] = u; out["device"] = d; out["src"] = s
        out_frames.append(out.reset_index())
    if not out_frames:
        raise RuntimeError("No resampled data produced")
    return pd.concat(out_frames, ignore_index=True)

ACTIVITY_ORDER = ["walking","sitting","standing","biking","stairsup","stairsdown"]
def _activity_to_int(lbl: str) -> int:
    s = str(lbl).lower().strip()
    # Check for null/unknown labels first
    if s in ("null", "none", "", "nan"):
        return -1  # Mark for filtering
    alias = {"walk":"walking","cycling":"biking","bike":"biking","sit":"sitting","stand":"standing","upstairs":"stairsup","downstairs":"stairsdown","stairup":"stairsup","stairdown":"stairsdown"}
    s = alias.get(s, s)
    if s not in ACTIVITY_ORDER:
        return -1  # Mark for filtering instead of raising error
    return ACTIVITY_ORDER.index(s)

def _resample_merge(acc: pd.DataFrame, gyro: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    if acc.empty or gyro.empty:
        raise RuntimeError("Missing accelerometer or gyroscope CSVs")
    a = _prep_and_resample(acc, target_hz).rename(columns={"x":"ax","y":"ay","z":"az"})
    g = _prep_and_resample(gyro, target_hz).rename(columns={"x":"gx","y":"gy","z":"gz"})
    m = pd.merge(a, g, on=["user","device","src","timestamp","activity"], how="inner").dropna()
    if m.empty:
        raise RuntimeError("Resampling produced no aligned rows")
    m["y"] = m["activity"].map(lambda s: _activity_to_int(s))
    # Filter out unknown labels (y >= 6 or y == -1)
    m = m[m["y"] >= 0].copy()
    m = m[m["y"] < 6].copy()
    return m[["user","device","src","timestamp","ax","ay","az","gx","gy","gz","y","activity"]]

def _windowize_with_users(
    df: pd.DataFrame,
    window_seconds: int = WINDOW_SECONDS,
    stride_seconds: int = WINDOW_STRIDE_SECONDS,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create sliding windows and return per-window user IDs for partitioning."""
    window_size = window_seconds * sample_rate_hz
    stride = stride_seconds * sample_rate_hz
    sensor_cols = ["ax","ay","az","gx","gy","gz"]
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    users: List[str] = []
    for (user, device), g in df.groupby(["user","device"]):
        vals = g[sensor_cols].values.astype(np.float32)
        acts = g["activity"].values
        if len(vals) < window_size:
            continue
        for i in range(0, len(vals) - window_size + 1, stride):
            w = vals[i:i+window_size]
            lbls = acts[i:i+window_size]
            lbl_ints = [_activity_to_int(a) for a in lbls]
            # Filter out invalid labels
            valid_lbls = [l for l in lbl_ints if 0 <= l < 6]
            if not valid_lbls:  # Skip windows with no valid labels
                continue
            maj = int(np.bincount(np.array(valid_lbls, dtype=np.int64)).argmax())
            if maj >= 6:  # Skip unknown majority labels
                continue
            Xs.append(w.T)
            ys.append(maj)
            users.append(str(user))
    if not Xs:
        raise RuntimeError("No windows created from HHAR")
    return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64), users

def load_hhar_windows_with_users() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, users) without caching, for partition construction."""
    ensure_hhar_dataset(DATA_ROOT, USE_WATCHES)
    acc, gyro = _load_csvs(DATA_ROOT, USE_WATCHES)
    merged = _resample_merge(acc, gyro, SAMPLE_RATE_HZ)
    X, y, users = _windowize_with_users(merged)
    return X, y, users

class HHARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True, means: np.ndarray | None = None, stds: np.ndarray | None = None):
        # Avoid unnecessary copies: ensure dtypes but keep memory mapping
        self.X = X if getattr(X, "dtype", None) == np.float32 else X.astype(np.float32, copy=False)
        self.y = y if getattr(y, "dtype", None) == np.int64  else y.astype(np.int64,  copy=False)
        if normalize:
            if means is not None and stds is not None:
                stds = np.where(stds == 0, 1.0, stds)
                self.means = means.reshape(-1, 1).astype(np.float32, copy=False)
                self.stds  = stds.reshape(-1, 1).astype(np.float32, copy=False)
            else:
                # Fallback: compute once per process if cached stats are not present
                n_samples = min(20000, len(self.X))
                idx = np.random.choice(len(self.X), n_samples, replace=False)
                subset = self.X[idx]
                means = subset.mean(axis=(0,2))
                stds = subset.std(axis=(0,2))
                stds = np.where(stds == 0, 1.0, stds)
                self.means = means.reshape(-1,1)
                self.stds = stds.reshape(-1,1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if hasattr(self, 'means'):
            x = (x - self.means) / self.stds
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.int64)

def _get_cache_key() -> str:
    s = f"{DATA_ROOT}_{USE_WATCHES}_{SAMPLE_RATE_HZ}_{WINDOW_SECONDS}_{WINDOW_STRIDE_SECONDS}"
    return hashlib.md5(s.encode()).hexdigest()[:8]

def load_hhar_data() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (X, y, means, stds) with NPZ caching and memory mapping.
    X shape=(N,6,T), y shape=(N,), means/stds shape=(6,) when available.
    """
    cache_dir = PROJECT_ROOT / "data" / "hhar_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hhar_{_get_cache_key()}.npz"
    if cache_file.exists():
        data = np.load(cache_file, mmap_mode='r')
        X = data['X']
        y = data['y']
        # Backward-compatible: means/stds may be absent in older caches
        if ('means' in data.files) and ('stds' in data.files):
            means = data['means']
            stds  = data['stds']
            return X, y, means, stds
        # Check for sidecar stats file to avoid rewriting large NPZ in place
        stats_path = cache_dir / f"hhar_{_get_cache_key()}_stats.npz"
        if stats_path.exists():
            s = np.load(stats_path)
            means = s['means']
            stds  = s['stds']
            try:
                s.close()
            except Exception:
                pass
            return X, y, means, stds
        # Compute once, then write compact sidecar stats for future processes
        means = X.mean(axis=(0,2)).astype(np.float32)
        stds  = X.std(axis=(0,2)).astype(np.float32)
        stds[stds == 0] = 1.0
        try:
            np.savez_compressed(stats_path, means=means, stds=stds)
        except Exception:
            pass  # Non-fatal; continue with in-memory stats
        return X, y, means, stds
    ensure_hhar_dataset(DATA_ROOT, USE_WATCHES)
    acc, gyro = _load_csvs(DATA_ROOT, USE_WATCHES)
    merged = _resample_merge(acc, gyro, SAMPLE_RATE_HZ)
    window_size = WINDOW_SECONDS * SAMPLE_RATE_HZ
    stride = WINDOW_STRIDE_SECONDS * SAMPLE_RATE_HZ
    sensor_cols = ["ax","ay","az","gx","gy","gz"]
    Xs, ys = [], []
    for (user, device), g in merged.groupby(["user","device"]):
        vals = g[sensor_cols].values.astype(np.float32)
        acts = g["activity"].values
        if len(vals) < window_size:
            continue
        for i in range(0, len(vals) - window_size + 1, stride):
            w = vals[i:i+window_size]
            lbls = acts[i:i+window_size]
            lbl_ints = [_activity_to_int(a) for a in lbls]
            # Filter out invalid labels
            valid_lbls = [l for l in lbl_ints if 0 <= l < 6]
            if not valid_lbls:  # Skip windows with no valid labels
                continue
            maj = int(np.bincount(np.array(valid_lbls, dtype=np.int64)).argmax())
            if maj >= 6:  # Skip unknown majority labels
                continue
            Xs.append(w.T)
            ys.append(maj)
    if not Xs:
        raise RuntimeError("No windows created from HHAR")
    X = np.stack(Xs, axis=0).astype(np.float32, copy=False)
    y = np.array(ys, dtype=np.int64)
    # Compute global channel stats once and store in cache
    means = X.mean(axis=(0,2)).astype(np.float32)
    stds  = X.std(axis=(0,2)).astype(np.float32)
    stds[stds == 0] = 1.0
    np.savez_compressed(cache_file, X=X, y=y, means=means, stds=stds)
    return X, y, means, stds

# ------------------------------ Model ------------------------------
class Net(nn.Module):
    """Tiny 1D CNN: Conv1d(6→64,k=5) → BN → ReLU → Conv1d(64→64,k=5) → BN → ReLU → GAP → Linear(64→6)."""
    def __init__(self, in_ch: int = 6, seq_len: int = 100, n_class: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, n_class)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.head(x)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    # Pull sensible defaults from [tool.flwr.system]
    try:
        sys_cfg = toml.load(PROJECT_ROOT / "pyproject.toml")["tool"]["flwr"]["system"]
        num_workers = int(sys_cfg.get("max_workers", num_workers))
    except Exception:
        pass
    # Pin if CUDA is present; leave off on pure CPU if you prefer
    pin = torch.cuda.is_available() or pin_memory
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

def subset(dataset: Dataset, indices: Sequence[int]) -> Dataset:
    return Subset(dataset, indices=list(indices))

def get_eval_loader(batch_size: int = 32) -> DataLoader:
    """Get HHAR centralized test DataLoader (20% tail split)."""
    X, y, means, stds = load_hhar_data()
    ds = HHARDataset(X, y, normalize=True, means=means, stds=stds)
    n = len(ds)
    n_train = int(0.8 * n)
    test_idx = np.arange(n_train, n, dtype=np.int64)
    # Use config-driven eval batch size if present
    try:
        eval_bsz = int(_get_hier_cfg().get("eval_batch_size", batch_size))
    except Exception:
        eval_bsz = batch_size
    pin = torch.cuda.is_available()
    return DataLoader(Subset(ds, test_idx), batch_size=eval_bsz, shuffle=False, pin_memory=pin)


def _to_plain_ints(arr: Sequence) -> List[int]:
    return [int(x if not hasattr(x, "item") else x.item()) for x in arr]


# Global partition cache to prevent repeated JSON loading
_PARTITION_CACHE = {}
_PARTITION_FILE_CACHE = None

def _load_partition_indices(server_id: int, partition_id: int) -> List[int]:
    """Load pre-created partition indices from JSON file with caching."""
    import json
    import os
    
    global _PARTITION_FILE_CACHE
    
    parts_json = os.getenv("PARTITIONS_JSON")
    if not parts_json:
        parts_json = str(PROJECT_ROOT / "rounds" / "partitions.json")
    
    # Cache the entire partition file to avoid repeated JSON parsing
    if _PARTITION_FILE_CACHE is None:
        logger.debug(f"📂 Loading partition file {parts_json} (first time only)")
        try:
            with open(parts_json, "r") as f:
                _PARTITION_FILE_CACHE = json.load(f)
            logger.info(f"✅ Cached partition file with {len(_PARTITION_FILE_CACHE)} servers")
        except FileNotFoundError:
            logger.error(f"❌ Partition file not found: {parts_json}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading partition file {parts_json}: {e}")
            return []
    
    # Check partition cache first
    cache_key = f"s{server_id}_c{partition_id}"
    if cache_key in _PARTITION_CACHE:
        logger.debug(f"🎯 Using cached partition for server={server_id}, client={partition_id}")
        return _PARTITION_CACHE[cache_key]
    
    try:
        indices = _PARTITION_FILE_CACHE[str(server_id)][str(partition_id)]
        _PARTITION_CACHE[cache_key] = indices
        # Reduced verbosity: only log first partition load per server
        if partition_id == 0:
            logger.info(f"✅ Loading partitions for server {server_id} ({len(indices)} samples for client 0)")
        return indices
        
    except KeyError as e:
        logger.error(f"❌ Partition key not found: server={server_id}, client={partition_id}")
        logger.error(f"Available servers: {list(_PARTITION_FILE_CACHE.keys()) if _PARTITION_FILE_CACHE else 'unknown'}")
        return []
    except Exception as e:
        logger.error(f"❌ Unexpected error loading partition {server_id}/{partition_id}: {e}")
        return []


# Remove _balanced_sampler - not needed for HHAR


def load_data(
    dataset_flag: str,             # kept for signature compatibility
    partition_id: int,
    num_partitions: int,
    *,
    batch_size: int = 16,
    indices: Optional[List[int]] = None,
    server_id: Optional[int] = None,
):
    if dataset_flag.lower() != "hhar":
        raise ValueError("This loader only supports dataset_flag='hhar'")

    # Load full HHAR windows and cached stats
    X, y, means, stds = load_hhar_data()
    full_ds = HHARDataset(X, y, normalize=True, means=means, stds=stds)

    # Determine indices for this client
    if indices is None:
        client_idx = _load_partition_indices(server_id or 0, partition_id)
    else:
        client_idx = indices

    # Split client data into train/test (80/20)
    n_client = len(client_idx)
    n_train = int(0.8 * n_client)
    np.random.seed(42 + (server_id or 0) * 100 + partition_id)
    shuffled = np.random.permutation(client_idx)
    train_idx = shuffled[:n_train]
    test_idx  = shuffled[n_train:]

    # Derive train/eval batch sizes from config if available
    try:
        hier = _get_hier_cfg()
        train_bsz = int(hier.get("batch_size", batch_size))
        eval_bsz  = int(hier.get("eval_batch_size", batch_size))
    except Exception:
        train_bsz, eval_bsz = batch_size, batch_size

    pin = torch.cuda.is_available()
    trainloader = DataLoader(Subset(full_ds, train_idx), batch_size=train_bsz, shuffle=True,  pin_memory=pin)
    testloader  = DataLoader(Subset(full_ds, test_idx),  batch_size=eval_bsz,  shuffle=False, pin_memory=pin)
    return trainloader, testloader, NUM_CLASSES


# ─────────────────────── Training / eval utils ─────────────────────────────
def _make_scheduler(opt: torch.optim.Optimizer, sched_type: str, lr: float):
    sched_type = sched_type.lower()
    if sched_type == "cosine":
        total_gr = int(os.getenv("TOTAL_GLOBAL_ROUNDS", "150"))
        return CosineAnnealingLR(opt, T_max=total_gr, eta_min=lr * 0.01)
    gamma = float(os.getenv("LR_GAMMA", "0.95"))
    return StepLR(opt, step_size=1, gamma=gamma)


def train(
    net: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    *,
    lr: Optional[float] = None,
    momentum: Optional[float] = None,
    weight_decay: Optional[float] = None,
    gamma: Optional[float] = None,  # kept for API compatibility
    clip_norm: Optional[float] = None,
    prox_mu: float = 0.0,
    ref_weights: Optional[List[np.ndarray]] = None,
    global_round: int = 0,
    scaffold_enabled: bool = False,
):
    net.to(device)

    # Read required training parameters from TOML - no fallbacks
    if lr is None or momentum is None or weight_decay is None or clip_norm is None:
        cfg = toml.load(PROJECT_ROOT / "pyproject.toml")
        hierarchy_config = cfg["tool"]["flwr"]["hierarchy"]
        
        if lr is None:
            lr = hierarchy_config["lr_init"]
        if momentum is None:
            momentum = hierarchy_config["momentum"]
        if weight_decay is None:
            weight_decay = hierarchy_config["weight_decay"]
        if clip_norm is None:
            clip_norm = hierarchy_config["clip_norm"]
    
    wd = weight_decay
    clip_val = clip_norm

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    sched = (
        None
        if global_round < int(os.getenv("WARMUP_ROUNDS", "5"))  # Keep warmup default for now
        else _make_scheduler(opt, os.getenv("SCHEDULER_TYPE", "step"), lr)  # Keep scheduler default
    )

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for stability
    
    # Convert reference weights to tensors with proper validation
    ref_tensors = None
    if prox_mu > 0 and ref_weights:
        state_dict_keys = list(net.state_dict().keys())
        if len(ref_weights) == len(state_dict_keys):
            ref_tensors = []
            for key, ref_w in zip(state_dict_keys, ref_weights):
                current_param = net.state_dict()[key]
                ref_tensor = torch.tensor(ref_w, dtype=current_param.dtype, device=device)
                if ref_tensor.shape == current_param.shape:
                    ref_tensors.append(ref_tensor)
                else:
                    ref_tensors = None
                    break

    running_loss, total_batches = 0.0, 0
    net.train()
    
    for epoch in range(epochs):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.flatten().long().to(device)
            opt.zero_grad()
            
            # Forward pass with mixed precision for stability
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = net(imgs)
                loss = ce(logits, labels)
                
                # FedProx regularization term with parameter normalization
                if prox_mu > 0 and ref_tensors is not None:
                    model_params = list(net.parameters())
                    if len(ref_tensors) == len(model_params):
                        prox_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)
                        for p, w0 in zip(model_params, ref_tensors):
                            prox_loss += torch.sum((p - w0) ** 2)
                        
                        # Normalize by parameter count to prevent scale blow-ups
                        num_params = sum(p.numel() for p in model_params)
                        normalized_prox_loss = prox_loss / max(1, num_params)
                        loss = loss + (prox_mu / 2.0) * normalized_prox_loss

            loss.backward()
            
            # ENHANCED: Check for NaN gradients before SCAFFOLD correction
            nan_grads = any(torch.isnan(p.grad).any() for p in net.parameters() if p.grad is not None)
            if nan_grads:
                logger.warning(f"NaN gradients detected in epoch {epoch}, batch {batch_idx}")
                opt.zero_grad()
                continue
            
            if scaffold_enabled and hasattr(net, "_scaffold_manager"):
                net._scaffold_manager.apply_scaffold_correction(net, opt.param_groups[0]["lr"])

            # ENHANCED: Gradient clipping with diagnostic logging
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            if grad_norm > clip_val:
                logger.debug(f"Gradient clipped: norm={grad_norm:.4f} -> {clip_val}")
            
            opt.step()
            
            # ENHANCED: Check for NaN parameters after optimization step
            nan_params = any(torch.isnan(p).any() for p in net.parameters())
            if nan_params:
                logger.error(f"NaN parameters detected after optimization step in epoch {epoch}")
                return float('nan')

            running_loss += loss.item()
            total_batches += 1

    if sched:
        sched.step()
    return float(running_loss / max(total_batches, 1))


@torch.no_grad()
def test(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Enhanced evaluation with proper error handling and diagnostic logging."""
    net.to(device).eval()
    
    # Use float64 for precise accumulation
    loss_sum = 0.0
    correct = 0
    total = 0
    
    # Track min/max logits for diagnostic purposes
    min_logit = float('inf')
    max_logit = float('-inf')
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.flatten().long().to(device)
        
        # Forward pass
        outputs = net(imgs)
        
        # Diagnostic: Check for NaN/inf in logits
        logit_min = outputs.min().item()
        logit_max = outputs.max().item()
        min_logit = min(min_logit, logit_min)
        max_logit = max(max_logit, logit_max)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logger.warning(f"NaN/inf detected in logits at batch {batch_idx}: min={logit_min:.4f}, max={logit_max:.4f}")
        
        # Use F.cross_entropy with reduction="sum" for proper accumulation
        batch_loss = F.cross_entropy(outputs, labels, reduction="sum").item()
        loss_sum += batch_loss
        
        # Accuracy calculation
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Guard against division by zero
    if total == 0:
        logger.error("No samples processed during evaluation!")
        return float('nan'), 0.0
    
    # Log diagnostic information
    logger.debug(f"Eval diagnostics: logit_range=[{min_logit:.4f}, {max_logit:.4f}], total_samples={total}")
    
    avg_loss = loss_sum / total  # Per-sample loss
    accuracy = correct / total
    
    return avg_loss, accuracy


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in net.state_dict().values()]


def set_weights(net: nn.Module, weights: Sequence[np.ndarray]):
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), weights)}))

