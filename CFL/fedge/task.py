# fedge/task.py — HHAR dataset processing and model definition (NO DIRICHLET)

from __future__ import annotations
import os, glob
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Cache constants
CACHE_ROOT = os.path.join("data", "hhar_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

# Cache helper functions
USERS_META = os.path.join(CACHE_ROOT, "users.json")

def _save_users_meta(users: list[str]):
    import json
    with open(USERS_META, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def _load_users_meta():
    import json
    if os.path.exists(USERS_META):
        with open(USERS_META, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _user_cache_paths(user_id: str, win=2, stride=1, hz=50):
    tag = f"w{win}_s{stride}_hz{hz}"
    tr = os.path.join(CACHE_ROOT, f"user_{user_id}_{tag}_train.npz")
    te = os.path.join(CACHE_ROOT, f"user_{user_id}_{tag}_test.npz")
    return tr, te

def _save_dataset_npz(path: str, loader: torch.utils.data.DataLoader):
    # Materialize once, then reuse forever via memmap loads
    Xs, Ys = [], []
    for xb, yb in loader:
        # xb: (B, C, T) → keep as float32; yb: (B,)
        Xs.append(xb.detach().cpu().numpy().astype(np.float32, copy=False))
        Ys.append(yb.detach().cpu().numpy().astype(np.int64,  copy=False))
    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 6, 100), dtype=np.float32)
    y = np.concatenate(Ys, axis=0) if Ys else np.empty((0,), dtype=np.int64)
    np.savez_compressed(path, X=X, y=y)

# -----------------------------
# Dataset presence check
# -----------------------------
def ensure_hhar_dataset(base: str, use_uci_official: bool = True, use_watches: bool = True) -> None:
    """
    Verify HHAR CSVs exist under `base`. Never downloads.
    Requires 'Activity recognition exp' folder with Phones_* (and optionally Watch_*).
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
            raise RuntimeError("Watch CSVs missing but use-watches=true. Either extract them or set use-watches=false.")

# -----------------------------
# Model (exact same as SCAFFOLD tiny 1D-CNN)
# -----------------------------
class Net(nn.Module):
    """
    Tiny 1D-CNN for HHAR windows. Input: (B, C=6, T), Output: (B, num_classes)
    """
    USE_LOGREG = False  # set True only if you want the ultra-basic logistic regression head

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
            x = self.pool(x).squeeze(-1)  # (B, C)
            return self.head(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)      # (B, 64)
        return self.head(x)

# -----------------------------
# Labels/util
# -----------------------------
ACTIVITY_ORDER = [
    "walking", "sitting", "standing", "biking", "stairsup", "stairsdown"
]

def _label_to_int(lbl: str) -> int:
    s = str(lbl).strip().lower().replace(" ", "")
    # Check for null/unknown labels first
    if s in ("null", "none", "", "nan"):
        return -1  # Mark for filtering
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
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    raise KeyError(f"none of {names} found in columns={list(df.columns)}")

# -----------------------------
# Load + merge phone/watch CSVs
# -----------------------------
def _load_csvs(base: str, use_watches: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _load_one(pattern: str) -> pd.DataFrame:
        files = [p for p in glob.glob(os.path.join(base, "**", pattern), recursive=True)
                 if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
        if not files:
            raise RuntimeError(f"Missing HHAR files for pattern: {pattern}")
        dfs = []
        for p in files:
            # Load in chunks to avoid memory issues with large files
            chunk_size = 50000  # Process 50k rows at a time
            chunks = []
            for chunk in pd.read_csv(
                p,
                chunksize=chunk_size,
                usecols=lambda c: c.strip().lower() in
                    {"time","timestamp","creation_time","arrival_time",
                     "user","gt","activity","device","model","x","y","z"},
                dtype={
                    "user": "string", "device": "string", "model": "string",
                    "x": "float32", "y": "float32", "z": "float32"
                },
                low_memory=False,
            ):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk["src"] = "phone" if pattern.startswith("Phones") else "watch"
                # Downsample to reduce pressure (keep your 1/5 sampling)
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
    return acc, gyro

def _resample_merge(acc: pd.DataFrame, gyro: pd.DataFrame, hz: int) -> pd.DataFrame:
    # normalize columns
    def norm(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        t = _col(df, ["timestamp", "Time", "time", "Creation_Time", "Arrival_Time"])
        ux = _col(df, ["User", "user"])
        dv = _col(df, ["Device", "device", "Model", "model"])
        act = _col(df, ["gt", "GT", "Activity", "activity"])
        x  = _col(df, ["x", "X"])
        y  = _col(df, ["y", "Y"])
        z  = _col(df, ["z", "Z"])
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df[t], errors="coerce"),
            "user": df[ux].astype(str),
            "device": df[dv].astype(str),
            "activity": df[act],  # keep raw for now; don't cast to str yet
            "x": df[x].astype(float),
            "y": df[y].astype(float),
            "z": df[z].astype(float),
            "src": df.get("src", kind),
        })
        out = out.dropna(subset=["timestamp"])

        # --- Clean activity labels BEFORE resampling ---
        # normalize case/whitespace where present; leave true NaNs as NaN
        out["activity"] = out["activity"].astype("string").str.strip().str.lower()
        # convert common "missing" tokens into NaN
        out["activity"] = out["activity"].replace(
            {"nan": pd.NA, "null": pd.NA, "none": pd.NA, "": pd.NA}
        )
        # forward/backward fill activity within each (user, device, src) ordered by time
        out = out.sort_values("timestamp")
        out["activity"] = (
            out.groupby(["user","device","src"], observed=True)["activity"]
              .ffill().bfill()
        )
        # drop any rows that are still missing activity
        out = out.dropna(subset=["activity"])

        return out

    acc = norm(acc, "acc")
    gyro = norm(gyro, "gyro")
    
    # Additional memory optimization: limit total dataset size
    max_rows = 500000  # Limit to 500k rows per dataframe
    if len(acc) > max_rows:
        acc = acc.sample(n=max_rows, random_state=42).sort_values("timestamp")
    if len(gyro) > max_rows:
        gyro = gyro.sample(n=max_rows, random_state=42).sort_values("timestamp")

    # resample per (user, device, src)
    def resample(df: pd.DataFrame, hz: int) -> pd.DataFrame:
        g = []
        for (u, d, s), grp in df.groupby(["user", "device", "src"], observed=True):
            # Limit group size to prevent memory issues
            if len(grp) > 100000:  # If group is too large, sample it
                grp = grp.sample(n=100000, random_state=42).sort_values("timestamp")
            else:
                grp = grp.sort_values("timestamp")
            
            # Deduplicate per timestamp
            grp = grp.groupby("timestamp", as_index=False).agg({
                "x": "mean", "y": "mean", "z": "mean", "activity": "last"
            })
            
            # Further limit after deduplication
            if len(grp) > 50000:
                grp = grp.iloc[::2]  # Take every other row
            
            # Resample (numeric-only mean + ffill activity)
            freq = f"{int(1000/hz)}ms"
            idxed = grp.set_index("timestamp")
            num = idxed[["x","y","z"]].resample(freq).mean().interpolate()
            act = idxed[["activity"]].resample(freq).ffill()
            
            # Join back
            grp = num.join(act)
            
            grp = grp.reset_index()
            grp["user"], grp["device"], grp["src"] = u, d, s
            g.append(grp)
        return pd.concat(g, ignore_index=True) if g else df

    acc_r  = resample(acc, hz)
    gyro_r = resample(gyro, hz)
    # merge nearest by timestamp
    merged = pd.merge_asof(
        acc_r.sort_values("timestamp"),
        gyro_r.sort_values("timestamp"),
        on="timestamp",
        by=["user","device","src"],
        direction="nearest",
        suffixes=("_acc","_gyro"),
        tolerance=pd.Timedelta(milliseconds=int(1000/hz))
    )
    merged = merged.dropna(subset=["x_acc","y_acc","z_acc","x_gyro","y_gyro","z_gyro"])
    # ensure the activity label from the accelerometer stream is present
    merged = merged.dropna(subset=["activity_acc"])
    merged["y"] = merged["activity_acc"].apply(_label_to_int)
    # Filter out unknown labels (y >= 6 or y == -1)
    merged = merged[merged["y"] >= 0].copy()
    merged = merged[merged["y"] < 6].copy()
    merged = merged.rename(columns={
        "x_acc":"ax", "y_acc":"ay", "z_acc":"az",
        "x_gyro":"gx", "y_gyro":"gy", "z_gyro":"gz"
    })
    return merged[["timestamp","user","device","src","y","ax","ay","az","gx","gy","gz"]]

def _windowize(merged: pd.DataFrame, win_sec: int, stride_sec: int, hz: int
              ) -> List[Tuple[pd.DataFrame, str, str]]:
    # group by user+src to keep natural heterogeneity
    frames = []
    for (u, s), grp in merged.groupby(["user","src"], observed=True):
        frames.append((grp.reset_index(drop=True), u, s))
    return frames

class WindowedSeq(Dataset):
    def __init__(self, frames: list[tuple[pd.DataFrame, str, str]], window_sec: int, stride_sec: int, hz: int, normalize=True, means=None, stds=None):
        self.T = window_sec * hz
        self.S = stride_sec * hz
        self.samples = []   # (arr, labels, start)
        self.means = None
        self.stds = None

        for df, u, s in frames:
            arr = df[["ax","ay","az","gx","gy","gz"]].to_numpy(dtype=np.float32, copy=False)
            lab = df["y"].to_numpy(dtype=np.int64, copy=False)
            for start in range(0, len(arr) - self.T + 1, self.S):
                self.samples.append((arr, lab, start))

        if normalize:
            if means is not None and stds is not None:
                self.means = np.asarray(means, dtype=np.float32)
                self.stds  = np.asarray(stds,  dtype=np.float32)
            elif self.samples:
                idx = np.linspace(0, len(self.samples)-1, num=min(20000, len(self.samples)), dtype=int)
                cat = np.vstack([self.samples[i][0][self.samples[i][2]:self.samples[i][2]+self.T] for i in idx])
                self.means = cat.mean(axis=0)
                self.stds  = cat.std(axis=0) + 1e-8

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        arr, lab, start = self.samples[i]
        x = arr[start:start+self.T].T  # (6, T)
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
# Public API (used by server/client apps)
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
    # --- FAST PATH: if we've already built metadata + cache, skip CSV entirely ---
    users_meta = _load_users_meta()
    if users_meta is not None and partition_id < len(users_meta):
        user_for_client = users_meta[partition_id]
        tr_path, te_path = _user_cache_paths(user_for_client, window_seconds, window_stride_seconds, sample_rate_hz)
        if os.path.exists(tr_path) and os.path.exists(te_path):
            tr_npz = np.load(tr_path, mmap_mode="r")
            te_npz = np.load(te_path, mmap_mode="r")

            class _NPZDataset(Dataset):
                def __init__(self, X, y):
                    self.X, self.y = X, y
                def __len__(self): return self.X.shape[0]
                def __getitem__(self, i):
                    return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

            trainloader = DataLoader(_NPZDataset(tr_npz["X"], tr_npz["y"]),
                                     batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            testloader  = DataLoader(_NPZDataset(te_npz["X"], te_npz["y"]),
                                     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            return trainloader, testloader, num_classes

    # Ensure HHAR exists
    ensure_hhar_dataset(data_root, use_uci_official=True, use_watches=use_watches)
    # Load & prep
    acc, gyro = _load_csvs(data_root, use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)
    frames = _windowize(merged, window_seconds, window_stride_seconds, sample_rate_hz)

    # Clients = users (no Dirichlet)
    user_frames: Dict[str, List[tuple[pd.DataFrame,str,str]]] = {}
    for df, u, s in frames:
        user_frames.setdefault(u, []).append((df, u, s))
    users = sorted(list(user_frames.keys()))
    assert len(users) >= num_partitions, f"Need >= {num_partitions} users; found {len(users)}"
    
    if _load_users_meta() is None:
        _save_users_meta(users)

    # deterministic mapping partition->user
    user_for_client = users[partition_id]
    client_frames = user_frames[user_for_client]

    # 80/20 split per client
    total_frames = len(client_frames)
    cut = int(0.8 * total_frames)
    train_frames = client_frames[:cut]
    test_frames  = client_frames[cut:] if cut < total_frames else client_frames[-max(1, total_frames//5):]

    train_ds = WindowedSeq(train_frames, window_seconds, window_stride_seconds, sample_rate_hz, normalize=True)
    # Reuse train stats for test
    test_ds  = WindowedSeq(test_frames,  window_seconds, window_stride_seconds, sample_rate_hz,
                           normalize=True, means=train_ds.means, stds=train_ds.stds)

    # --- CACHE BLOCK: materialize once, then memory-map in future runs ---
    tr_path, te_path = _user_cache_paths(user_for_client, window_seconds, window_stride_seconds, sample_rate_hz)
    if not (os.path.exists(tr_path) and os.path.exists(te_path)):
        # Build loaders once to save NPZ
        _tmp_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        _tmp_te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
        _save_dataset_npz(tr_path, _tmp_tr)
        _save_dataset_npz(te_path, _tmp_te)

    # Reload from NPZ with memmap and build final DataLoaders (no Pandas now)
    tr_npz = np.load(tr_path, mmap_mode="r")
    te_npz = np.load(te_path, mmap_mode="r")

    class _NPZDataset(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i):
            # X saved as (N, C, T) already
            return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

    trainloader = DataLoader(_NPZDataset(tr_npz["X"], tr_npz["y"]),
                             batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testloader  = DataLoader(_NPZDataset(te_npz["X"], te_npz["y"]),
                             batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return trainloader, testloader, num_classes

@torch.no_grad()
def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.eval()
    net.to(device)
    ce = nn.CrossEntropyLoss(reduction="mean")
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()
            logits = net(x)
            loss = ce(logits, y)
            loss_sum += float(loss.item()) * int(y.shape[0])
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.shape[0])
    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    return float(avg_loss), float(acc)

def get_weights(net: nn.Module):
    return [v.detach().cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, weights):
    state = net.state_dict()
    for (k, _), w in zip(state.items(), weights):
        state[k] = torch.tensor(w)
    net.load_state_dict(state, strict=True)
