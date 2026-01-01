# fedge/task.py — HHAR ONLY (exact Scaffold implementation)
from __future__ import annotations

import os
import glob
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple
import requests

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ───────────────────────── Public flags ─────────────────────────
DATA_FLAGS = ["hhar"]  # HHAR ONLY

# Activity labels in order (matching Scaffold exactly)
ACTIVITY_ORDER = ["biking", "sitting", "standing", "walking", "stairsup", "stairsdown"]

def _activity_to_int(lbl: str) -> int:
    """Convert activity label to integer. Returns -1 for unknown labels to be filtered out."""
    s = str(lbl).lower().strip()
    if s in ("null", "none", "", "nan"):
        return -1  # Mark for filtering
    else:
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

def ensure_hhar_dataset(data_root: str = "hhar", use_uci_official: bool = True, use_watches: bool = True):
    """Download HHAR dataset if not present."""
    root_path = Path(data_root)
    
    # Check if we already have CSV files
    csv_files = list(root_path.glob("*.csv")) + list(root_path.glob("**/*.csv"))
    if csv_files:
        return
    
    # Create directory
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Download URLs (using legacy UCI endpoints that work)
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/"
    files = [
        "Activity recognition exp.zip",
        "Still exp.zip"
    ]
    
    print(f"Downloading HHAR dataset to {root_path}...")
    
    for filename in files:
        url = base_url + filename
        local_path = root_path / filename
        
        if local_path.exists():
            print(f"  {filename} already exists")
            continue
            
        print(f"  Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✓ Downloaded {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            continue
    
    # Extract zip files
    for zip_file in root_path.glob("*.zip"):
        print(f"Extracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(root_path)
            print(f"  ✓ Extracted {zip_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to extract {zip_file.name}: {e}")

def _load_csvs(data_root: str, use_watches: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load phone/watch acc/gyro CSVs from UCI HHAR - EXACT Scaffold implementation."""
    # Try typical UCI filenames - EXCLUDE Still exp files
    phone_acc  = [p for p in glob.glob(os.path.join(data_root, "**", "Phones_accelerometer*.csv"), recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    phone_gyro = [p for p in glob.glob(os.path.join(data_root, "**", "Phones_gyroscope*.csv"),    recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    watch_acc  = [p for p in glob.glob(os.path.join(data_root, "**", "Watch_accelerometer*.csv"),  recursive=True)
                  if ("\\Still exp\\" not in p and "/Still exp/" not in p)]
    watch_gyro = [p for p in glob.glob(os.path.join(data_root, "**", "Watch_gyroscope*.csv"),      recursive=True)
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

    # EXACT Scaffold logic: separate acc and gyro processing
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
    """Clean one sensor table and resample to target_hz in a memory-safe way (Scaffold exact)."""
    # Keep only what we need and ensure tight dtypes
    cols = ["user", "device", "src", "timestamp", "x", "y", "z", "activity"]
    df = df[cols].dropna(subset=["timestamp"]).copy()
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype("float32", copy=False)
    df = df.sort_values(["user", "device", "src", "timestamp"])

    rule = f"{int(round(1000 / target_hz))}ms"

    out_frames = []
    # observed=True prevents cartesian-product reindexing with categoricals
    for (user, device, src), group in df.groupby(["user", "device", "src"], observed=True):
        group = group.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        if len(group) < 2: continue

        # Resample with forward/backward fill
        try:
            resampled = group.resample(rule).first().ffill().bfill()
            if len(resampled) == 0: continue
            resampled = resampled.reset_index()
            resampled["user"] = user
            resampled["device"] = device
            resampled["src"] = src
            out_frames.append(resampled)
        except Exception as e:
            print(f"Resample failed for {user}/{device}/{src}: {e}")
            continue

    if not out_frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(out_frames, ignore_index=True)

def _resample_merge(acc: pd.DataFrame, gyro: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """Resample and merge accelerometer and gyroscope data (Scaffold exact)."""
    acc_clean = _prep_and_resample(acc, target_hz)
    gyro_clean = _prep_and_resample(gyro, target_hz)

    if acc_clean.empty or gyro_clean.empty:
        raise ValueError("No data after resampling")

    # Merge on (user, device, src, timestamp)
    merged = pd.merge(
        acc_clean, gyro_clean,
        on=["user", "device", "src", "timestamp"],
        suffixes=("_acc", "_gyro"),
        how="inner"
    )
    
    if merged.empty:
        raise ValueError("No overlapping timestamps after merge")

    return merged

def _windowize(merged: pd.DataFrame, window_sec: int, stride_sec: int, sample_rate: int) -> List[Tuple[pd.DataFrame, str, str]]:
    """Create sliding windows from merged data (Scaffold exact)."""
    T = window_sec * sample_rate
    stride = stride_sec * sample_rate
    
    frames = []
    
    # Group by user and device (Scaffold groups by user+device, not user+src)
    for (user, device), group in merged.groupby(["user", "device"], observed=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < T:
            continue
        
        # Create sliding windows
        for start in range(0, len(group) - T + 1, stride):
            window_df = group.iloc[start:start + T].copy()
            frames.append((window_df, str(user), str(device)))
    
    return frames

class HHARDataset(Dataset):
    """HHAR Dataset for PyTorch (Scaffold exact)."""
    
    def __init__(self, frames: List[Tuple[pd.DataFrame, str, str]], T: int, normalize: bool = True):
        self.T = T
        self.samples = []
        
        # Process frames into samples
        for df, user, device in frames:
            if len(df) < T:
                continue
            
            # Build 6-channel array: [x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro]
            sensor_data = np.column_stack([
                df["x_acc"].values, df["y_acc"].values, df["z_acc"].values,
                df["x_gyro"].values, df["y_gyro"].values, df["z_gyro"].values
            ]).astype(np.float32)
            
            # Get activity labels (use _acc version as primary)
            activity_col = "activity_acc" if "activity_acc" in df.columns else "activity"
            if activity_col not in df.columns:
                print(f"Warning: No activity column for user {user}, device {device}")
                continue
                
            labels = df[activity_col].values
            
            # Convert to integers using Scaffold's mapping
            label_ints = np.array([_activity_to_int(lbl) for lbl in labels])
            # Filter out unknown labels (>= 6 or == -1)
            valid_mask = (label_ints >= 0) & (label_ints < 6)
            if valid_mask.sum() > 0:  # Only keep if there are valid labels
                sensor_data = sensor_data[valid_mask]
                label_ints = label_ints[valid_mask]
                # Store sample: (sensor_data, labels, start_idx)
                self.samples.append((sensor_data, label_ints, 0))
        
        # Compute normalization statistics if requested
        self.means = None
        self.stds = None
        if normalize and self.samples:
            # compute global mean/std over a random subset to avoid huge scans
            idx = np.linspace(0, len(self.samples)-1, num=min(20000, len(self.samples)), dtype=int)
            cat = np.vstack([self.samples[i][0][self.samples[i][2]:self.samples[i][2]+self.T] for i in idx])
            self.means = cat.mean(axis=0)
            self.stds = cat.std(axis=0) + 1e-8

    def __len__(self): 
        return len(self.samples)

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

# ───────────────────────── Model (1D CNN) ───────────────────────
class HAR1DNet(nn.Module):
    """Tiny 1D CNN for HAR matching Scaffold. Accepts [B,C,T] and uses GAP."""
    def __init__(self, in_ch: int = 6, n_class: int = 6):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 5, padding=2, bias=False),  nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B,C,T] or [B,C,1,T] from image-shaped loaders
        if x.ndim == 4:  # [B,C,H,W] -> expect H=1
            b, c, h, w = x.shape
            if h != 1:
                x = x.view(b, c, h * w)
            else:
                x = x.squeeze(2)  # -> [B,C,T]
        elif x.ndim == 3:
            pass  # already [B,C,T]
        elif x.ndim == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")
        x = self.fe(x).squeeze(-1)
        return self.fc(x)

# Keep external API stable for other modules
class Net(nn.Module):
    """Shim so callers using Net(in_ch, img_h, img_w, n_class, ...) still work."""
    def __init__(self, in_ch: int = 6, img_h: int | None = None, img_w: int | None = None,
                 n_class: int = 6, seq_len: int | None = None):
        super().__init__()
        self.backbone = HAR1DNet(in_ch=in_ch, n_class=n_class)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# ──────────────────────── HHAR dataset loader ───────────────────
# We try to reuse an existing, proven implementation if you kept `task-templated.py`
# in the project root. Otherwise we expect a prebuilt cache NPZ at:
#   ./hhar/hhar_w{HHAR_WINDOW}_s{HHAR_STRIDE}_norm.npz
# containing arrays X:[N,6,T], y:[N].
def _maybe_import_hhar_from_template():
    tpl = Path("task-templated.py")
    if not tpl.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("task_templated", str(tpl.resolve()))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception:
        return None
    return getattr(mod, "HHARDataset", None)

def _ensure_csvs():
    """If HHAR csvs are not in HHAR_ROOT, try to extract the first zip in HHAR_ZIPDIR."""
    HHAR_ROOT.mkdir(parents=True, exist_ok=True)
    # Check for CSVs in HHAR_ROOT or subdirectories
    if list(HHAR_ROOT.glob("*.csv")) or list(HHAR_ROOT.glob("**/*.csv")):
        return
    zips = list(HHAR_ZIPDIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No CSVs in {HHAR_ROOT} and no zip in {HHAR_ZIPDIR}. "
            "Drop raw HHAR CSVs into ./hhar or a zip into ./hhar_zip."
        )
    import zipfile
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(HHAR_ROOT)

class _SimpleHHAR(Dataset):
    """Memory-efficient HHAR dataset that processes CSV files with caching and avoids OOM."""
    def __init__(self, root: Path, window: int, stride: int):
        self.root = root
        self.window = window
        self.stride = stride
        
        # Create cache file path
        cache_file = root / f"hhar_w{window}_s{stride}_processed.npz"
        
        # Try to load from cache first
        if cache_file.exists():
            print(f"Loading HHAR from cache: {cache_file}")
            cache = np.load(str(cache_file), allow_pickle=False, mmap_mode="r")
            self.X = torch.from_numpy(cache["X"]).float()
            self.y = torch.from_numpy(cache["y"]).long()
            self.users = cache["users"]
            self.n_class = int(cache["n_class"])
        else:
            print(f"Processing HHAR CSVs and creating cache: {cache_file}")
            self._process_csvs_to_cache(cache_file)
            # Load the newly created cache with memory mapping
            cache = np.load(str(cache_file), allow_pickle=False, mmap_mode="r")
            self.X = torch.from_numpy(cache["X"]).float()
            self.y = torch.from_numpy(cache["y"]).long()
            self.users = cache["users"]
            self.n_class = int(cache["n_class"])
        
        print(f"Loaded HHAR dataset: {len(self.X)} samples, {len(np.unique(self.users))} users")
    
    def _process_csvs_to_cache(self, cache_file: Path):
        """Process CSV files in chunks to avoid OOM and save to cache."""
        import pandas as pd
        
        # Find CSV files
        csv_files = list(self.root.glob("*.csv")) + list(self.root.glob("**/*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.root} or subdirectories")
        
        activity_map = {
            'bike': 0, 'sit': 1, 'stand': 2, 'walk': 3, 
            'stairsup': 4, 'stairsdown': 5, 'null': 6
        }
        
        all_X, all_y, all_users = [], [], []
        
        # Process each CSV file separately to avoid loading all at once
        for csv_file in csv_files:
            if not ("accelerometer" in csv_file.name.lower() or "gyroscope" in csv_file.name.lower()):
                continue
            
            print(f"Processing {csv_file.name}...")
            
            # Read CSV in chunks to avoid memory issues
            chunk_size = 50000  # Process 50k rows at a time
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                # Process this chunk
                for user in chunk['User'].unique():
                    user_data = chunk[chunk['User'] == user]
                    
                    for activity in user_data['gt'].unique():
                        if activity == 'null':
                            continue
                        
                        activity_data = user_data[user_data['gt'] == activity]
                        if len(activity_data) < self.window:
                            continue
                        
                        # Extract sensor values (x, y, z) - only what we need
                        sensor_values = activity_data[['x', 'y', 'z']].values.astype(np.float32)
                        
                        # Create windows with stride
                        for start_idx in range(0, len(sensor_values) - self.window + 1, self.stride):
                            window_data = sensor_values[start_idx:start_idx + self.window].copy()
                            
                            # Create 6-channel data (pad with zeros if needed)
                            if window_data.shape[1] == 3:
                                padded = np.zeros((self.window, 6), dtype=np.float32)
                                padded[:, :3] = window_data
                                window_data = padded
                            
                            # Normalize per channel (z-score)
                            for ch in range(window_data.shape[1]):
                                mean_ch = np.mean(window_data[:, ch])
                                std_ch = np.std(window_data[:, ch])
                                if std_ch > 1e-8:
                                    window_data[:, ch] = (window_data[:, ch] - mean_ch) / std_ch
                            
                            # Store in [C, T] format
                            all_X.append(window_data.T)  # Transpose to [C, T]
                            all_y.append(activity_map.get(activity, 6))
                            all_users.append(user)
                
                # Clear chunk from memory
                del chunk
        
        # Convert to numpy arrays
        X_array = np.stack(all_X, axis=0).astype(np.float32) if all_X else np.empty((0, 6, self.window), dtype=np.float32)
        y_array = np.array(all_y, dtype=np.int64)
        users_array = np.array(all_users, dtype='U1')  # Single character strings
        
        # Save to cache with compression
        np.savez_compressed(
            str(cache_file),
            X=X_array,
            y=y_array,
            users=users_array,
            n_class=len(activity_map)
        )
        
        # Clear temporary arrays
        del all_X, all_y, all_users, X_array, y_array, users_array
    
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, i): 
        return self.X[i], self.y[i]
    
    @property
    def user(self):
        return self.users

def _make_hhar_dataset() -> Dataset:
    """Return a Dataset yielding (x:[C,T] or [C,1,T], y) and exposing n_class."""
    # Prefer the user's proven template if present
    HHAR = _maybe_import_hhar_from_template()
    if HHAR is not None:
        # The template expects: root, window, stride, normalize, cache, cache_lock
        try:
            ds = HHAR(root=str(HHAR_ROOT), window=HHAR_WINDOW, stride=HHAR_STRIDE,
                      normalize=True, cache=0 if HHAR_CACHE_DISABLE else 1, cache_lock=True)
        except TypeError:
            # older/newer signatures: try fewer args
            ds = HHAR(root=str(HHAR_ROOT), window=HHAR_WINDOW, stride=HHAR_STRIDE)
        # Ensure attributes exist
        if not hasattr(ds, "n_class"):
            ys = [int(ds[i][1]) for i in range(min(1024, len(ds)))] if len(ds)>0 else []
            ds.n_class = max(ys)+1 if ys else 0
        return ds
    # Else, use simple CSV-based loader
    _ensure_csvs()  # best effort to make raw files available
    return _SimpleHHAR(HHAR_ROOT, HHAR_WINDOW, HHAR_STRIDE)

# ───────────────────── User-based Partitioning (Natural Heterogeneity) ───────────────────
def _encode_users_to_int(users_attr) -> np.ndarray:
    """Map arbitrary user identifiers (strings/objects) to contiguous int IDs [0..U-1]."""
    users_np = np.asarray(users_attr)
    # If it's a torch tensor, move to CPU numpy
    if hasattr(users_np, "cpu"):
        users_np = users_np.cpu().numpy()
    # Normalize to string to be safe
    users_np = users_np.astype(str)
    uniq = np.unique(users_np)
    lut = {u: i for i, u in enumerate(sorted(uniq))}
    return np.array([lut[u] for u in users_np], dtype=np.int64)

def _user_partition_indices(user_ids: np.ndarray, n_clients: int) -> List[List[int]]:
    """Map distinct users to client ids deterministically for natural heterogeneity."""
    # Map distinct users to client ids deterministically
    uniq = np.unique(user_ids)
    assert len(uniq) >= n_clients, f"Need at least {n_clients} users, got {len(uniq)}"
    # Assign each user to one client (round-robin)
    mapping = {u: i % n_clients for i, u in enumerate(sorted(uniq))}
    bins = [[] for _ in range(n_clients)]
    for idx, u in enumerate(user_ids):
        bins[mapping[u]].append(idx)
    return bins

# ───────────────────────── Public API (Scaffold-compatible) ───────────────────────────
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
    """Load HHAR data with Scaffold-compatible signature. Returns (trainloader, valloader, n_classes)."""
    # 0) Ensure HHAR dataset is available
    ensure_hhar_dataset(data_root, use_uci_official=True, use_watches=use_watches)
    
    # 1) Load CSVs and build merged stream
    acc, gyro = _load_csvs(data_root, use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)

    # 2) Group by user and device
    frames = _windowize(merged, window_seconds, window_stride_seconds, sample_rate_hz)

    # 3) Clients = users (9 users => 9 clients). No Dirichlet.
    user_frames = {}
    for df, user, device in frames:
        if user not in user_frames:
            user_frames[user] = []
        user_frames[user].append((df, user, device))
    
    users = sorted(list(user_frames.keys()))
    assert len(users) >= num_partitions, f"Need >= {num_partitions} users; found {len(users)}"

    # 4) Assign this client's user(s)
    my_user = users[partition_id % len(users)]
    my_frames = user_frames[my_user]
    
    print(f"Client {partition_id}: user={my_user}, {len(my_frames)} frame groups")

    # 5) Build dataset from this client's frames
    T = window_seconds * sample_rate_hz  # e.g., 2*50=100
    dataset = HHARDataset(my_frames, T, normalize=True)
    
    if len(dataset) == 0:
        print(f"Warning: Client {partition_id} has no data!")
        # Return empty loaders
        empty_dataset = TensorDataset(torch.empty(0, 6, T), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_dataset, batch_size=batch_size, shuffle=False)
        return empty_loader, empty_loader, num_classes

    # 6) 80/20 split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    
    # Deterministic split based on client ID
    torch.manual_seed(42 + partition_id)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    # 7) DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, valloader, num_classes

# ────────────────────── test / weights I/O ──────────────────────
def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.eval(); net.to(device)
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = (y if isinstance(y, torch.Tensor) else torch.tensor(y)).to(device).long().view(-1)
            # Accept 2D, 3D, or 4D tensors
            if x.ndim == 2:
                x = x.unsqueeze(1)          # [B,T]→[B,1,T]
            if x.ndim == 4 and x.shape[2] == 1:
                x = x.squeeze(2)            # [B,C,1,T]→[B,C,T]
            elif x.ndim == 3 and x.shape[1] > x.shape[2]:
                x = x.transpose(1, 2)       # [B,T,C]→[B,C,T]
            logits = net(x)
            loss_sum += ce(logits, y).item()
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)
            batches  += 1
    if total == 0 or batches == 0: 
        return 0.0, 0.0
    return loss_sum / batches, correct / total

def get_weights(net: nn.Module):
    return [v.detach().cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, weights):
    keys = list(net.state_dict().keys())
    sd = OrderedDict({k: torch.tensor(w) for k, w in zip(keys, weights)})
    net.load_state_dict(sd, strict=True)
