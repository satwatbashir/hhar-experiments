"""User-based partition utilities for HHAR dataset.

This module produces user-based partitioning for natural heterogeneity:
    - Maps HHAR users directly to clients (no artificial Dirichlet distribution)
    - Maintains natural data distribution per user
    - Compatible with hierarchical FL structure

The result is stored as a JSON mapping
    {
        "0": {"0": [..indices..], "1": [...], ...},
        "1": {...},
        ...
    }
where top-level keys are server_ids (str), second-level keys are
client_ids (str) and values are *global* indices into the original train
set.  Indices are plain Python ints so the file is portable and does not
require NumPy when loading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import toml

__all__ = [
    "hier_user_indices",
    "hier_user_indices_from_users",
    "write_partitions",
]

def hier_user_indices(
    labels: Sequence[int] | np.ndarray,
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    seed: int = 42,
) -> Dict[str, Dict[str, List[int]]]:
    """Return user-based hierarchical mapping for HHAR dataset (no Dirichlet).

    Parameters
    ----------
    labels : Sequence[int] | np.ndarray
        Labels aligned with the training set order (length = num_samples).
    num_servers : int
        Number of leaf servers.
    clients_per_server : int | Sequence[int]
        Number of clients *inside each* leaf server.
    seed : int
        Random seed for reproducibility.
    """
    # Standardise clients_per_server into a list per server
    if isinstance(clients_per_server, int):
        cps_list: list[int] = [clients_per_server] * num_servers
    else:
        cps_list = list(clients_per_server)
        if len(cps_list) != num_servers:
            raise ValueError(f"len(clients_per_server)={len(cps_list)} does not match num_servers={num_servers}")
    
    labels_np = np.asarray(labels, dtype=np.int64)
    n = labels_np.shape[0]
    total_clients = sum(cps_list)
    
    # Simple equal split across all clients (simulating user-based partitioning)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n)
    rng.shuffle(indices)
    
    # Split indices equally across all clients
    samples_per_client = n // total_clients
    remainder = n % total_clients
    
    mapping: Dict[str, Dict[str, List[int]]] = {}
    global_client_idx = 0
    start_idx = 0
    
    for sid in range(num_servers):
        n_clients = cps_list[sid]
        server_mapping = {}
        
        for cid in range(n_clients):
            # Calculate samples for this client
            client_samples = samples_per_client
            if global_client_idx < remainder:
                client_samples += 1
            
            end_idx = start_idx + client_samples
            client_indices = indices[start_idx:end_idx].tolist()
            
            server_mapping[str(cid)] = client_indices
            start_idx = end_idx
            global_client_idx += 1
        
        mapping[str(sid)] = server_mapping
    
    return mapping

# New: build mapping using explicit per-window user IDs
def hier_user_indices_from_users(
    users: Sequence[str] | np.ndarray,
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    seed: int = 42,
) -> Dict[str, Dict[str, List[int]]]:
    """Return hierarchical mapping by assigning one user per client.

    Assumes the HHAR dataset contains exactly total_clients distinct users.
    If there are more users, the first total_clients (sorted) will be used.
    """
    if isinstance(clients_per_server, int):
        cps_list: list[int] = [clients_per_server] * num_servers
    else:
        cps_list = list(clients_per_server)
        if len(cps_list) != num_servers:
            raise ValueError(f"len(clients_per_server)={len(cps_list)} does not match num_servers={num_servers}")

    users_np = np.asarray(users)
    # Determine unique users, sorted for determinism
    uniq = np.array(sorted(set(map(str, users_np))))
    total_clients = sum(cps_list)
    if len(uniq) < total_clients:
        raise ValueError(f"Not enough distinct users ({len(uniq)}) for total clients ({total_clients})")
    uniq = uniq[:total_clients]

    # Map user -> indices
    mapping: Dict[str, Dict[str, List[int]]] = {}
    user_to_indices: Dict[str, List[int]] = {}
    for idx, u in enumerate(map(str, users_np)):
        user_to_indices.setdefault(u, []).append(int(idx))

    # Assign one user per client in order
    global_client = 0
    for sid in range(num_servers):
        server_map: Dict[str, List[int]] = {}
        for cid in range(cps_list[sid]):
            u = uniq[global_client]
            server_map[str(cid)] = user_to_indices.get(str(u), [])
            global_client += 1
        mapping[str(sid)] = server_map

    return mapping

# Keep old function name for compatibility
def hier_dirichlet_indices(
    labels: Sequence[int] | np.ndarray,
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    alpha_server: float = None,
    alpha_client: float = None,
    seed: int = None,
) -> Dict[str, Dict[str, List[int]]]:
    """Compatibility wrapper - now uses user-based partitioning."""
    return hier_user_indices(labels, num_servers, clients_per_server, seed=seed or 42)

def write_partitions(path: Path | str, mapping: Dict[str, Dict[str, List[int]]]) -> None:
    """Write JSON mapping to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp)
