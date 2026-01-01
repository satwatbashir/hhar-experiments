"""Hierarchical Dirichlet partition utilities.

This module produces a two-level Dirichlet split compatible with the
HierFL paper/codebase:
    1.  Outer Dirichlet across *servers* (alpha_server)
    2.  Inner Dirichlet across *clients within each server* (alpha_client)

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
from flwr_datasets.partitioner import DirichletPartitioner
import datasets as hf_datasets  # type: ignore – runtime dependency already required elsewhere

__all__ = [
    "hier_dirichlet_indices",
    "write_partitions",
]

def _build_dirichlet_partitioner(num_parts: int, alpha: float, seed: int) -> DirichletPartitioner:  # noqa: D401,E501
    """Return a *fresh* HuggingFace DirichletPartitioner instance."""
    return DirichletPartitioner(
        num_partitions=num_parts,
        partition_by="label",
        alpha=alpha,
        min_partition_size=10,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )

def hier_dirichlet_indices(
    labels_ds: "hf_datasets.Dataset",
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    alpha_server: float = 0.5,
    alpha_client: float = 0.3,
    seed: int = 42,
) -> Dict[str, Dict[str, List[int]]]:
    """Return a hierarchical mapping of indices for *all* clients.

    Parameters
    ----------
    labels_ds : hf_datasets.Dataset
        A HF dataset **aligned with the training set order** and with a
        scalar int64 column named "label".
    num_servers : int
        Number of leaf servers.
    clients_per_server : int | Sequence[int]
        Number of clients *inside each* leaf server.
    alpha_server/alpha_client : float
        Dirichlet concentration parameters for outer / inner split.
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

    # ── Outer split: across servers ────────────────────────────────────────
    outer = _build_dirichlet_partitioner(num_servers, alpha_server, seed)
    outer.dataset = labels_ds
    # Private API call – Flower team exposes no public method yet
    outer._partition_id_to_indices = {}
    outer._partition_id_to_indices_determined = False
    outer._determine_partition_id_to_indices_if_needed()

    mapping: Dict[str, Dict[str, List[int]]] = {}

    # ── Inner split: per-server across that server's clients ───────────────
    for sid in range(num_servers):
        server_indices = outer._partition_id_to_indices[sid]
        # Select subset dataset *in the same order* so index translation is trivial
        sub_ds = labels_ds.select(server_indices)

        inner = _build_dirichlet_partitioner(cps_list[sid], alpha_client, seed + sid)
        inner.dataset = sub_ds
        inner._partition_id_to_indices = {}
        inner._partition_id_to_indices_determined = False
        inner._determine_partition_id_to_indices_if_needed()

        client_map: Dict[str, List[int]] = {}
        for cid in range(cps_list[sid]):
            # Indices returned by `inner` are *relative to sub_ds* (0..len-1).
            rel_idx = inner._partition_id_to_indices[cid]
            # Translate back to *global* indices of the original train set.
            global_idx = [int(server_indices[i]) for i in rel_idx]
            client_map[str(cid)] = global_idx
        mapping[str(sid)] = client_map

    return mapping

def write_partitions(path: Path | str, mapping: Dict[str, Dict[str, List[int]]]) -> None:
    """Write JSON mapping to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp)
