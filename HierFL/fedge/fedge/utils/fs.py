from pathlib import Path

def get_signals_dir(project_root: Path) -> Path:
    """Get or create the global signals directory (for cloud start/complete signals)."""
    signals_dir = project_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    return signals_dir

def get_global_rounds_dir(project_root: Path) -> Path:
    """Get or create the base directory for global rounds."""
    global_dir = project_root / "rounds" / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    return global_dir

def get_leaf_rounds_dir(project_root: Path) -> Path:
    """Get or create the base directory for leaf rounds."""
    leaf_dir = project_root / "rounds" / "leaf"
    leaf_dir.mkdir(parents=True, exist_ok=True)
    return leaf_dir

def get_global_round_dir(project_root: Path, round_num: int) -> Path:
    """
    Get or create the directory for a specific global round.
    Uses zero-based indexing for directory names (round_0, round_1, etc.)
    """
    # Use 0-indexed directory names for consistency
    round_dir = get_global_rounds_dir(project_root) / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir

def get_leaf_round_dir(project_root: Path, round_num: int) -> Path:
    """
    Get or create the directory for a specific leaf round.
    Uses zero-based indexing for directory names (round_0, round_1, etc.)
    """
    # Use 0-indexed directory names for consistency
    round_dir = get_leaf_rounds_dir(project_root) / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir

# ──────────────────────────────────────────────────────────────────────────────
#  NEW flat-directory helpers (2025-06 refactor)
# ──────────────────────────────────────────────────────────────────────────────

def leaf_server_dir(project_root: Path, server_id: int) -> Path:
    """Return <root>/rounds/leaf/server_<sid> and ensure it exists."""
    d = project_root / "rounds" / "leaf" / f"server_{server_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def global_dir(project_root: Path) -> Path:
    """Return <root>/rounds/global and ensure it exists."""
    d = project_root / "rounds" / "global"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ──────────────────────────────────────────────────────────────────────────────
#  Legacy helpers (kept for backward compatibility; do NOT use in new code)
# ──────────────────────────────────────────────────────────────────────────────

def get_server_dir(project_root: Path, round_num: int, server_id: int) -> Path:
    """DEPRECATED: round_num is ignored. Use leaf_server_dir instead."""
    # Ignore round_num and forward to new flat helper
    return leaf_server_dir(project_root, server_id)

def get_proxy_dir(project_root: Path, round_num: int, proxy_id: int) -> Path:
    """DEPRECATED: Use flat helpers instead."""
    """Get or create the proxy directory for a given round and proxy ID."""
    proxy_dir = get_leaf_round_dir(project_root, round_num) / "proxies" / f"proxy_{proxy_id}"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    return proxy_dir

def get_logs_dir(project_root: Path, round_num: int) -> Path:
    """Get or create the logs directory for a given round."""
    logs_dir = project_root / "logs" / f"round_{round_num}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def get_global_round_dir(project_root: Path, round_num: int) -> Path:
    """DEPRECATED: Use flat helpers instead."""
    """
    Get or create the directory for a specific global round.
    Uses zero-based indexing for directory names (round_0, round_1, etc.)
    """
    # Use 0-indexed directory names for consistency
    round_dir = get_global_rounds_dir(project_root) / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir

def get_leaf_round_dir(project_root: Path, round_num: int) -> Path:
    """DEPRECATED: Use flat helpers instead."""
    """
    Get or create the directory for a specific leaf round.
    Uses zero-based indexing for directory names (round_0, round_1, etc.)
    """
    # Use 0-indexed directory names for consistency
    round_dir = get_leaf_rounds_dir(project_root) / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir
