#!/usr/bin/env python3
"""
Generate hierarchical FL plots for HierFL with cloud + 3 servers metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
import sys
import warnings
from dataclasses import dataclass


@dataclass
class HierFLMetrics:
    """Container for HierFL hierarchical metrics."""
    global_df: pd.DataFrame
    server_dfs: Dict[int, pd.DataFrame]


# --------------------------
# Data Loading
# --------------------------
def find_latest_run_dir(base_path: Path) -> Optional[Path]:
    """Find the latest run directory in fedge/runs/."""
    runs_dir = base_path / "fedge" / "runs"
    if not runs_dir.exists():
        return None
    
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Return the most recent run directory
    return max(run_dirs, key=lambda x: x.name)


def load_hierfl_metrics(base_path: Path) -> Optional[HierFLMetrics]:
    """Load HierFL metrics from the latest run directory and server directories."""
    # Find latest run directory
    latest_run = find_latest_run_dir(base_path)
    if not latest_run:
        warnings.warn("No run directory found in fedge/runs/")
        return None
    
    # Load global model metrics
    global_metrics_path = latest_run / "global_model_metrics.csv"
    if not global_metrics_path.exists():
        warnings.warn(f"Global model metrics not found: {global_metrics_path}")
        return None
    
    try:
        global_df = pd.read_csv(global_metrics_path)
    except Exception as e:
        warnings.warn(f"Failed to load global metrics: {e}")
        return None
    
    # Load server metrics from fedge/rounds/leaf/server_X/
    server_dfs = {}
    rounds_dir = base_path / "fedge" / "rounds" / "leaf"
    
    for server_id in range(3):  # 3 servers: 0, 1, 2
        server_dir = rounds_dir / f"server_{server_id}"
        server_metrics_path = server_dir / "server_metrics.csv"
        
        if server_metrics_path.exists():
            try:
                server_df = pd.read_csv(server_metrics_path)
                server_dfs[server_id] = server_df
            except Exception as e:
                warnings.warn(f"Failed to load server {server_id} metrics: {e}")
    
    if not server_dfs:
        warnings.warn("No server metrics found")
        return None
    
    return HierFLMetrics(global_df=global_df, server_dfs=server_dfs)


# --------------------------
# Plotting
# --------------------------
def ensure_output_dir(out_dir: Path) -> None:
    """Create the output directory and verify it exists."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {out_dir}") from e
    if not out_dir.exists() or not out_dir.is_dir():
        raise RuntimeError(f"Output directory could not be verified: {out_dir}")


def apply_smoothing(data: pd.Series, window: int) -> pd.Series:
    """Apply rolling average smoothing to data."""
    if window <= 1:
        return data
    return data.rolling(window=window, min_periods=1).mean()


def make_hierfl_plots(
    metrics: HierFLMetrics,
    out_dir: Path,
    title: Optional[str] = None,
    smooth: int = 1,
) -> Path:
    """Create hierarchical FL plots with cloud + 3 servers."""
    sns.set_theme(style="white", context="talk", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
    })

    ensure_output_dir(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    # Plot 1: Test Accuracy (Cloud + 3 Servers)
    ax1.set_title("Hierarchical Test Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Test Accuracy")
    
    # Cloud centralized accuracy
    if 'global_test_accuracy_centralized' in metrics.global_df.columns:
        rounds = metrics.global_df['global_round'] + 1
        cloud_acc = apply_smoothing(metrics.global_df['global_test_accuracy_centralized'], smooth)
        ax1.plot(rounds, cloud_acc, label='Cloud (Centralized)', linewidth=2, color='red', linestyle='-')
    
    # Server accuracies
    colors = ['blue', 'green', 'orange']
    for server_id, server_df in metrics.server_dfs.items():
        if 'server_test_accuracy_on_full_dataset' in server_df.columns:
            rounds = server_df['global_round'] + 1
            server_acc = apply_smoothing(server_df['server_test_accuracy_on_full_dataset'], smooth)
            ax1.plot(rounds, server_acc, label=f'Server {server_id}', linewidth=2, 
                    color=colors[server_id], linestyle='--')
    
    ax1.set_xlim(1, 100)
    ax1.set_xticks(range(1, 101, 9))
    ax1.set_xticks(list(range(1, 101, 9)) + [100], minor=False)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Loss (Cloud + 3 Servers)
    ax2.set_title("Hierarchical Test Loss")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Test Loss")
    
    # Cloud centralized loss
    if 'global_test_loss_centralized' in metrics.global_df.columns:
        rounds = metrics.global_df['global_round'] + 1
        cloud_loss = apply_smoothing(metrics.global_df['global_test_loss_centralized'], smooth)
        ax2.plot(rounds, cloud_loss, label='Cloud (Centralized)', linewidth=2, color='red', linestyle='-')
    
    # Server losses
    for server_id, server_df in metrics.server_dfs.items():
        if 'server_test_loss_on_full_dataset' in server_df.columns:
            rounds = server_df['global_round'] + 1
            server_loss = apply_smoothing(server_df['server_test_loss_on_full_dataset'], smooth)
            ax2.plot(rounds, server_loss, label=f'Server {server_id}', linewidth=2, 
                    color=colors[server_id], linestyle='--')
    
    ax2.set_xlim(1, 100)
    ax2.set_xticks(range(1, 101, 9))
    ax2.set_xticks(list(range(1, 101, 9)) + [100], minor=False)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    fname_base = "hierfl_overview"
    png_path = out_dir / f"{fname_base}.png"
    svg_path = out_dir / f"{fname_base}.svg"
    try:
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to save figures to: {out_dir}") from e
    plt.close(fig)
    return png_path


# --------------------------
# CLI
# --------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate HierFL hierarchical plots.")
    p.add_argument("--base-path", type=str, default="..", help="Base path to HierFL project.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save plots.")
    p.add_argument("--smooth", type=int, default=1, help="Rolling average window for smoothing.")
    p.add_argument("--title", type=str, default=None, help="Custom figure title.")
    p.add_argument("--verbose", action="store_true", help="Print verbose logs.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_path = Path(args.base_path).resolve()
    
    if not base_path.exists():
        print(f"[Error] Base path not found: {base_path}", file=sys.stderr)
        return 2

    # Determine output dir
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        script_dir = Path(__file__).parent.resolve()
        out_dir = script_dir

    try:
        ensure_output_dir(out_dir)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        return 3

    if args.verbose:
        print(f"[Info] Base path: {base_path}")
        print(f"[Info] Output directory: {out_dir}")

    # Load metrics
    metrics = load_hierfl_metrics(base_path)
    if not metrics:
        print("[Error] Failed to load HierFL metrics", file=sys.stderr)
        return 4

    if args.verbose:
        print(f"[Info] Loaded global metrics with {len(metrics.global_df)} rounds")
        print(f"[Info] Loaded {len(metrics.server_dfs)} server metrics")

    # Generate title
    title = args.title or "HierFL Hierarchical Overview"
    smooth = max(1, int(args.smooth))

    try:
        out_path = make_hierfl_plots(metrics, out_dir, title=title, smooth=smooth)
    except Exception as e:
        print(f"[Error] Plotting failed: {e}", file=sys.stderr)
        return 5

    print(f"Saved HierFL plots to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
