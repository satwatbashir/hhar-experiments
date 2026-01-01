#!/usr/bin/env python3
"""
Generate a 2x2 overview plot from federated pFedMe metrics CSVs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
import sys
import warnings
from dataclasses import dataclass


@dataclass
class CentralMetrics:
    """Container for centralized evaluation metrics."""
    df: pd.DataFrame


@dataclass
class ClientMetrics:
    """Container for client metrics."""
    df: pd.DataFrame
    per_client: Dict[str, pd.DataFrame]


# --------------------------
# Data Loading
# --------------------------
def load_central_metrics(metrics_dir: Path) -> Optional[CentralMetrics]:
    """Load centralized metrics from CSV file."""
    central_path = metrics_dir / "centralized_metrics.csv"
    if not central_path.exists():
        return None
    
    try:
        df = pd.read_csv(central_path)
        return CentralMetrics(df=df)
    except Exception as e:
        warnings.warn(f"Failed to load centralized metrics: {e}")
        return None


def load_client_metrics(metrics_dir: Path) -> ClientMetrics:
    """Load client metrics from CSV files."""
    clients_path = metrics_dir / "clients.csv"
    rounds_path = metrics_dir / "rounds.csv"
    
    per_client = {}
    df = pd.DataFrame()
    
    # Try to load clients.csv (per-client data)
    if clients_path.exists():
        try:
            df = pd.read_csv(clients_path)
            # Group by client ID
            for cid, group in df.groupby('cid'):
                per_client[str(cid)] = group.copy()
        except Exception as e:
            warnings.warn(f"Failed to load client metrics: {e}")
    
    # Try to load rounds.csv as fallback
    if rounds_path.exists() and df.empty:
        try:
            df = pd.read_csv(rounds_path)
        except Exception as e:
            warnings.warn(f"Failed to load rounds metrics: {e}")
    
    return ClientMetrics(df=df, per_client=per_client)


# --------------------------
# Plotting
# --------------------------
def ensure_output_dir(out_dir: Path) -> None:
    """
    Create the output directory and verify it exists. Raises if creation fails.
    """
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


def make_overview_plot(
    central: Optional[CentralMetrics],
    clients: ClientMetrics,
    out_dir: Path,
    title: Optional[str] = None,
    smooth: int = 1,
    style: str = "darkgrid",
) -> Path:
    """Create a 2x2 overview plot of pFedMe metrics."""
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

    # Ensure output directory exists before creating the figure/files
    ensure_output_dir(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    # Plot 1: Centralized Test Accuracy over Rounds
    ax1.set_title("Centralized Test Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Test Accuracy")
    
    if central and 'central_test_accuracy' in central.df.columns:
        rounds = central.df['round'] + 1  # Start from 1 instead of 0
        test_acc = apply_smoothing(central.df['central_test_accuracy'], smooth)
        ax1.plot(rounds, test_acc, label='Centralized Test Accuracy', linewidth=2, color='blue')
    
    ax1.set_xlim(1, 100)
    ax1.set_xticks(range(1, 101, 9))  # 1, 11, 21, ..., 141
    ax1.set_xticks(list(range(1, 101, 9)) + [100], minor=False)  # Include 150
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Centralized Test Loss
    ax2.set_title("Centralized Test Loss")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Test Loss")
    
    if central and 'central_test_loss' in central.df.columns:
        rounds = central.df['round'] + 1  # Start from 1 instead of 0
        test_loss = apply_smoothing(central.df['central_test_loss'], smooth)
        ax2.plot(rounds, test_loss, label='Centralized Test Loss', linewidth=2, color='purple')
    
    ax2.set_xlim(1, 100)
    ax2.set_xticks(range(1, 101, 9))  # 1, 11, 21, ..., 141
    ax2.set_xticks(list(range(1, 101, 9)) + [100], minor=False)  # Include 150
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average Client Test Accuracy per Round
    ax3.set_title("Average Client Test Accuracy")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Test Accuracy")
    
    if not clients.df.empty and 'test_accuracy' in clients.df.columns:
        # Compute average test accuracy per round across all clients
        client_avg_test_acc = clients.df.groupby('round')['test_accuracy'].mean()
        rounds = client_avg_test_acc.index + 1  # Start from 1 instead of 0
        avg_test_acc = apply_smoothing(client_avg_test_acc.values, smooth)
        ax3.plot(rounds, avg_test_acc, label='Avg Client Test Accuracy', linewidth=2, color='green')
    
    ax3.set_xlim(1, 100)
    ax3.set_xticks(range(1, 101, 9))  # 1, 11, 21, ..., 141
    ax3.set_xticks(list(range(1, 101, 9)) + [100], minor=False)  # Include 150
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Average Client Test Loss per Round
    ax4.set_title("Average Client Test Loss")
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Test Loss")
    
    if not clients.df.empty and 'test_loss' in clients.df.columns:
        # Compute average test loss per round across all clients
        client_avg_test_loss = clients.df.groupby('round')['test_loss'].mean()
        rounds = client_avg_test_loss.index + 1  # Start from 1 instead of 0
        avg_test_loss = apply_smoothing(client_avg_test_loss.values, smooth)
        ax4.plot(rounds, avg_test_loss, label='Avg Client Test Loss', linewidth=2, color='orange')
    
    ax4.set_xlim(1, 100)
    ax4.set_xticks(range(1, 101, 9))  # 1, 11, 21, ..., 141
    ax4.set_xticks(list(range(1, 101, 9)) + [100], minor=False)  # Include 150
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Tight layout and save
    fig.tight_layout()

    fname_base = "pfedme_overview"
    png_path = out_dir / f"{fname_base}.png"
    svg_path = out_dir / f"{fname_base}.svg"
    try:
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to save figures to: {out_dir}. Ensure the path is writable.") from e
    plt.close(fig)
    return png_path


# --------------------------
# CLI
# --------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate pFedMe overview plots from metrics CSVs.")
    p.add_argument("--metrics-dir", type=str, default="../metrics", help="Path to the metrics folder containing CSV files.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save plots. Defaults to same directory as script.")
    p.add_argument("--smooth", type=int, default=1, help="Rolling average window for smoothing curves (>=1, 1 disables smoothing).")
    p.add_argument("--style", type=str, default="darkgrid", choices=["whitegrid", "darkgrid", "white", "ticks"], help="Seaborn style.")
    p.add_argument("--title", type=str, default=None, help="Custom figure title.")
    p.add_argument("--verbose", action="store_true", help="Print verbose progress logs.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    metrics_dir = Path(args.metrics_dir).resolve()
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        print(f"[Error] Metrics directory not found: {metrics_dir}", file=sys.stderr)
        return 2

    # Determine output dir - default to same directory as script
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        script_dir = Path(__file__).parent.resolve()
        out_dir = script_dir

    # Ensure output directory exists early and report it
    try:
        ensure_output_dir(out_dir)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        return 3

    if args.verbose:
        print(f"[Info] Metrics directory: {metrics_dir}")
        print(f"[Info] Output directory:  {out_dir}")

    # Load data
    central = load_central_metrics(metrics_dir)
    clients = load_client_metrics(metrics_dir)

    if central is None:
        warnings.warn("No central/global metrics CSV detected. Will attempt to use client data for fallbacks.")
    if not clients.per_client:
        warnings.warn("No client metrics detected. Some subplots may be empty.")
    else:
        if args.verbose:
            print(f"[Info] Loaded client metrics for {len(clients.per_client)} clients.")

    # Derive a nice title from metrics dir if not provided
    title = args.title
    if not title:
        exp_name = metrics_dir.name
        title = f"pFedMe Overview — {exp_name}"

    # Clamp smooth
    smooth = max(1, int(args.smooth))

    try:
        out_path = make_overview_plot(central, clients, out_dir, title=title, smooth=smooth, style=args.style)
    except Exception as e:
        print(f"[Error] Plotting failed: {e}", file=sys.stderr)
        return 4

    print(f"Saved overview plot to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
