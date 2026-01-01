import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
from pathlib import Path

def create_plots_folder():
    """Create or recreate the plots folder"""
    plots_dir = Path("plots")
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir()
    return plots_dir

def load_metrics_data():
    """Load all metrics data from CSV files"""
    # Find all leaf server directories
    leaf_servers = []
    rounds_dir = Path("rounds/leaf")
    
    if not rounds_dir.exists():
        raise FileNotFoundError("rounds/leaf directory not found")
    
    for server_dir in rounds_dir.iterdir():
        if server_dir.is_dir() and server_dir.name.startswith("server_"):
            server_id = int(server_dir.name.split("_")[1])
            leaf_servers.append(server_id)
    
    leaf_servers.sort()
    
    # Load data for each server
    server_data = {}
    for server_id in leaf_servers:
        server_dir = rounds_dir / f"server_{server_id}"
        
        # Load server metrics (centralized test results)
        server_metrics_file = server_dir / "server_metrics.csv"
        client_eval_file = server_dir / "client_eval_metrics.csv"
        
        if server_metrics_file.exists() and client_eval_file.exists():
            server_metrics = pd.read_csv(server_metrics_file)
            client_eval = pd.read_csv(client_eval_file)
            
            server_data[server_id] = {
                'server_metrics': server_metrics,
                'client_eval': client_eval
            }
    
    # Load global metrics
    global_metrics = None
    runs_dir = Path("runs")
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                global_metrics_file = run_dir / "global_model_metrics.csv"
                if global_metrics_file.exists():
                    global_metrics = pd.read_csv(global_metrics_file)
                    break
    
    return server_data, leaf_servers, global_metrics

def calculate_client_averages(client_eval_df):
    """Calculate average client test accuracy and loss per round"""
    # Group by global_round and calculate averages
    avg_by_round = client_eval_df.groupby('global_round').agg({
        'client_test_accuracy': 'mean',
        'client_test_loss': 'mean'
    }).reset_index()
    
    return avg_by_round

def create_server_plots(server_data, leaf_servers, plots_dir):
    """Create plots for each leaf server"""
    
    for server_id in leaf_servers:
        if server_id not in server_data:
            continue
            
        server_metrics = server_data[server_id]['server_metrics']
        client_eval = server_data[server_id]['client_eval']
        
        # Calculate client averages
        client_averages = calculate_client_averages(client_eval)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'HierFL Server {server_id} - HHAR overview', fontsize=16, fontweight='bold')
        
        # Ensure we have data for all 150 rounds
        rounds = np.arange(1, 151)
        
        # Plot 1: Server Centralized Test Accuracy
        ax1 = axes[0, 0]
        server_acc = server_metrics['server_test_accuracy_on_full_dataset'].values[:150]
        ax1.plot(rounds[:len(server_acc)], server_acc, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_title('Server Centralized Test Accuracy', fontweight='bold')
        ax1.set_xlabel('Global Rounds')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlim(1, 150)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(np.arange(10, 151, 10))
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Server Centralized Test Loss
        ax2 = axes[0, 1]
        server_loss = server_metrics['server_test_loss_on_full_dataset'].values[:150]
        ax2.plot(rounds[:len(server_loss)], server_loss, 'r-', linewidth=2, marker='o', markersize=3)
        ax2.set_title('Server Centralized Test Loss', fontweight='bold')
        ax2.set_xlabel('Global Rounds')
        ax2.set_ylabel('Loss')
        ax2.set_xlim(1, 150)
        ax2.set_xticks(np.arange(10, 151, 10))
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Client Test Accuracy
        ax3 = axes[1, 0]
        if not client_averages.empty:
            client_acc = client_averages['client_test_accuracy'].values[:150]
            client_rounds = client_averages['global_round'].values[:150] + 1  # Convert to 1-based
            ax3.plot(client_rounds, client_acc, 'g-', linewidth=2, marker='o', markersize=3)
        ax3.set_title('Average Client Test Accuracy', fontweight='bold')
        ax3.set_xlabel('Global Rounds')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlim(1, 150)
        ax3.set_ylim(0, 1)
        ax3.set_xticks(np.arange(10, 151, 10))
        ax3.set_yticks(np.arange(0, 1.1, 0.2))
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Average Client Test Loss
        ax4 = axes[1, 1]
        if not client_averages.empty:
            client_loss = client_averages['client_test_loss'].values[:150]
            ax4.plot(client_rounds, client_loss, 'm-', linewidth=2, marker='o', markersize=3)
        ax4.set_title('Average Client Test Loss', fontweight='bold')
        ax4.set_xlabel('Global Rounds')
        ax4.set_ylabel('Loss')
        ax4.set_xlim(1, 150)
        ax4.set_xticks(np.arange(10, 151, 10))
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        plot_filename = plots_dir / f'leaf_server_{server_id}_metrics.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created plot for Leaf Server {server_id}: {plot_filename}")

def create_global_plot(global_metrics, plots_dir):
    """Create global metrics plot with 4 subplots"""
    if global_metrics is None:
        print("No global metrics data found, skipping global plot")
        return
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HierFL - HHAR Cloud Metrics overview', fontsize=16, fontweight='bold')
    
    # Ensure we have data for all 150 rounds
    rounds = np.arange(1, 151)
    
    # Plot 1: Global Centralized Test Accuracy
    ax1 = axes[0, 0]
    global_acc = global_metrics['global_test_accuracy_centralized'].values[:150]
    global_rounds = global_metrics['global_round'].values[:150] + 1  # Convert to 1-based
    ax1.plot(global_rounds, global_acc, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_title('Global Centralized Test Accuracy', fontweight='bold')
    ax1.set_xlabel('Global Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim(1, 150)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(np.arange(10, 151, 10))
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Global Centralized Test Loss
    ax2 = axes[0, 1]
    global_loss = global_metrics['global_test_loss_centralized'].values[:150]
    ax2.plot(global_rounds, global_loss, 'r-', linewidth=2, marker='o', markersize=3)
    ax2.set_title('Global Centralized Test Loss', fontweight='bold')
    ax2.set_xlabel('Global Rounds')
    ax2.set_ylabel('Loss')
    ax2.set_xlim(1, 150)
    ax2.set_xticks(np.arange(10, 151, 10))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average of Three Servers Centralized Test Accuracy
    ax3 = axes[1, 0]
    avg_server_acc = global_metrics['avg_test_accuracy_across_servers'].values[:150]
    ax3.plot(global_rounds, avg_server_acc, 'g-', linewidth=2, marker='o', markersize=3)
    ax3.set_title('Average Server Centralized Test Accuracy', fontweight='bold')
    ax3.set_xlabel('Global Rounds')
    ax3.set_ylabel('Accuracy')
    ax3.set_xlim(1, 150)
    ax3.set_ylim(0, 1)
    ax3.set_xticks(np.arange(10, 151, 10))
    ax3.set_yticks(np.arange(0, 1.1, 0.2))
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average of Three Servers Centralized Test Loss
    ax4 = axes[1, 1]
    avg_server_loss = global_metrics['avg_test_loss_across_servers'].values[:150]
    ax4.plot(global_rounds, avg_server_loss, 'm-', linewidth=2, marker='o', markersize=3)
    ax4.set_title('Average Server Centralized Test Loss', fontweight='bold')
    ax4.set_xlabel('Global Rounds')
    ax4.set_ylabel('Loss')
    ax4.set_xlim(1, 150)
    ax4.set_xticks(np.arange(10, 151, 10))
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plot_filename = plots_dir / 'global_model_metrics.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created global metrics plot: {plot_filename}")

def main():
    """Main function to generate all plots"""
    try:
        # Create plots directory
        plots_dir = create_plots_folder()
        print(f"Created plots directory: {plots_dir}")
        
        # Load metrics data
        print("Loading metrics data...")
        server_data, leaf_servers, global_metrics = load_metrics_data()
        print(f"Found {len(leaf_servers)} leaf servers: {leaf_servers}")
        
        # Create plots for each server
        print("Generating leaf server plots...")
        create_server_plots(server_data, leaf_servers, plots_dir)
        
        # Create global metrics plot
        print("Generating global metrics plot...")
        create_global_plot(global_metrics, plots_dir)
        
        print(f"\nAll plots generated successfully in '{plots_dir}' directory!")
        print(f"Generated {len(leaf_servers)} leaf server plots + 1 global metrics plot.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()