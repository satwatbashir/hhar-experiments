import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Create HHAR/Plots folder if it doesn't exist
plots_dir = "d:/learn/HHAR/Plots"
os.makedirs(plots_dir, exist_ok=True)

# Define the metrics folder path (consolidated HHAR metrics)
metrics_dir = "d:/learn/HHAR/HHAR-metrics"

# Load data for each method
def load_and_process_loss_data():
    data = {}
    
    # 1. Scaffold - centralized test loss
    try:
        scaffold_df = pd.read_csv(os.path.join(metrics_dir, 'scaffold_centralized_metrics.csv'))
        data['Scaffold'] = scaffold_df[['round', 'central_test_loss']].copy()
    except Exception:
        data['Scaffold'] = pd.DataFrame(columns=['round', 'central_test_loss'])

    # 2. FedProx - centralized test loss
    try:
        fedprox_df = pd.read_csv(os.path.join(metrics_dir, 'fedprox_centralized_metrics.csv'))
        data['FedProx'] = fedprox_df[['round', 'central_test_loss']].copy()
    except Exception:
        data['FedProx'] = pd.DataFrame(columns=['round', 'central_test_loss'])

    # 3. pFedMe - centralized test loss
    try:
        pfedme_df = pd.read_csv(os.path.join(metrics_dir, 'pfedme_centralized_metrics.csv'))
        data['pFedMe'] = pfedme_df[['round', 'central_test_loss']].copy()
    except Exception:
        data['pFedMe'] = pd.DataFrame(columns=['round', 'central_test_loss'])

    # 4. HierFL - global test loss centralized (cloud-level)
    try:
        hierfl_df = pd.read_csv(os.path.join(metrics_dir, 'hierfl_global_model_metrics.csv'))
        tmp = hierfl_df[['global_round', 'global_test_loss_centralized']].copy()
        tmp.columns = ['round', 'central_test_loss']
        data['HierFL'] = tmp
    except Exception:
        data['HierFL'] = pd.DataFrame(columns=['round', 'central_test_loss'])

    # 5. CFL - aggregated client test loss per round
    try:
        cfl_rounds = pd.read_csv(os.path.join(metrics_dir, 'cfl_rounds_metrics.csv'))
        data['CFL'] = cfl_rounds[['round', 'test_loss_mean']].rename(columns={'test_loss_mean': 'central_test_loss'})
    except Exception:
        data['CFL'] = pd.DataFrame(columns=['round', 'central_test_loss'])

    # 6. Fedge-1000 - cluster mean loss computed from rounds (servers) file
    try:
        fedge_path = os.path.join(metrics_dir, 'fedge-1000_rounds.csv')
        used_fallback = False
        if not os.path.exists(fedge_path):
            # Fallback to original Fedge-100 location if consolidated copy is missing
            fedge_path = "d:/learn/HHAR/Fedge-100/fedge/metrics/servers/rounds.csv"
            used_fallback = True
        fedge_rounds = pd.read_csv(fedge_path)
        loss_col = 'server_partition_test_loss' if 'server_partition_test_loss' in fedge_rounds.columns else (
            'client_test_loss_mean' if 'client_test_loss_mean' in fedge_rounds.columns else None)
        if loss_col is not None and 'global_round' in fedge_rounds.columns:
            agg = (fedge_rounds
                   .groupby('global_round')[loss_col]
                   .mean()
                   .reset_index()
                   .rename(columns={'global_round': 'round', loss_col: 'central_test_loss'})
                   .sort_values('round')
                   .reset_index(drop=True))
            data['Fedge-v1'] = agg
            # Consolidated one-time diagnostic log to avoid duplicate prints
            if not getattr(load_and_process_loss_data, "_fedge_logged", False):
                src = 'fallback path' if used_fallback else 'HHAR-metrics copy'
                print(f"[Fedge-1000] {src}; rows={len(fedge_rounds)}; path={fedge_path}; loss_col={loss_col}")
                load_and_process_loss_data._fedge_logged = True
        else:
            print("[Fedge-1000] Missing required columns; skipping")
            data['Fedge-v1'] = pd.DataFrame(columns=['round', 'central_test_loss'])
    except Exception as e:
        print(f"[Fedge-1000] Failed to load: {e}")
        data['Fedge-v1'] = pd.DataFrame(columns=['round', 'central_test_loss'])

    return data

# No filtering needed - we'll plot all data points but only label x-axis at intervals
def prepare_data(data):
    # Just return the data as-is, no filtering
    return data

# Create the plot
def create_loss_plot():
    # Load and process data
    data = load_and_process_loss_data()
    
    # Use all data points (no filtering)
    plot_data = prepare_data(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and line styles for each method
    colors = {
        'Scaffold': '#1f77b4',
        'FedProx': '#ff7f0e', 
        'pFedMe': '#2ca02c',
        'HierFL': '#d62728',
        'CFL': '#9467bd',
        'Fedge-v1': '#8b4513'
    }
    
    line_styles = {
        'Scaffold': '-',
        'FedProx': '-',
        'pFedMe': '-',
        'HierFL': '-',
        'CFL': '-',
        'Fedge-v1': '-'
    }
    
    # Plot each method
    for method, df in plot_data.items():
        if not df.empty:
            plt.plot(df['round'], df['central_test_loss'], 
                    color=colors[method], 
                    linestyle=line_styles[method],
                    marker='o', 
                    markersize=2,
                    linewidth=2,
                    label=method)
    
    # Customize the plot
    plt.xlabel('Global Rounds', fontsize=14, fontweight='bold')
    plt.ylabel('Test Loss', fontsize=14, fontweight='bold')
    plt.title('Loss vs Rounds', fontsize=16, fontweight='bold')
    
    # Set axis limits and ticks dynamically based on data
    max_round = 0
    for df in plot_data.values():
        if not df.empty:
            max_round = max(max_round, int(df['round'].max()))
    if max_round == 0:
        max_round = 100
    plt.xlim(0, max_round)
    # Let y-axis auto-scale based on data
    step = max(1, max_round // 10)
    x_ticks = list(range(0, max_round + 1, step))
    if 1 not in x_ticks:
        x_ticks = sorted(set([1] + x_ticks))
    plt.xticks(x_ticks, rotation=45)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'loss_vs_rounds.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nFinal Loss Summary (at last available round):")
    print("-" * 50)
    for method, df in plot_data.items():
        if not df.empty:
            final_loss = df['central_test_loss'].iloc[-1]
            final_round = df['round'].iloc[-1]
            print(f"{method:12}: {final_loss:.4f} (Round {final_round})")

if __name__ == "__main__":
    create_loss_plot()
