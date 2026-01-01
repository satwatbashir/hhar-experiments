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
def load_and_process_data():
    data = {}
    
    # 1. Scaffold - centralized test accuracy
    try:
        scaffold_df = pd.read_csv(os.path.join(metrics_dir, 'scaffold_centralized_metrics.csv'))
        data['Scaffold'] = scaffold_df[['round', 'central_test_accuracy']].copy()
    except Exception:
        data['Scaffold'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])

    # 2. FedProx - centralized test accuracy
    try:
        fedprox_df = pd.read_csv(os.path.join(metrics_dir, 'fedprox_centralized_metrics.csv'))
        data['FedProx'] = fedprox_df[['round', 'central_test_accuracy']].copy()
    except Exception:
        data['FedProx'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])

    # 3. pFedMe - centralized test accuracy
    try:
        pfedme_df = pd.read_csv(os.path.join(metrics_dir, 'pfedme_centralized_metrics.csv'))
        data['pFedMe'] = pfedme_df[['round', 'central_test_accuracy']].copy()
    except Exception:
        data['pFedMe'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])

    # 4. HierFL - global test accuracy centralized (cloud-level)
    try:
        hierfl_df = pd.read_csv(os.path.join(metrics_dir, 'hierfl_global_model_metrics.csv'))
        tmp = hierfl_df[['global_round', 'global_test_accuracy_centralized']].copy()
        tmp.columns = ['round', 'central_test_accuracy']
        data['HierFL'] = tmp
    except Exception:
        data['HierFL'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])

    # 5. CFL - aggregated client test accuracy per round
    try:
        cfl_rounds = pd.read_csv(os.path.join(metrics_dir, 'cfl_rounds_metrics.csv'))
        data['CFL'] = cfl_rounds[['round', 'test_acc_mean']].rename(columns={'test_acc_mean': 'central_test_accuracy'})
    except Exception:
        data['CFL'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])

    # 6. Fedge-1000 - cluster mean accuracy computed from rounds (servers) file
    try:
        fedge_path = os.path.join(metrics_dir, 'fedge-1000_rounds.csv')
        used_fallback = False
        if not os.path.exists(fedge_path):
            # Fallback to original Fedge-100 location if consolidated copy is missing
            fedge_path = "d:/learn/HHAR/Fedge-100/fedge/metrics/servers/rounds.csv"
            used_fallback = True
        fedge_rounds = pd.read_csv(fedge_path)
        acc_col = 'server_partition_test_accuracy' if 'server_partition_test_accuracy' in fedge_rounds.columns else (
            'client_test_accuracy_mean' if 'client_test_accuracy_mean' in fedge_rounds.columns else None)
        if acc_col is not None and 'global_round' in fedge_rounds.columns:
            agg = (fedge_rounds
                   .groupby('global_round')[acc_col]
                   .mean()
                   .reset_index()
                   .rename(columns={'global_round': 'round', acc_col: 'central_test_accuracy'})
                   .sort_values('round')
                   .reset_index(drop=True))
            data['Fedge-v1'] = agg
            src = 'fallback path' if used_fallback else 'HHAR-metrics copy'
            print(f"[Fedge-1000] Loaded {len(fedge_rounds)} rows from {src}: {fedge_path}")
            print(f"[Fedge-1000] Using accuracy column: {acc_col}")
        else:
            print("[Fedge-1000] Missing required columns; skipping")
            data['Fedge-v1'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])
    except Exception as e:
        print(f"[Fedge-1000] Failed to load: {e}")
        data['Fedge-v1'] = pd.DataFrame(columns=['round', 'central_test_accuracy'])

    return data

# No filtering needed - we'll plot all data points but only label x-axis at intervals
def prepare_data(data):
    # Just return the data as-is, no filtering
    return data

# Create the plot
def create_accuracy_plot():
    # Load and process data
    data = load_and_process_data()
    
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
        'Fedge-v1': '#8b4513',
    }
    
    line_styles = {
        'Scaffold': '-',
        'FedProx': '-',
        'pFedMe': '-',
        'HierFL': '-',
        'CFL': '-',
        'Fedge-v1': '-',
    }
    
    # Plot each method
    for method, df in plot_data.items():
        if not df.empty:
            plt.plot(df['round'], df['central_test_accuracy'], 
                    color=colors[method], 
                    linestyle=line_styles[method],
                    marker='o', 
                    markersize=2,
                    linewidth=2,
                    label=method)
    
    # Customize the plot
    plt.xlabel('Global Rounds', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title('Accuracy vs Rounds', fontsize=16, fontweight='bold')
    
    # Set axis limits and ticks dynamically based on data
    max_round = 0
    for df in plot_data.values():
        if not df.empty:
            max_round = max(max_round, int(df['round'].max()))
    if max_round == 0:
        max_round = 100
    plt.xlim(0, max_round)
    plt.ylim(0, 1.0)

    # Set x-axis ticks to show rounds at intervals of ~10
    step = max(1, max_round // 10)
    x_ticks = list(range(0, max_round + 1, step))
    if 1 not in x_ticks:
        x_ticks = sorted(set([1] + x_ticks))
    plt.xticks(x_ticks, rotation=45)
    
    # Set y-axis ticks
    plt.yticks(np.arange(0, 1.1, 0.2))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'accuracy_vs_rounds.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nFinal Accuracy Summary (at last available round):")
    print("-" * 50)
    for method, df in plot_data.items():
        if not df.empty:
            final_acc = df['central_test_accuracy'].iloc[-1]
            final_round = df['round'].iloc[-1]
            print(f"{method:12}: {final_acc:.4f} (Round {final_round})")

if __name__ == "__main__":
    create_accuracy_plot()

