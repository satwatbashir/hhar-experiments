import pandas as pd
import numpy as np
import os

# Define the metrics folder path
metrics_dir = "d:/learn/HHAR/HHAR-metrics"

def extract_final_metrics():
    """Extract comprehensive metrics for all methods"""
    results = []
    
    # 1. Scaffold - compute stats from clients.csv and central from centralized_metrics
    try:
        scaffold_central = pd.read_csv(os.path.join(metrics_dir, 'scaffold_centralized_metrics.csv'))
        scaffold_clients = pd.read_csv(os.path.join(metrics_dir, 'scaffold_clients.csv'))

        final_round = int(scaffold_central['round'].max())
        final_central = scaffold_central[scaffold_central['round'] == final_round]
        final_clients = scaffold_clients[scaffold_clients['round'] == final_round]

        final_acc = float(final_central['central_test_accuracy'].iloc[0]) * 100
        final_loss = float(final_central['central_test_loss'].iloc[0])
        gen_gap = float(final_central['central_accuracy_gap'].iloc[0]) * 100 if 'central_accuracy_gap' in final_central.columns else np.nan

        if not final_clients.empty and 'test_accuracy' in final_clients.columns:
            client_acc = final_clients['test_accuracy'].astype(float) * 100
            std_dev = float(client_acc.std(ddof=1)) if len(client_acc) > 1 else 0.0
            n_clients = len(client_acc)
            margin_error = 1.96 * (std_dev / np.sqrt(n_clients)) if n_clients > 0 else 0.0
            ci_95 = f"[{final_acc - margin_error:.2f}, {final_acc + margin_error:.2f}]"
        else:
            std_dev = None
            ci_95 = "--"

        results.append({
            'Method': 'Scaffold',
            'Final_Accuracy_%': f"{final_acc:.2f}",
            'Std_Dev_%': f"{std_dev:.2f}" if std_dev is not None else "--",
            'CI_95%': ci_95,
            'Generalization_Gap_%': f"{gen_gap:.2f}" if not np.isnan(gen_gap) else "--",
            'Final_Loss': f"{final_loss:.4f}"
        })
    except Exception as e:
        print(f"Error processing Scaffold: {e}")
    
    # 2. FedProx
    try:
        fedprox_central = pd.read_csv(os.path.join(metrics_dir, 'fedprox_centralized_metrics.csv'))
        fedprox_clients = pd.read_csv(os.path.join(metrics_dir, 'fedprox_clients.csv'))

        final_round = int(fedprox_central['round'].max())
        final_central = fedprox_central[fedprox_central['round'] == final_round]
        final_clients = fedprox_clients[fedprox_clients['round'] == final_round] if 'round' in fedprox_clients.columns else pd.DataFrame()

        final_acc = float(final_central['central_test_accuracy'].iloc[0]) * 100
        final_loss = float(final_central['central_test_loss'].iloc[0])
        gen_gap = float(final_central['central_accuracy_gap'].iloc[0]) * 100 if 'central_accuracy_gap' in final_central.columns else np.nan

        if not final_clients.empty and 'test_accuracy' in final_clients.columns:
            client_acc = final_clients['test_accuracy'].astype(float) * 100
            std_dev = float(client_acc.std(ddof=1)) if len(client_acc) > 1 else 0.0
            n_clients = len(client_acc)
            margin_error = 1.96 * (std_dev / np.sqrt(n_clients)) if n_clients > 0 else 0.0
            ci_95 = f"[{final_acc - margin_error:.2f}, {final_acc + margin_error:.2f}]"
        else:
            std_dev = None
            ci_95 = "--"

        results.append({
            'Method': 'FedProx',
            'Final_Accuracy_%': f"{final_acc:.2f}",
            'Std_Dev_%': f"{std_dev:.2f}" if std_dev is not None else "--",
            'CI_95%': ci_95,
            'Generalization_Gap_%': f"{gen_gap:.2f}" if not np.isnan(gen_gap) else "--",
            'Final_Loss': f"{final_loss:.4f}"
        })
    except Exception as e:
        print(f"Error processing FedProx: {e}")
    
    # 3. pFedMe
    try:
        pfedme_central = pd.read_csv(os.path.join(metrics_dir, 'pfedme_centralized_metrics.csv'))
        pfedme_clients = pd.read_csv(os.path.join(metrics_dir, 'pfedme_clients.csv'))

        final_round = int(pfedme_central['round'].max())
        final_central = pfedme_central[pfedme_central['round'] == final_round]
        final_clients = pfedme_clients[pfedme_clients['round'] == final_round] if 'round' in pfedme_clients.columns else pd.DataFrame()

        final_acc = float(final_central['central_test_accuracy'].iloc[0]) * 100
        final_loss = float(final_central['central_test_loss'].iloc[0])
        gen_gap = float(final_central['central_accuracy_gap'].iloc[0]) * 100 if 'central_accuracy_gap' in final_central.columns else np.nan

        if not final_clients.empty and 'test_accuracy' in final_clients.columns:
            client_acc = final_clients['test_accuracy'].astype(float) * 100
            std_dev = float(client_acc.std(ddof=1)) if len(client_acc) > 1 else 0.0
            n_clients = len(client_acc)
            margin_error = 1.96 * (std_dev / np.sqrt(n_clients)) if n_clients > 0 else 0.0
            ci_95 = f"[{final_acc - margin_error:.2f}, {final_acc + margin_error:.2f}]"
        else:
            std_dev = None
            ci_95 = "--"

        results.append({
            'Method': 'pFedMe',
            'Final_Accuracy_%': f"{final_acc:.2f}",
            'Std_Dev_%': f"{std_dev:.2f}" if std_dev is not None else "--",
            'CI_95%': ci_95,
            'Generalization_Gap_%': f"{gen_gap:.2f}" if not np.isnan(gen_gap) else "--",
            'Final_Loss': f"{final_loss:.4f}"
        })
    except Exception as e:
        print(f"Error processing pFedMe: {e}")
    
    # 4. HierFL - use aggregated HHAR-metrics files
    try:
        hierfl_global = pd.read_csv(os.path.join(metrics_dir, 'hierfl_global_model_metrics.csv'))
        hierfl_clients = pd.read_csv(os.path.join(metrics_dir, 'hierfl_client_eval_metrics.csv'))

        final_round = int(hierfl_global['global_round'].max())
        final_global = hierfl_global[hierfl_global['global_round'] == final_round]

        final_acc = float(final_global['global_test_accuracy_centralized'].iloc[0]) * 100
        final_loss = float(final_global['global_test_loss_centralized'].iloc[0])

        # std dev and CI from client eval accuracies at final round
        final_clients = hierfl_clients[hierfl_clients['global_round'] == final_round] if 'global_round' in hierfl_clients.columns else pd.DataFrame()
        if not final_clients.empty and 'client_test_accuracy' in final_clients.columns:
            client_acc_array = final_clients['client_test_accuracy'].astype(float).to_numpy() * 100
            std_dev = float(np.std(client_acc_array, ddof=1)) if len(client_acc_array) > 1 else 0.0
            n_clients = len(client_acc_array)
            margin_error = 1.96 * (std_dev / np.sqrt(n_clients)) if n_clients > 0 else 0.0
            ci_95 = f"[{final_acc - margin_error:.2f}, {final_acc + margin_error:.2f}]"
        else:
            std_dev = None
            ci_95 = "--"

        avg_server_acc = float(final_global['avg_test_accuracy_across_servers'].iloc[0]) * 100 if 'avg_test_accuracy_across_servers' in final_global.columns else np.nan
        gen_gap = final_acc - avg_server_acc if not np.isnan(avg_server_acc) else np.nan

        results.append({
            'Method': 'HierFL',
            'Final_Accuracy_%': f"{final_acc:.2f}",
            'Std_Dev_%': f"{std_dev:.2f}" if std_dev is not None else "--",
            'CI_95%': ci_95,
            'Generalization_Gap_%': f"{gen_gap:.2f}" if not np.isnan(gen_gap) else "--",
            'Final_Loss': f"{final_loss:.4f}"
        })
    except Exception as e:
        print(f"Error processing HierFL: {e}")
    
    # 5. CFL - Clustered FL
    try:
        cfl_clients = pd.read_csv(os.path.join(metrics_dir, 'cfl_clients_metrics.csv'))

        final_round = int(cfl_clients['round'].max())
        final_clients = cfl_clients[cfl_clients['round'] == final_round]

        # Use per-client test_acc
        client_accuracies = final_clients['test_acc'].astype(float).to_numpy() * 100
        final_acc = float(np.mean(client_accuracies))
        std_dev = float(np.std(client_accuracies, ddof=1)) if len(client_accuracies) > 1 else 0.0
        n_clients = len(client_accuracies)
        margin_error = 1.96 * (std_dev / np.sqrt(n_clients)) if n_clients > 0 else 0.0
        ci_95 = f"[{final_acc - margin_error:.2f}, {final_acc + margin_error:.2f}]"

        final_loss = float(final_clients['test_loss'].astype(float).mean()) if 'test_loss' in final_clients.columns else np.nan
        gen_gap = float(final_clients['acc_gap'].astype(float).mean() * 100) if 'acc_gap' in final_clients.columns else np.nan

        results.append({
            'Method': 'CFL',
            'Final_Accuracy_%': f"{final_acc:.2f}",
            'Std_Dev_%': f"{std_dev:.2f}",
            'CI_95%': ci_95,
            'Generalization_Gap_%': f"{gen_gap:.2f}" if not np.isnan(gen_gap) else "--",
            'Final_Loss': f"{final_loss:.4f}" if not np.isnan(final_loss) else "--"
        })
    except Exception as e:
        print(f"Error processing CFL: {e}")
    
    # 6. Fedge-1000 - compute from server rounds (cluster mean across servers)
    try:
        fedge_path = os.path.join(metrics_dir, 'fedge-1000_rounds.csv')
        used_fallback = False
        if not os.path.exists(fedge_path):
            fedge_path = 'd:/learn/HHAR/Fedge-100/fedge/metrics/servers/rounds.csv'
            used_fallback = True
        fedge_rounds = pd.read_csv(fedge_path)
        if 'global_round' not in fedge_rounds.columns:
            raise ValueError('fedge-1000_rounds.csv missing global_round')
        final_round = int(fedge_rounds['global_round'].max())
        final_rows = fedge_rounds[fedge_rounds['global_round'] == final_round]
        # Choose columns with fallbacks
        acc_col = 'server_partition_test_accuracy' if 'server_partition_test_accuracy' in final_rows.columns else (
            'client_test_accuracy_mean' if 'client_test_accuracy_mean' in final_rows.columns else None)
        loss_col = 'server_partition_test_loss' if 'server_partition_test_loss' in final_rows.columns else (
            'client_test_loss_mean' if 'client_test_loss_mean' in final_rows.columns else None)
        if acc_col is None or loss_col is None:
            raise ValueError('fedge-1000_rounds.csv missing accuracy/loss columns')
        # Compute mean/stats across servers at final round
        acc_vals = final_rows[acc_col].astype(float).to_numpy() * 100.0
        final_acc = float(np.mean(acc_vals))
        std_dev = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0
        n = len(acc_vals)
        margin_error = 1.96 * (std_dev / np.sqrt(n)) if n > 0 else 0.0
        ci_95 = f"[{final_acc - margin_error:.2f}, {final_acc + margin_error:.2f}]"
        final_loss = float(final_rows[loss_col].astype(float).mean())
        print(f"[Fedge-1000] Using path: {fedge_path} ({'fallback' if used_fallback else 'HHAR-metrics copy'})")
        print(f"[Fedge-1000] Final round: {final_round}, acc_col: {acc_col}, loss_col: {loss_col}")
        results.append({
            'Method': 'Fedge-v1',
            'Final_Accuracy_%': f"{final_acc:.2f}",
            'Std_Dev_%': f"{std_dev:.2f}",
            'CI_95%': ci_95,
            'Generalization_Gap_%': "--",
            'Final_Loss': f"{final_loss:.4f}"
        })
    except Exception as e:
        print(f"Error processing Fedge-1000: {e}")
    
    return results

def create_results_csv():
    """Create comprehensive results CSV file"""
    results = extract_final_metrics()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = "d:/learn/HHAR/Plots/comprehensive_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to: {output_path}")
    print("\nComprehensive Results Table:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    create_results_csv()
