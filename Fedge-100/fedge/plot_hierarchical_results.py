#!/usr/bin/env python3
"""
Clean hierarchical federated learning results plotter for HHAR.
Generates focused plots for server accuracy, loss, and performance analysis.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class HierarchicalResultsAnalyzer:
    """Clean analyzer for hierarchical FL results"""
    
    def __init__(self, metrics_dir="./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.cloud_metrics_dir = self.metrics_dir / "cloud"
        self.server_data = {}
        self.cloud_data = pd.DataFrame()
        self.available_rounds = []
        
        # Create plots directory
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all data
        self._load_data()
    
    def _load_data(self):
        """Load server and cloud metrics from correct locations"""
        print("📊 Loading hierarchical FL metrics...")
        
        # Load server metrics from consolidated CSV: metrics/servers/rounds.csv
        servers_dir = self.metrics_dir / "servers"
        servers_rounds_csv = servers_dir / "rounds.csv"
        if servers_rounds_csv.exists():
            try:
                df_all = pd.read_csv(servers_rounds_csv)
                # Split by server_id
                for sid in sorted(df_all["server_id"].unique()):
                    df_sid = (
                        df_all[df_all["server_id"] == sid]
                        .sort_values("global_round")
                        .reset_index(drop=True)
                    )
                    self.server_data[int(sid)] = df_sid
                    print(f"✅ Loaded {len(df_sid)} rows for Server {sid} from servers/rounds.csv")

                # Compute cluster metrics across servers per round
                acc_col = next((c for c in [
                    'server_partition_test_accuracy',
                    'client_test_accuracy_mean',
                    'agg_acc'
                ] if c in df_all.columns), None)
                loss_col = next((c for c in [
                    'server_partition_test_loss',
                    'client_test_loss_mean',
                    'agg_loss'
                ] if c in df_all.columns), None)
                if acc_col and loss_col:
                    g = df_all.groupby('global_round')
                    cluster_df = pd.DataFrame({
                        'global_round': g.size().index.values,
                        'cluster_accuracy_mean': g[acc_col].mean().values,
                        'cluster_accuracy_std': g[acc_col].std().fillna(0.0).values,
                        'cluster_loss_mean': g[loss_col].mean().values,
                        'cluster_loss_std': g[loss_col].std().fillna(0.0).values,
                        'num_clusters': g['server_id'].nunique().values,
                    })
                    self.cloud_data = cluster_df.sort_values('global_round').reset_index(drop=True)
                    print(f"✅ Computed cluster metrics from servers: {len(self.cloud_data)} rounds, {int(self.cloud_data['num_clusters'].max())} clusters")
                else:
                    print("⚠️ Could not compute cluster metrics (missing accuracy/loss columns)")
            except Exception as e:
                print(f"⚠️ Error loading servers/rounds.csv: {e}")
        else:
            print(f"⚠️ Server metrics not found at {servers_rounds_csv}")
        
        # Load cloud metrics from metrics/global_metrics.csv (Fedge-100 format) as a fallback
        global_metrics_csv = self.metrics_dir / "global_metrics.csv"
        if global_metrics_csv.exists():
            try:
                df_global = pd.read_csv(global_metrics_csv)
                # Only use this if we don't already have cluster metrics
                if getattr(self.cloud_data, 'empty', True):
                    self.cloud_data = df_global
                    print(f"✅ Loaded {len(self.cloud_data)} global rounds from global_metrics.csv")
                else:
                    print(f"ℹ️ Global metrics available ({len(df_global)} rounds) but cluster metrics will be used for plotting")
            except Exception as e:
                print(f"⚠️ Error loading global metrics: {e}")
        else:
            # Fallback to legacy location if present
            cloud_file = self.cloud_metrics_dir / "cloud_round_metrics.csv"
            if cloud_file.exists():
                try:
                    df_legacy = pd.read_csv(cloud_file)
                    if getattr(self.cloud_data, 'empty', True):
                        self.cloud_data = df_legacy
                        print(f"✅ Loaded {len(self.cloud_data)} cloud rounds from cloud/cloud_round_metrics.csv")
                    else:
                        print(f"ℹ️ Legacy cloud metrics available ({len(df_legacy)} rounds) but cluster metrics will be used for plotting")
                except Exception as e:
                    print(f"⚠️ Error loading cloud metrics: {e}")
            else:
                print(f"⚠️ Cloud/global metrics not found at {global_metrics_csv} or {self.cloud_metrics_dir / 'cloud_round_metrics.csv'}")
        
        # Determine available rounds (prefer union of server rounds, else cloud)
        if self.server_data:
            all_rounds = set()
            for df in self.server_data.values():
                if "global_round" in df.columns:
                    all_rounds.update(df["global_round"].astype(int).tolist())
            if all_rounds:
                self.available_rounds = sorted(all_rounds)
                print(f"🔍 Found {len(self.available_rounds)} rounds from servers: {self.available_rounds[0]}–{self.available_rounds[-1]}")
        elif not getattr(self.cloud_data, "empty", True):
            if "global_round" in self.cloud_data.columns:
                gr = self.cloud_data["global_round"].astype(int)
                if not gr.empty:
                    self.available_rounds = gr.tolist()
                    print(f"🔍 Found {len(self.available_rounds)} rounds from cloud metrics")
    
    
    
    def plot_server_accuracy(self):
        """Plot server test accuracy vs rounds"""
        if not self.server_data:
            print("⚠️ No server data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        for server_id in range(3):
            if server_id in self.server_data:
                df = self.server_data[server_id]
                rounds = df['global_round'].values
                acc_col = next((c for c in [
                    'server_partition_test_accuracy',
                    'client_test_accuracy_mean',
                    'agg_acc'
                ] if c in df.columns), None)
                if not acc_col:
                    print(f"⚠️ No accuracy column found for Server {server_id}")
                    continue
                accuracies = df[acc_col].values
                
                ax.plot(rounds, accuracies, color=colors[server_id], 
                       linewidth=2, marker=markers[server_id], markersize=4,
                       label=f'Server {server_id}')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Server Test Accuracy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'server_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Server accuracy plot saved")
    
    def plot_server_loss(self):
        """Plot server test loss vs rounds"""
        if not self.server_data:
            print("⚠️ No server data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        for server_id in range(3):
            if server_id in self.server_data:
                df = self.server_data[server_id]
                rounds = df['global_round'].values
                loss_col = next((c for c in [
                    'server_partition_test_loss',
                    'client_test_loss_mean',
                    'agg_loss'
                ] if c in df.columns), None)
                if not loss_col:
                    print(f"⚠️ No loss column found for Server {server_id}")
                    continue
                losses = df[loss_col].values
                
                ax.plot(rounds, losses, color=colors[server_id], 
                       linewidth=2, marker=markers[server_id], markersize=4,
                       label=f'Server {server_id}')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Loss')
        ax.set_title('Server Test Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'server_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Server loss plot saved")
    
    def plot_individual_servers(self):
        """Create individual accuracy/loss plots for each server"""
        if not self.server_data:
            print("⚠️ No server data available for plotting")
            return
        
        for server_id in range(3):
            if server_id in self.server_data:
                df = self.server_data[server_id]
                rounds = df['global_round'].values
                acc_col = next((c for c in [
                    'server_partition_test_accuracy',
                    'client_test_accuracy_mean',
                    'agg_acc'
                ] if c in df.columns), None)
                loss_col = next((c for c in [
                    'server_partition_test_loss',
                    'client_test_loss_mean',
                    'agg_loss'
                ] if c in df.columns), None)
                if not acc_col or not loss_col:
                    print(f"⚠️ Missing columns for Server {server_id}: acc_col={acc_col}, loss_col={loss_col}")
                    continue
                accuracies = df[acc_col].values
                losses = df[loss_col].values
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f'Server {server_id} Performance', fontsize=16, fontweight='bold')
                
                # Accuracy plot
                ax1.plot(rounds, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
                ax1.set_xlabel('Training Round')
                ax1.set_ylabel('Test Accuracy')
                ax1.set_title(f'Server {server_id} Test Accuracy')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1)
                ax1.set_yticks(np.arange(0, 1.1, 0.2))
                
                # Loss plot
                ax2.plot(rounds, losses, 'r-', linewidth=2, marker='s', markersize=3)
                ax2.set_xlabel('Training Round')
                ax2.set_ylabel('Test Loss')
                ax2.set_title(f'Server {server_id} Test Loss')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'server_{server_id}_performance.png', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✅ Server {server_id} individual plot saved")
    
    def plot_global_vs_servers(self):
        """Plot cloud clustering vs server performance comparison"""
        if not self.server_data or self.cloud_data.empty:
            print("⚠️ No data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cloud accuracy plot
        rounds = self.cloud_data['global_round'].values
        # Prefer cluster metrics; fallback to global metrics if clusters not available
        if 'cluster_accuracy_mean' in self.cloud_data.columns:
            cloud_acc = self.cloud_data['cluster_accuracy_mean'].values
            ax1.plot(rounds, cloud_acc, 'darkgreen', linewidth=2, marker='s', 
                    markersize=5, label='Cluster Mean Accuracy')
        elif 'global_accuracy' in self.cloud_data.columns:
            cloud_acc = self.cloud_data['global_accuracy'].values
            ax1.plot(rounds, cloud_acc, 'darkblue', linewidth=2, marker='o', 
                    markersize=5, label='Global Accuracy')
        
        # Add individual server averages
        for server_id in range(3):
            if server_id in self.server_data:
                df = self.server_data[server_id]
                server_rounds = df['global_round'].values
                acc_col = next((c for c in [
                    'server_partition_test_accuracy',
                    'client_test_accuracy_mean',
                    'agg_acc'
                ] if c in df.columns), None)
                if acc_col:
                    server_acc = df[acc_col].values
                    ax1.plot(server_rounds, server_acc, '--', alpha=0.7, linewidth=2,
                           label=f'Server {server_id}')
        
        ax1.set_xlabel('Training Round')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Cloud Clustering vs Server Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Cloud loss plot
        if 'cluster_loss_mean' in self.cloud_data.columns:
            cl_loss = self.cloud_data['cluster_loss_mean'].values
            ax2.plot(rounds, cl_loss, 'darkgreen', linewidth=2, marker='s', 
                    markersize=5, label='Cluster Mean Loss')
        elif 'global_loss' in self.cloud_data.columns:
            cloud_loss_g = self.cloud_data['global_loss'].values
            ax2.plot(rounds, cloud_loss_g, 'darkred', linewidth=2, marker='o', 
                    markersize=5, label='Global Loss')
        
        # Add individual server losses
        for server_id in range(3):
            if server_id in self.server_data:
                df = self.server_data[server_id]
                server_rounds = df['global_round'].values
                loss_col = next((c for c in [
                    'server_partition_test_loss',
                    'client_test_loss_mean',
                    'agg_loss'
                ] if c in df.columns), None)
                if loss_col:
                    server_loss = df[loss_col].values
                    ax2.plot(server_rounds, server_loss, '--', alpha=0.7, linewidth=2,
                           label=f'Server {server_id}')
        
        ax2.set_xlabel('Training Round')
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Cloud Clustering Loss Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'global_vs_servers.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Cloud vs servers plot saved")
    
    def generate_report(self):
        """Generate summary report"""
        if not self.server_data and self.cloud_data.empty:
            print("⚠️ No data available for report")
            return
        
        print("\n" + "="*70)
        print("HIERARCHICAL FEDERATED LEARNING ANALYSIS REPORT")
        print("="*70)
        
        # Server performance summary
        if self.server_data:
            print(f"\n📊 SERVER PERFORMANCE SUMMARY:")
            for server_id in range(3):
                if server_id in self.server_data:
                    df = self.server_data[server_id]
                    acc_col = next((c for c in [
                        'server_partition_test_accuracy',
                        'client_test_accuracy_mean',
                        'agg_acc'
                    ] if c in df.columns), None)
                    loss_col = next((c for c in [
                        'server_partition_test_loss',
                        'client_test_loss_mean',
                        'agg_loss'
                    ] if c in df.columns), None)
                    if not acc_col or not loss_col:
                        print(f"   • Server {server_id}: Missing columns, skipping summary")
                        continue
                    final_acc = df[acc_col].iloc[-1]
                    final_loss = df[loss_col].iloc[-1]
                    max_acc = df[acc_col].max()
                    print(f"   • Server {server_id}: Final={final_acc:.3f}, Peak={max_acc:.3f}, Loss={final_loss:.3f}")
        
        # Cloud clustering summary
        if not self.cloud_data.empty:
            print(f"\n🌩️ CLOUD CLUSTERING SUMMARY (Preferred):")
            final_cloud = self.cloud_data.iloc[-1]
            if 'cluster_accuracy_mean' in self.cloud_data.columns:
                final_cluster_acc = float(final_cloud['cluster_accuracy_mean'])
                max_cluster_acc = float(self.cloud_data['cluster_accuracy_mean'].max())
                print(f"   • Final Cluster Mean Accuracy: {final_cluster_acc:.3f}")
                print(f"   • Peak Cluster Mean Accuracy: {max_cluster_acc:.3f}")
            if 'cluster_loss_mean' in self.cloud_data.columns:
                final_cluster_loss = float(final_cloud['cluster_loss_mean'])
                print(f"   • Final Cluster Mean Loss: {final_cluster_loss:.3f}")
            if 'num_clusters' in self.cloud_data.columns:
                final_clusters = int(final_cloud['num_clusters'])
                print(f"   • Clusters (servers) considered per round: {final_clusters}")
            # Only show global if cluster is not available
            if 'cluster_accuracy_mean' not in self.cloud_data.columns and 'global_accuracy' in self.cloud_data.columns:
                final_global_acc = float(final_cloud['global_accuracy'])
                max_global_acc = float(self.cloud_data['global_accuracy'].max())
                print(f"   • Final Global Accuracy: {final_global_acc:.3f}")
                print(f"   • Peak Global Accuracy: {max_global_acc:.3f}")
            print(f"   • Total Rounds Analyzed: {len(self.cloud_data)}")
        
        print("="*70)
    
    def plot_cloud_metrics(self):
        """Plot cloud metrics (accuracy and loss) vs rounds"""
        if self.cloud_data.empty:
            print("⚠️ No cloud data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        rounds = self.cloud_data['global_round'].values
        
        # Cloud accuracy plot (prefer cluster metrics)
        if 'cluster_accuracy_mean' in self.cloud_data.columns:
            cluster_acc = self.cloud_data['cluster_accuracy_mean'].values
            ax1.plot(rounds, cluster_acc, 'darkgreen', linewidth=2, marker='s', 
                    markersize=5, label='Cluster Mean Accuracy')
            # Optionally overlay global if available for reference
            if 'global_accuracy' in self.cloud_data.columns:
                cloud_acc = self.cloud_data['global_accuracy'].values
                ax1.plot(rounds, cloud_acc, color='gray', linewidth=2, linestyle='--', 
                        markersize=3, label='Global Accuracy (ref)')
        elif 'global_accuracy' in self.cloud_data.columns:
            cloud_acc = self.cloud_data['global_accuracy'].values
            ax1.plot(rounds, cloud_acc, 'darkblue', linewidth=2, marker='o', 
                    markersize=5, label='Global Accuracy')
        
        ax1.set_xlabel('Global Round')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Cluster Accuracy Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Cloud loss plot (prefer cluster metrics)
        if 'cluster_loss_mean' in self.cloud_data.columns:
            cluster_loss = self.cloud_data['cluster_loss_mean'].values
            ax2.plot(rounds, cluster_loss, 'darkgreen', linewidth=2, marker='s', 
                    markersize=5, label='Cluster Mean Loss')
            # Optionally overlay global if available for reference
            if 'global_loss' in self.cloud_data.columns:
                cloud_loss = self.cloud_data['global_loss'].values
                ax2.plot(rounds, cloud_loss, color='gray', linewidth=2, linestyle='--', 
                        markersize=3, label='Global Loss (ref)')
        elif 'global_loss' in self.cloud_data.columns:
            cloud_loss = self.cloud_data['global_loss'].values
            ax2.plot(rounds, cloud_loss, 'darkred', linewidth=2, marker='o', 
                    markersize=5, label='Global Loss')
        
        ax2.set_xlabel('Global Round')
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Cluster Loss Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cloud_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Cloud metrics plot saved")

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Hierarchical FL Analysis...")
        
        if not self.server_data and self.cloud_data.empty:
            print("❌ No data found for analysis")
            return
        
        # Generate plots
        print("\n📈 Generating plots...")
        # Prioritize cloud metrics
        if not self.cloud_data.empty:
            self.plot_cloud_metrics()
            self.plot_global_vs_servers()
        
        # Also show server-level metrics
        self.plot_server_accuracy()
        self.plot_server_loss()
        self.plot_individual_servers()
        
        # Generate report
        self.generate_report()
        
        print(f"\n✅ Analysis complete! Plots saved to: {self.output_dir.resolve()}")

if __name__ == "__main__":
    analyzer = HierarchicalResultsAnalyzer()
    analyzer.run_analysis()
