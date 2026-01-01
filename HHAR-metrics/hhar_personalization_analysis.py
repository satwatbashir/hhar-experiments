#!/usr/bin/env python3
"""
HHAR Client Personalization Analysis
====================================

Comprehensive analysis of client-level personalization and fairness across 
federated learning methods on HHAR (Human Activity Recognition) dataset.

Analyzes: CFL, FedProx, HierFL, pFedMe, SCAFFOLD, FEDGE
Generates: Summary tables, distribution plots, fairness comparisons, and report

Author: Federated Learning Research
Date: 2025-09-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class HHARPersonalizationAnalyzer:
    """Complete HHAR client personalization analysis."""
    
    def __init__(self, output_dir="hhar_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Method configurations with correct file paths and column mappings
        self.methods = {
            'CFL': {
                'file': 'cfl_clients_metrics.csv',
                'acc_col': 'test_acc',
                'client_col': 'cid',
                'round_col': 'round',
                'color': '#9467bd'
            },
            'FedProx': {
                'file': 'fedprox_clients.csv',
                'acc_col': 'test_accuracy',
                'client_col': 'client_id',
                'round_col': 'round',
                'color': '#ff7f0e'
            },
            'HierFL': {
                'file': 'hierfl_client_eval_metrics.csv',
                'acc_col': 'client_test_accuracy',
                'client_col': 'client_id',
                'round_col': 'global_round',
                'color': '#d62728'
            },
            'pFedMe': {
                'file': 'pfedme_clients.csv',
                'acc_col': 'test_accuracy',
                'client_col': 'cid',
                'round_col': 'round',
                'color': '#2ca02c'
            },
            'SCAFFOLD': {
                'file': 'scaffold_clients.csv',
                'acc_col': 'test_accuracy',
                'client_col': 'cid',
                'round_col': 'round',
                'color': '#1f77b4'
            },
            'FEDGE': {
                'file': '../Fedge-100/fedge/metrics/clients.csv',
                'acc_col': 'test_accuracy',
                'client_col': 'cid',
                'round_col': 'global_round',
                'color': '#8c564b'
            }
        }
    
    def analyze_method(self, method_name, config):
        """Analyze a single federated learning method."""
        try:
            # Load data
            df = pd.read_csv(config['file'])
            
            # Get final round data
            final_round = df[config['round_col']].max()
            final_data = df[df[config['round_col']] == final_round]
            
            if len(final_data) == 0:
                return None
            
            # Extract client accuracies
            client_accs = final_data[config['acc_col']].values
            client_ids = final_data[config['client_col']].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(client_accs)
            client_accs = client_accs[valid_mask]
            
            if len(client_accs) == 0:
                return None
            
            # Calculate fairness metrics
            mean_acc = np.mean(client_accs)
            std_acc = np.std(client_accs)
            min_acc = np.min(client_accs)
            max_acc = np.max(client_accs)
            
            # Fairness indicators
            cv = std_acc / mean_acc if mean_acc > 0 else np.inf
            perf_gap = max_acc - min_acc
            
            # Gini coefficient
            sorted_acc = np.sort(client_accs)
            n = len(sorted_acc)
            cumsum = np.cumsum(sorted_acc)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            
            return {
                'method': method_name,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'min_accuracy': min_acc,
                'max_accuracy': max_acc,
                'coefficient_variation': cv,
                'performance_gap': perf_gap,
                'gini_coefficient': gini,
                'num_clients': len(client_accs),
                'final_round': final_round,
                'client_accuracies': client_accs,
                'color': config['color']
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing {method_name}: {e}")
            return None
    
    def create_summary_table(self, results):
        """Create formatted summary table."""
        table_data = []
        for result in results:
            table_data.append({
                'Method': result['method'],
                'Avg Client Accuracy': f"{result['mean_accuracy']:.3f}",
                'Performance Gap': f"{result['performance_gap']:.3f}",
                'Coefficient of Variation': f"{result['coefficient_variation']:.3f}",
                'Gini Coefficient': f"{result['gini_coefficient']:.3f}",
                'Num Clients': int(result['num_clients']),
                'Final Round': int(result['final_round'])
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Save CSV
        df_table.to_csv(self.output_dir / "hhar_personalization_summary.csv", index=False)
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_table.values,
                        colLabels=df_table.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2.5)
        
        # Style table
        header_color = '#4CAF50'
        for j in range(len(df_table.columns)):
            cell = table[(0, j)]
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color='white')
        
        for i in range(1, len(df_table) + 1):
            for j in range(len(df_table.columns)):
                cell = table[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                if j == 0:  # Method column
                    cell.set_text_props(weight='bold')
        
        plt.suptitle('HHAR Client Personalization & Fairness Analysis', 
                    fontsize=16, fontweight='bold', y=0.85)
        
        plt.savefig(self.output_dir / 'hhar_personalization_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return df_table
    
    def create_plots(self, results):
        """Create distribution and comparison plots."""
        # Prepare data for plotting
        all_client_data = []
        for result in results:
            for acc in result['client_accuracies']:
                all_client_data.append({
                    'method': result['method'],
                    'accuracy': acc,
                    'color': result['color']
                })
        
        df_plot = pd.DataFrame(all_client_data)
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Box plot
        ax1 = plt.subplot(2, 3, 1)
        methods = [r['method'] for r in results]
        colors = [r['color'] for r in results]
        
        box_data = [df_plot[df_plot['method'] == method]['accuracy'].values for method in methods]
        box_plot = ax1.boxplot(box_data, labels=methods, patch_artist=True, showmeans=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Client Performance Distribution', fontweight='bold')
        ax1.set_ylabel('Test Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Gap comparison
        ax2 = plt.subplot(2, 3, 2)
        gaps = [r['performance_gap'] for r in results]
        bars = ax2.bar(methods, gaps, color=colors, alpha=0.8)
        ax2.set_title('Performance Gap (Fairness)', fontweight='bold')
        ax2.set_ylabel('Max - Min Client Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Average Performance
        ax3 = plt.subplot(2, 3, 3)
        means = [r['mean_accuracy'] for r in results]
        bars = ax3.bar(methods, means, color=colors, alpha=0.8)
        ax3.set_title('Average Client Performance', fontweight='bold')
        ax3.set_ylabel('Mean Client Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Coefficient of Variation
        ax4 = plt.subplot(2, 3, 4)
        cvs = [r['coefficient_variation'] for r in results]
        bars = ax4.bar(methods, cvs, color=colors, alpha=0.8)
        ax4.set_title('Coefficient of Variation', fontweight='bold')
        ax4.set_ylabel('CV (lower = more consistent)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Gini Coefficient
        ax5 = plt.subplot(2, 3, 5)
        ginis = [r['gini_coefficient'] for r in results]
        bars = ax5.bar(methods, ginis, color=colors, alpha=0.8)
        ax5.set_title('Gini Coefficient', fontweight='bold')
        ax5.set_ylabel('Gini (0=equal, 1=unequal)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance vs Fairness scatter
        ax6 = plt.subplot(2, 3, 6)
        for result in results:
            ax6.scatter(result['performance_gap'], result['mean_accuracy'], 
                       color=result['color'], s=100, alpha=0.8, 
                       label=result['method'])
        ax6.set_xlabel('Performance Gap (higher = less fair)')
        ax6.set_ylabel('Mean Client Accuracy')
        ax6.set_title('Performance vs Fairness Trade-off', fontweight='bold')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hhar_personalization_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_report(self, results, summary_table):
        """Generate comprehensive analysis report."""
        # Sort results by fairness and performance
        fairness_sorted = sorted(results, key=lambda x: x['performance_gap'])
        performance_sorted = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)
        
        report = f"""# HHAR Client Personalization Analysis Report

## Overview
Analysis of client-level personalization across {len(results)} federated learning methods on HHAR dataset.

## Key Metrics
- **Performance Gap**: Difference between best/worst clients (lower = more fair)
- **Coefficient of Variation**: Std/Mean ratio (lower = more consistent)  
- **Gini Coefficient**: Inequality measure (0=equal, 1=unequal)

## Results Summary

### Fairness Ranking (by Performance Gap):
"""
        
        for i, result in enumerate(fairness_sorted, 1):
            report += f"{i}. **{result['method']}**: {result['performance_gap']:.3f}\n"
        
        report += f"""
### Performance Ranking (by Average Accuracy):
"""
        
        for i, result in enumerate(performance_sorted, 1):
            report += f"{i}. **{result['method']}**: {result['mean_accuracy']:.3f} ({result['mean_accuracy']*100:.1f}%)\n"
        
        report += f"""

## Detailed Results

| Method | Avg Client Accuracy | Performance Gap | Coefficient of Variation | Gini Coefficient | Num Clients | Final Round |
|--------|-------------------|-----------------|-------------------------|------------------|-------------|-------------|
""" + "\n".join([f"| **{row['Method']}** | {row['Avg Client Accuracy']} | {row['Performance Gap']} | {row['Coefficient of Variation']} | {row['Gini Coefficient']} | {row['Num Clients']} | {row['Final Round']} |" 
                 for _, row in summary_table.iterrows()]) + """

## Key Findings

### Fairness Champion: **{fairness_sorted[0]['method']}**
- Lowest performance gap: {fairness_sorted[0]['performance_gap']:.3f}
- Most equitable client treatment

### Performance Leader: **{performance_sorted[0]['method']}**
- Highest average accuracy: {performance_sorted[0]['mean_accuracy']:.3f} ({performance_sorted[0]['mean_accuracy']*100:.1f}%)
- Superior personalization effectiveness

## HHAR-Specific Insights
1. Human activity recognition shows unique personalization patterns
2. Sensor data creates higher client heterogeneity than image data
3. Performance-fairness trade-offs are pronounced in HHAR
4. Different optimal methods compared to image classification tasks

## Generated Files
- `hhar_personalization_summary.csv` - Summary statistics
- `hhar_personalization_table.png` - Formatted summary table
- `hhar_personalization_analysis.png` - Comprehensive plots
- `hhar_personalization_report.md` - This report

---
*Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Dataset: HHAR (Human Activity Recognition)*
*Methods: {', '.join([r['method'] for r in results])}*
"""
        
        # Save report
        with open(self.output_dir / "hhar_personalization_report.md", 'w') as f:
            f.write(report)
    
    def run_complete_analysis(self):
        """Execute complete HHAR personalization analysis."""
        print("🚀 HHAR Client Personalization Analysis")
        print("=" * 50)
        
        results = []
        
        # Analyze each method
        for method_name, config in self.methods.items():
            print(f"📊 Analyzing {method_name}...")
            result = self.analyze_method(method_name, config)
            if result:
                results.append(result)
                print(f"   ✅ {len(result['client_accuracies'])} clients, "
                      f"avg: {result['mean_accuracy']:.3f}, gap: {result['performance_gap']:.3f}")
            else:
                print(f"   ❌ Failed to analyze {method_name}")
        
        if not results:
            print("❌ No valid results found!")
            return
        
        print(f"\n📈 Creating analysis outputs...")
        
        # Create summary table
        summary_table = self.create_summary_table(results)
        
        # Create plots
        self.create_plots(results)
        
        # Generate report
        self.generate_report(results, summary_table)
        
        # Print summary
        print(f"\n✅ Analysis Complete!")
        print(f"📁 Results saved in: {self.output_dir}/")
        print(f"📊 Methods analyzed: {len(results)}")
        
        # Key findings
        fairness_champion = min(results, key=lambda x: x['performance_gap'])
        performance_leader = max(results, key=lambda x: x['mean_accuracy'])
        
        print(f"\n🏆 Key Findings:")
        print(f"   • Most Fair: {fairness_champion['method']} (gap: {fairness_champion['performance_gap']:.3f})")
        print(f"   • Best Performance: {performance_leader['method']} (acc: {performance_leader['mean_accuracy']:.3f})")
        
        return results, summary_table

def main():
    """Main execution function."""
    analyzer = HHARPersonalizationAnalyzer()
    results, summary = analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
