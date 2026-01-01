# HHAR Client Personalization Analysis Report

## Overview
Analysis of client-level personalization across 6 federated learning methods on HHAR dataset.

## Key Metrics
- **Performance Gap**: Difference between best/worst clients (lower = more fair)
- **Coefficient of Variation**: Std/Mean ratio (lower = more consistent)  
- **Gini Coefficient**: Inequality measure (0=equal, 1=unequal)

## Results Summary

### Fairness Ranking (by Performance Gap):
1. **CFL**: 0.149
2. **FedProx**: 0.200
3. **SCAFFOLD**: 0.236
4. **pFedMe**: 0.337
5. **FEDGE**: 0.511
6. **HierFL**: 0.855

### Performance Ranking (by Average Accuracy):
1. **FEDGE**: 0.814 (81.4%)
2. **pFedMe**: 0.350 (35.0%)
3. **HierFL**: 0.284 (28.4%)
4. **CFL**: 0.209 (20.9%)
5. **SCAFFOLD**: 0.204 (20.4%)
6. **FedProx**: 0.182 (18.2%)


## Detailed Results

| Method | Avg Client Accuracy | Performance Gap | Coefficient of Variation | Gini Coefficient | Num Clients | Final Round |
|--------|-------------------|-----------------|-------------------------|------------------|-------------|-------------|
| **CFL** | 0.209 | 0.149 | 0.199 | 0.109 | 9 | 100 |
| **FedProx** | 0.182 | 0.200 | 0.350 | 0.198 | 9 | 100 |
| **HierFL** | 0.284 | 0.855 | 0.845 | 0.366 | 9 | 99 |
| **pFedMe** | 0.350 | 0.337 | 0.267 | 0.149 | 9 | 100 |
| **SCAFFOLD** | 0.204 | 0.236 | 0.369 | 0.205 | 9 | 100 |
| **FEDGE** | 0.814 | 0.511 | 0.205 | 0.108 | 9 | 100 |

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
