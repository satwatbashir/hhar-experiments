# fedge/stats.py
"""
Shared statistical utilities for confidence intervals and convergence metrics.
"""

import math


def _t_critical_95(df: int) -> float:
    """Get critical t-value for 95% confidence interval given degrees of freedom."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
    }
    if df <= 0:
        return 0.0
    if df <= 30:
        return table[df]
    return 1.96


def _mean_std_ci(values: list[float]) -> tuple[float, float, float, float]:
    """
    Calculate mean, standard deviation, and 95% confidence interval using t-distribution.

    Returns:
        tuple: (mean, std, ci_low, ci_high)
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(sum(values)) / n
    if n == 1:
        return (mean, 0.0, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t = _t_critical_95(n - 1)
    margin = t * std / math.sqrt(n)
    return (mean, std, mean - margin, mean + margin)
