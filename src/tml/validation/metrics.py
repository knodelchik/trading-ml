"""Evaluation metrics with honest statistics.

Overlapping labels (horizon H > 1 bar) make consecutive prediction errors
serially correlated, so naive t-stats overstate significance; IC significance
therefore uses Newey-West HAC standard errors with lag = H. Sharpe estimates
get the Probabilistic Sharpe Ratio treatment, and the Deflated Sharpe Ratio
accounts for how many strategy variants were tried (from the trials ledger).
References: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def spearman_ic(scores: np.ndarray, y: np.ndarray) -> float:
    if len(scores) < 3:
        return float("nan")
    ic = stats.spearmanr(scores, y).statistic
    return float(ic) if ic is not None else float("nan")


def per_ts_ic(ts: np.ndarray, scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional Spearman IC per timestamp (needs >= 3 names per ts)."""
    order = np.argsort(ts, kind="stable")
    ts, scores, y = ts[order], scores[order], y[order]
    uniq, starts = np.unique(ts, return_index=True)
    bounds = np.append(starts, len(ts))
    out_ts, out_ic = [], []
    for i in range(len(uniq)):
        s, e = bounds[i], bounds[i + 1]
        if e - s >= 3:
            ic = spearman_ic(scores[s:e], y[s:e])
            if not math.isnan(ic):
                out_ts.append(uniq[i])
                out_ic.append(ic)
    return np.array(out_ts), np.array(out_ic)


def newey_west_tstat(x: np.ndarray, lag: int) -> float:
    """t-stat of mean(x) with Newey-West (Bartlett kernel) HAC variance."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(xc @ xc) / n
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * float(xc[k:] @ xc[:-k]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / math.sqrt(var / n))


def hit_rate(scores: np.ndarray, y: np.ndarray) -> float:
    mask = (scores != 0) & (y != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(scores[mask]) == np.sign(y[mask])).mean())


def decile_spread(scores: np.ndarray, y: np.ndarray) -> float:
    """Mean label in the top score-decile minus the bottom decile."""
    if len(scores) < 20:
        return float("nan")
    lo, hi = np.quantile(scores, [0.1, 0.9])
    return float(y[scores >= hi].mean() - y[scores <= lo].mean())


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float) -> float:
    """PSR: probability that the true SR exceeds sr_benchmark.

    sr and sr_benchmark are per-period (NOT annualized); kurt is Pearson
    (normal => 3). Bailey & López de Prado (2012).
    """
    if n_obs < 2 or math.isnan(sr):
        return float("nan")
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    sr: float, trial_srs: list[float], n_obs: int, skew: float, kurt: float
) -> dict[str, float]:
    """DSR: PSR against the expected max SR from `len(trial_srs)` trials.

    trial_srs: per-period SR estimates of ALL configurations ever evaluated
    (from the trials ledger). With fewer than 2 trials the expected-max
    benchmark degenerates to 0 (DSR == PSR vs 0).
    """
    n_trials = len(trial_srs)
    if n_trials >= 2:
        var_trials = float(np.var(np.asarray(trial_srs, dtype=np.float64), ddof=1))
        sr0 = math.sqrt(max(var_trials, 0.0)) * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e))
        )
    else:
        sr0 = 0.0
    return {
        "n_trials": float(n_trials),
        "sr_benchmark_expected_max": sr0,
        "dsr": probabilistic_sharpe(sr, sr0, n_obs, skew, kurt),
    }


def moments(returns: np.ndarray) -> tuple[float, float]:
    """(skew, Pearson kurtosis) with NaNs stripped; kurtosis of normal = 3."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 4:
        return 0.0, 3.0
    return float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
