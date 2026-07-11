"""Ensemble combination of per-model test predictions.

The tradeable ensemble score is the plain MEAN of member predictions: every
member regresses the same vol-scaled forward return, so the mean stays in
those units and the backtest's hysteresis thresholds keep their economic
meaning (enter when expected 24h return clears 0.5 vol units). A
rank-average is also computed as a scale-free diagnostic for the report,
but it is deliberately NOT traded — that would be a second strategy variant
and a new ledger trial.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ensemble(scores: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.column_stack(list(scores.values())), axis=1)


def rank_average(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Scale-free combination: average percentile rank, centered on 0."""
    ranks = [stats.rankdata(s) / (len(s) + 1) - 0.5 for s in scores.values()]
    return np.mean(np.column_stack(ranks), axis=1)
