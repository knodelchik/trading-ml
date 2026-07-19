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


def horizon_scales(train_stds: dict[int, float], base: int) -> dict[int, float]:
    """FROZEN per-horizon rescale factors for the multi-horizon blend (S4).

    Each non-base horizon's score is multiplied by std_base/std_h, where the
    stds are of the horizon-ensemble's predictions over its TRAIN window —
    computed at fit time and persisted, never runtime statistics — so every
    blend member shares the base horizon's dispersion and the blended score
    keeps the units the policy thresholds assume. The base scale is exactly
    1.0 (no float noise on the single-horizon path).
    """
    return {h: 1.0 if h == base else train_stds[base] / max(train_stds[h], 1e-12) for h in train_stds}


def blend_horizons(scores: dict[int, np.ndarray], scales: dict[int, float]) -> np.ndarray:
    """Equal-weight blend of per-horizon ensemble scores: weight 1/n each,
    i.e. 0.5*score_h24 + 0.5*rescaled(score_h48) for the registered
    two-horizon trial. A single horizon passes through bit-identical."""
    if len(scores) == 1:
        return next(iter(scores.values()))
    w = 1.0 / len(scores)
    out = np.zeros_like(next(iter(scores.values())))
    for h, s in scores.items():
        out += w * scales[h] * s
    return out
