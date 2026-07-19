"""Cross-sectional features.

Ranks are computed strictly within each timestamp across whichever pairs
exist at that timestamp — point-in-time by construction. No expression here
may aggregate over the time axis.
"""

from __future__ import annotations

import polars as pl


def add_cs_ranks(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Add per-timestamp percentile ranks (in [0, 1]) for the given columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"cs_rank_cols not present in features: {missing}")
    exprs = [
        ((pl.col(c).rank(method="average").over("ts") - 1) / (pl.len().over("ts") - 1).clip(1))
        .cast(pl.Float32)
        .alias(f"cs_rank_{c}")
        for c in cols
    ]
    return df.with_columns(exprs)


def add_cs_zscores(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Add per-timestamp z-scores across whichever pairs exist at each ts.

    A ts with a single pair has no dispersion (std is null under ddof=1) —
    the 0-fill makes its z exactly 0, never a null that would trip the
    engine's no-nulls guard.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"cs_z cols not present in features: {missing}")
    exprs = [
        ((pl.col(c) - pl.col(c).mean().over("ts")) / (pl.col(c).std().over("ts").fill_null(0.0) + 1e-12))
        .cast(pl.Float32)
        .alias(f"cs_z_{c}")
        for c in cols
    ]
    return df.with_columns(exprs)
