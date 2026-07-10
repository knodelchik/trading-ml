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
