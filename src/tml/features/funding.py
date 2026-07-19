"""Funding-rate features, joined to bars strictly point-in-time.

Per funding-settlement series (per pair):
    fund_rate  — the settled rate itself
    fund_ewm   — EWM mean over ~7 days of settlements (span 21)
    fund_z     — z-score vs a trailing ~30-day window (90 settlements)
Bars receive the latest settlement at or before their close via a backward
asof join — the millisecond jitter in settlement times makes this join,
not grid arithmetic, the only safe alignment. Bars before the perp listing
(or before the z-window warms up) get neutral zeros plus an availability
flag so the model can tell "no funding regime" from "flat funding".
"""

from __future__ import annotations

import polars as pl

from tml.config import Config
from tml.data.funding import scan_funding

EWM_SPAN = 21
Z_WINDOW = 90

FUND_COLS = ("fund_rate", "fund_ewm", "fund_z", "fund_known")


def _per_pair_series(lf: pl.LazyFrame) -> pl.DataFrame:
    rate = pl.col("rate")
    return (
        lf.sort("ts")
        .with_columns(
            fund_rate=rate,
            fund_ewm=rate.ewm_mean(span=EWM_SPAN, adjust=False),
            fund_z=(rate - rate.rolling_mean(window_size=Z_WINDOW))
            / (rate.rolling_std(window_size=Z_WINDOW) + 1e-12),
        )
        .select("ts", "pair", "fund_rate", "fund_ewm", "fund_z")
        .collect()
    )


def add_funding_features(bars: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """bars: (ts, pair, ...) -> same frame + FUND_COLS (float32, 0-filled)."""
    from tml.universe import pairs_for

    # all tradable pairs, not the static cfg.data.pairs list: with the PIT
    # universe enabled the dataset spans every ever-member, and looping the
    # static list left ~80 pairs with fund_known=0 despite funding parquet
    # existing on disk
    series = []
    for pair in pairs_for(cfg):
        lf = scan_funding(cfg, pair)
        if lf is not None:
            series.append(_per_pair_series(lf))
    if not series:
        raise RuntimeError("data.funding is enabled but no funding parquet exists — run `tml ingest`")
    fund = pl.concat(series).sort("pair", "ts")

    out = (
        bars.sort("pair", "ts")
        .join_asof(fund, on="ts", by="pair", strategy="backward")
        .with_columns(fund_known=pl.col("fund_rate").is_not_null().cast(pl.Float32))
        .with_columns(
            [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in ("fund_rate", "fund_ewm", "fund_z")]
        )
        .sort("ts", "pair")
    )
    return out
