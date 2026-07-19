"""Perp-basis features, joined to bars strictly point-in-time.

Per premium-kline series (per pair, 1h grid):
    basis_close   — the fractional perp-spot basis at the bar close
    basis_ewm     — EWM mean over ~1 day (span 24)
    basis_mom_24h — 24h change in the basis
    basis_z       — z-score vs a trailing 30-day window (720 bars)
Bars receive the latest kline at or before their close via a backward asof
join (funding's alignment discipline). One improvement over funding:
basis_known goes to 0 not only before the perp lists but also whenever the
freshest available kline is older than MAX_STALE_BARS — a delisted perp or
a failed live refresh yields flagged zeros, never a silently frozen basis.
"""

from __future__ import annotations

import polars as pl

from tml.config import Config
from tml.data.basis import scan_basis

EWM_SPAN = 24
MOM_BARS = 24
Z_WINDOW = 720  # 30 days of hourly bars
MAX_STALE_BARS = 2  # same tolerance as the tick's bar-staleness guard

BASIS_COLS = ("basis_close", "basis_ewm", "basis_mom_24h", "basis_z", "basis_known")
_VALUE_COLS = BASIS_COLS[:-1]


def _per_pair_series(lf: pl.LazyFrame) -> pl.DataFrame:
    b = pl.col("basis")
    return (
        lf.sort("ts")
        .with_columns(
            basis_close=b,
            basis_ewm=b.ewm_mean(span=EWM_SPAN, adjust=False),
            basis_mom_24h=b - b.shift(MOM_BARS),
            basis_z=(b - b.rolling_mean(window_size=Z_WINDOW))
            / (b.rolling_std(window_size=Z_WINDOW) + 1e-12),
            basis_ts=pl.col("ts"),  # carried through the asof join for the staleness cutoff
        )
        .select("ts", "pair", *_VALUE_COLS, "basis_ts")
        .collect()
    )


def add_basis_features(bars: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """bars: (ts, pair, ...) -> same frame + BASIS_COLS (float32, 0-filled)."""
    from tml.universe import pairs_for

    series = []
    for pair in pairs_for(cfg):
        lf = scan_basis(cfg, pair)
        if lf is not None:
            series.append(_per_pair_series(lf))
    if not series:
        raise RuntimeError("data.basis is enabled but no basis parquet exists — run `tml ingest`")
    prem = pl.concat(series).sort("pair", "ts")

    max_stale = pl.duration(milliseconds=MAX_STALE_BARS * cfg.data.bar_ms)
    unknown = pl.col("basis_ts").is_null() | ((pl.col("ts") - pl.col("basis_ts")) > max_stale)
    out = (
        bars.sort("pair", "ts")
        .join_asof(prem, on="ts", by="pair", strategy="backward")
        .with_columns(basis_known=(~unknown).cast(pl.Float32))
        .with_columns(
            [
                pl.when(unknown)
                .then(0.0)
                .otherwise(pl.col(c))
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias(c)
                for c in _VALUE_COLS
            ]
        )
        .drop("basis_ts")
        .sort("ts", "pair")
    )
    return out
