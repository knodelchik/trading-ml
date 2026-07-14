"""Taker-flow features, joined to bars point-in-time.

Per pair, on the shared bar-close grid:
    taker_ratio  — the bar's aggressive-buy share, centered on 0 (ratio-0.5)
    taker_ewm    — EWM over ~1 day of the centered ratio (buying-pressure trend)
    taker_z      — z-score of the ratio vs a trailing window
Exact join on (ts, pair) since the taker archive shares the OHLCV grid;
bars with no taker record get neutral zeros plus a taker_known flag.
"""

from __future__ import annotations

import polars as pl

from tml.config import Config
from tml.data.taker import scan_taker

EWM_SPAN = 24
Z_WINDOW = 168

TAKER_COLS = ("taker_ratio", "taker_ewm", "taker_z", "taker_known")


def _per_pair(lf: pl.LazyFrame) -> pl.DataFrame:
    centered = pl.col("taker_buy_ratio") - 0.5
    return (
        lf.sort("ts")
        .with_columns(
            taker_ratio=centered,
            taker_ewm=centered.ewm_mean(span=EWM_SPAN, adjust=False),
            taker_z=(centered - centered.rolling_mean(window_size=Z_WINDOW))
            / (centered.rolling_std(window_size=Z_WINDOW) + 1e-12),
        )
        .select("ts", "pair", "taker_ratio", "taker_ewm", "taker_z")
        .collect()
    )


def add_taker_features(bars: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    from tml.universe import pairs_for

    series = []
    for pair in pairs_for(cfg):
        lf = scan_taker(cfg, pair)
        if lf is not None:
            series.append(_per_pair(lf))
    if not series:
        raise RuntimeError("features.taker enabled but no taker parquet — run `tml ingest`")
    flow = pl.concat(series).sort("pair", "ts")

    out = (
        bars.join(flow, on=["ts", "pair"], how="left")
        .with_columns(taker_known=pl.col("taker_ratio").is_not_null().cast(pl.Float32))
        .with_columns(
            [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in ("taker_ratio", "taker_ewm", "taker_z")]
        )
        .sort("ts", "pair")
    )
    return out
