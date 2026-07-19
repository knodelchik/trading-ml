"""BTC-residual momentum features, computed strictly from past data.

Per pair, on hourly log-returns aligned to BTC's by bar close time:
    beta_btc          — rolling OLS beta to BTC returns (window 720h = 30d)
    corr_btc          — rolling correlation to BTC returns (window 168h = 7d)
    resid_mom_{24,72,168} — trailing sums of the residual return
                        r_pair - beta_btc * r_btc over each lookback
    resid_known       — 1 only once EVERY window above is fully formed
Series are computed from RAW closes (full history, before the per-pair
warmup drop) and joined to the feature frame exactly on (ts, pair) — the
raw grid is a superset of the feature grid. Every rolling window is
trailing through the row's own bar close (min_samples = window size), so
prefix invariance holds including the cross-pair BTC path. Rows whose 720h
beta window has not warmed up get neutral zeros plus the resid_known flag
instead of being dropped: the residual warmup deliberately does NOT extend
fc.max_lookback, so enabling the flag never shrinks the dataset.
"""

from __future__ import annotations

import polars as pl

from tml.config import Config
from tml.data import store

BTC_PAIR = "BTC/USDT"
BETA_WINDOW = 720  # 30 days of hourly bars
CORR_WINDOW = 168  # 7 days
MOM_WINDOWS = (24, 72, 168)

RESID_COLS = (
    "beta_btc",
    "corr_btc",
    "resid_mom_24",
    "resid_mom_72",
    "resid_mom_168",
    "resid_known",
)
# per-ts cross-sectional z-scores added alongside (see engine.build_features)
CS_Z_COLS = ("mom_24", "vol_ewm_168", "resid_mom_72")


def _btc_returns(cfg: Config) -> pl.DataFrame:
    """BTC 1-bar log returns keyed by bar close ts (the regressor series)."""
    d = store.raw_pair_dir(cfg, BTC_PAIR)
    if not d.exists() or not any(d.rglob("*.parquet")):
        raise RuntimeError("data.btc_residual is enabled but no BTC/USDT raw bars exist — run `tml ingest`")
    log_close = pl.col("close").log()
    return (
        store.scan_raw(cfg, BTC_PAIR)
        .sort("ts")
        .select("ts", r_btc=log_close - log_close.shift(1))
        .collect()
    )


def _per_pair(raw: pl.DataFrame, btc: pl.DataFrame) -> pl.DataFrame:
    """(ts, pair, RESID_COLS) for one pair's full raw history, 0-filled."""
    log_close = pl.col("close").log()
    r_pair, r_btc = pl.col("r_pair"), pl.col("r_btc")
    df = (
        raw.sort("ts")
        .select("ts", "pair", "close")
        .join(btc, on="ts", how="left")
        .with_columns(r_pair=log_close - log_close.shift(1))
        .with_columns(
            # OLS beta = cov(r_pair, r_btc) / var(r_btc), both over the same
            # trailing window ending at the row's own close (ddof cancels)
            beta_btc=pl.rolling_cov(r_pair, r_btc, window_size=BETA_WINDOW)
            / (r_btc.rolling_var(window_size=BETA_WINDOW) + 1e-12),
            corr_btc=pl.rolling_corr(r_pair, r_btc, window_size=CORR_WINDOW),
        )
        .with_columns(resid=r_pair - pl.col("beta_btc") * r_btc)
        .with_columns(
            [pl.col("resid").rolling_sum(window_size=h).alias(f"resid_mom_{h}") for h in MOM_WINDOWS]
        )
    )
    value_cols = [c for c in RESID_COLS if c != "resid_known"]
    return (
        df.with_columns(
            resid_known=pl.all_horizontal([pl.col(c).is_not_null() for c in value_cols]).cast(
                pl.Float32
            )
        )
        .with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in value_cols])
        .select("ts", "pair", *RESID_COLS)
    )


def add_residual_features(bars: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """bars: (ts, pair, ...) -> same frame + RESID_COLS (float32, 0-filled)."""
    from tml.universe import pairs_for

    btc = _btc_returns(cfg)
    series = [_per_pair(store.scan_raw(cfg, pair).collect(), btc) for pair in pairs_for(cfg)]
    resid = pl.concat(series).sort("pair", "ts")

    out = (
        bars.join(resid, on=["ts", "pair"], how="left")
        .with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in RESID_COLS])
        .sort("ts", "pair")
    )
    return out
