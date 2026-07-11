"""Per-pair causal feature engineering.

Every feature is a trailing-window polars expression over data through the
row's own close time. The discipline is mechanical and non-negotiable:
no `shift(-n)`, no `center=True` windows, no statistics computed over the
full history. `tests/test_features_causal.py::test_prefix_invariance`
enforces this for the whole module at once: features computed on a series
truncated at T must be identical to the first T rows of features computed
on the full series.
"""

from __future__ import annotations

import polars as pl

from tml.config import Config, FeaturesConfig
from tml.data import store
from tml.features.cross_sectional import add_cs_ranks


def pair_features(df: pl.DataFrame, fc: FeaturesConfig) -> pl.DataFrame:
    """Compute causal features for a single pair, sorted by ts ascending."""
    df = df.sort("ts")
    log_close = pl.col("close").log()

    exprs: list[pl.Expr] = []

    # Multi-horizon log returns (trailing)
    for h in fc.ret_horizons:
        exprs.append((log_close - log_close.shift(h)).alias(f"ret_{h}"))

    # EWMA volatility of 1-bar log returns
    r1 = log_close - log_close.shift(1)
    for span in fc.vol_spans:
        exprs.append(r1.ewm_std(span=span, adjust=False).alias(f"vol_ewm_{span}"))

    # Momentum = trailing return normalized by trailing vol
    for h in fc.mom_horizons:
        exprs.append(
            ((log_close - log_close.shift(h)) / (r1.ewm_std(span=h, adjust=False) * (h**0.5) + 1e-12)).alias(
                f"mom_{h}"
            )
        )

    # Wilder RSI (causal: EWMs of gains/losses, alpha = 1/period)
    delta = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1 / fc.rsi_period, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1 / fc.rsi_period, adjust=False)
    exprs.append((100 - 100 / (1 + avg_gain / (avg_loss + 1e-12))).alias("rsi"))

    # ATR / close (Wilder smoothing of true range)
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    exprs.append(
        (tr.ewm_mean(alpha=1 / fc.atr_period, adjust=False) / pl.col("close")).alias("atr_norm")
    )

    # Donchian position: where close sits in the trailing [min, max] channel
    w = fc.donchian_window
    roll_hi = pl.col("high").rolling_max(window_size=w)
    roll_lo = pl.col("low").rolling_min(window_size=w)
    exprs.append(((pl.col("close") - roll_lo) / (roll_hi - roll_lo + 1e-12)).alias("donchian_pos"))

    # Volume z-score over a trailing window (log volume to tame spikes)
    lv = (pl.col("volume") + 1.0).log()
    zw = fc.volume_z_window
    exprs.append(
        ((lv - lv.rolling_mean(window_size=zw)) / (lv.rolling_std(window_size=zw) + 1e-12)).alias(
            "volume_z"
        )
    )

    # Calendar encodings (functions of the row's own timestamp only)
    hour = pl.col("ts").dt.hour()
    dow = pl.col("ts").dt.weekday()
    exprs += [
        (2 * 3.141592653589793 * hour / 24).sin().alias("hod_sin"),
        (2 * 3.141592653589793 * hour / 24).cos().alias("hod_cos"),
        (2 * 3.141592653589793 * dow / 7).sin().alias("dow_sin"),
        (2 * 3.141592653589793 * dow / 7).cos().alias("dow_cos"),
    ]

    out = df.with_columns(exprs)
    # Drop warmup rows: every trailing window is fully formed after max_lookback
    out = out.slice(fc.max_lookback, None)
    feature_cols = [c for c in out.columns if c not in ("open", "high", "low", "volume")]
    return out.select(feature_cols).with_columns(
        [pl.col(c).cast(pl.Float32) for c in out.select(feature_cols).columns if c not in ("ts", "pair", "close")]
    )


def build_features(cfg: Config) -> pl.DataFrame:
    """Per-pair causal features, then per-timestamp cross-sectional ranks."""
    frames = []
    for pair in cfg.data.pairs:
        raw = store.scan_raw(cfg, pair).collect()
        frames.append(pair_features(raw, cfg.features))
    all_pairs = pl.concat(frames, how="vertical").sort("ts", "pair")
    if cfg.data.funding:
        from tml.features.funding import add_funding_features

        all_pairs = add_funding_features(all_pairs, cfg)
    return add_cs_ranks(all_pairs, list(cfg.features.cs_rank_cols))


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Model input columns: everything except keys and raw close."""
    return [c for c in df.columns if c not in ("ts", "pair", "close")]


def run_features(cfg: Config) -> int:
    df = build_features(cfg)
    n_null = int(df.null_count().sum_horizontal().item())
    if n_null:
        raise RuntimeError(f"features contain {n_null} nulls after warmup drop — investigate before proceeding")
    path = store.features_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"features: {df.height} rows x {len(feature_columns(df))} feature cols -> {path}")
    return 0
