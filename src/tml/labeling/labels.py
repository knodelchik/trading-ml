"""Label construction: volatility-scaled fixed-horizon forward log return.

    y[t] = (log close[t+H] - log close[t]) / (ewm_vol[t] * sqrt(H))

The `shift(-H)` here is the ONLY negative shift permitted in the codebase —
it defines the prediction target, which is future by definition. The
volatility scaler uses strictly trailing data through t.

`label_end_ts = ts + H bars` is carried explicitly on every row; the CV
purge is driven by it, so if the labeling scheme changes (e.g. triple
barrier in M2, where label end varies per sample), purging stays correct
with no changes to the splitter.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from tml.config import Config
from tml.data import store


def pair_labels(df: pl.DataFrame, horizon_bars: int, vol_span: int, bar_ms: int) -> pl.DataFrame:
    df = df.sort("ts")
    log_close = pl.col("close").log()
    r1 = log_close - log_close.shift(1)
    # min_samples: EWMA vol over fewer than vol_span observations is unstable
    # (near-zero divisor -> absurd y); those leading rows get null labels.
    vol = r1.ewm_std(span=vol_span, adjust=False, min_samples=vol_span)
    fwd = log_close.shift(-horizon_bars) - log_close  # the target: future by definition
    y = fwd / (vol * horizon_bars**0.5 + 1e-12)
    return df.select(
        "ts",
        "pair",
        y.cast(pl.Float32).alias("y"),
        (pl.col("ts") + timedelta(milliseconds=bar_ms * horizon_bars)).alias("label_end_ts"),
    )


def build_labels(cfg: Config) -> pl.DataFrame:
    frames = [
        pair_labels(
            store.scan_raw(cfg, pair).collect(),
            cfg.labels.horizon_bars,
            cfg.labels.vol_span,
            cfg.data.bar_ms,
        )
        for pair in cfg.data.pairs
    ]
    return pl.concat(frames, how="vertical").sort("ts", "pair")


def run_labels(cfg: Config) -> int:
    df = build_labels(cfg)
    n_null = int(df["y"].null_count())
    expected_nulls = (cfg.labels.horizon_bars + cfg.labels.vol_span) * len(cfg.data.pairs)
    if n_null != expected_nulls:
        raise RuntimeError(
            f"unexpected null label count {n_null} (expected exactly {expected_nulls}: "
            "vol_span leading + horizon trailing bars per pair) — investigate"
        )
    path = store.labels_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"labels: {df.height} rows ({n_null} trailing nulls to be excluded) -> {path}")
    return 0
