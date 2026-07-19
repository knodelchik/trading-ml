"""Label construction: volatility-scaled fixed-horizon forward log return.

    y[t] = (log close[t+H] - log close[t]) / (ewm_vol[t] * sqrt(H))

The `shift(-H)` here is the ONLY negative shift permitted in the codebase —
it defines the prediction target, which is future by definition. The
volatility scaler uses strictly trailing data through t.

`label_end_ts` is carried explicitly on every row as the ACTUAL timestamp
of the bar H rows ahead (not `ts + H*bar` wall-clock: across exchange
maintenance gaps the H-th bar closes later than wall-clock arithmetic
suggests, and purging on the shorter wall-clock end would under-purge near
gaps). The CV purge is driven by it, so if the labeling scheme changes
(e.g. triple barrier in M2, where label end varies per sample), purging
stays correct with no changes to the splitter.

Multi-horizon (S4): when labels.horizons is set, every extra horizon H adds
`y_h{H}` / `label_end_ts_h{H}` columns built by the exact same primitive,
so each horizon's purge cutoff reads its own label end.
"""

from __future__ import annotations

import polars as pl

from tml.config import Config
from tml.data import store
from tml.universe import pairs_for


def pair_labels(df: pl.DataFrame, horizon_bars: int, vol_span: int, bar_ms: int) -> pl.DataFrame:
    df = df.sort("ts")
    log_close = pl.col("close").log()
    r1 = log_close - log_close.shift(1)
    # min_samples: EWMA vol over fewer than vol_span observations is unstable
    # (near-zero divisor -> absurd y); those leading rows get null labels.
    vol = r1.ewm_std(span=vol_span, adjust=False, min_samples=vol_span)
    fwd = log_close.shift(-horizon_bars) - log_close  # the target: future by definition
    label_end = pl.col("ts").shift(-horizon_bars)
    # a label whose ACTUAL horizon vastly exceeds H bars spans a trading
    # halt (e.g. LUNA's 18-day 2022 stop): the "24h return" would really be
    # a cross-halt identity jump — null it out
    max_span = pl.duration(milliseconds=int(bar_ms * horizon_bars * 1.5))
    y = (fwd / (vol * horizon_bars**0.5 + 1e-12)).cast(pl.Float32)
    y = pl.when(label_end - pl.col("ts") <= max_span).then(y).otherwise(None)
    return df.select(
        "ts",
        "pair",
        y.alias("y"),
        # actual close time of the bar H rows ahead (null for the last H rows,
        # which carry null y and are excluded downstream anyway)
        label_end.alias("label_end_ts"),
    )


def horizon_cols(horizon: int, base: int) -> tuple[str, str]:
    """(y column, label_end_ts column) for one horizon of the labels frame.

    The base horizon keeps the unsuffixed names, so the single-horizon
    artifact schema — and everything downstream that reads `y` /
    `label_end_ts` — is unchanged when the multi-horizon flag is off.
    """
    if horizon == base:
        return "y", "label_end_ts"
    return f"y_h{horizon}", f"label_end_ts_h{horizon}"


def build_labels(cfg: Config) -> pl.DataFrame:
    horizons = cfg.labels.all_horizons
    base = horizons[0]
    frames: list[pl.DataFrame] = []
    for pair in pairs_for(cfg):
        raw = store.scan_raw(cfg, pair).collect()
        lab = pair_labels(raw, base, cfg.labels.vol_span, cfg.data.bar_ms)
        for h in horizons[1:]:
            ycol, endcol = horizon_cols(h, base)
            extra = pair_labels(raw, h, cfg.labels.vol_span, cfg.data.bar_ms).rename(
                {"y": ycol, "label_end_ts": endcol}
            )
            lab = lab.join(extra, on=["ts", "pair"], how="left", validate="1:1")
        frames.append(lab)
    return pl.concat(frames, how="vertical").sort("ts", "pair")


def run_labels(cfg: Config) -> int:
    df = build_labels(cfg)
    n_null = int(df["y"].null_count())
    expected_min = (cfg.labels.horizon_bars + cfg.labels.vol_span) * len(pairs_for(cfg))
    if n_null < expected_min or (n_null - expected_min) > 0.02 * df.height:
        raise RuntimeError(
            f"suspicious null label count {n_null} (floor {expected_min} = vol_span leading "
            "+ horizon trailing per pair; extras must be rare cross-halt labels) — investigate"
        )
    path = store.labels_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"labels: {df.height} rows ({n_null} trailing nulls to be excluded) -> {path}")
    return 0
