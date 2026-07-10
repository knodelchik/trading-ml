"""Research pipeline: features + labels -> purged walk-forward OOS predictions.

The backtest consumes exactly the OOS predictions written here, so the
retraining schedule in the backtest is identical to CV by construction.

`run_cv_frames` is pure (frames in, frame out) so tests can drive it with
synthetic data; `run_cv` adds the IO shell and the trials-ledger record.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tml.config import Config
from tml.data import store
from tml.models.lgbm import LGBMAdapter
from tml.validation import metrics
from tml.validation.splits import Fold, PurgedWalkForwardCV, write_folds
from tml.validation.trials import TrialsLedger

KEY_COLS = ("ts", "pair", "close", "y", "label_end_ts")


def dataset(cfg: Config) -> pl.DataFrame:
    feats = pl.read_parquet(store.features_path(cfg))
    labels = pl.read_parquet(store.labels_path(cfg))
    df = feats.join(labels, on=["ts", "pair"], how="inner", validate="1:1")
    return df.filter(pl.col("y").is_not_null()).sort("ts", "pair")


def feature_cols(df: pl.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c not in KEY_COLS)


def run_cv_frames(df: pl.DataFrame, cfg: Config) -> tuple[pl.DataFrame, list[Fold]]:
    """Fit per fold, predict its test window, concatenate OOS predictions."""
    cols = feature_cols(df)
    ts_index = df["ts"].unique().sort()
    folds = PurgedWalkForwardCV(cfg.cv, cfg.data.bar_ms).split(ts_index)
    if not folds:
        raise RuntimeError(
            f"no folds: {len(ts_index)} unique timestamps < train_min({cfg.cv.train_min_bars}) "
            f"+ test_span({cfg.cv.test_span_bars})"
        )
    parts: list[pl.DataFrame] = []
    for fold in folds:
        train = df.filter(fold.train_expr())
        test = df.filter(fold.test_expr())
        if train.is_empty() or test.is_empty():
            raise RuntimeError(f"fold {fold.fold_id} has empty train or test slice")
        model = LGBMAdapter(cfg.model.lgbm)
        model.fit(
            train.select(cols).to_numpy().astype(np.float32),
            train["y"].to_numpy().astype(np.float32),
            cols,
        )
        score = model.predict(test.select(cols).to_numpy().astype(np.float32))
        parts.append(
            test.select("ts", "pair", pl.col("y").alias("y_true")).with_columns(
                score=pl.Series(score, dtype=pl.Float32),
                fold_id=pl.lit(fold.fold_id, dtype=pl.Int32),
            )
        )
        print(f"fold {fold.fold_id}: train={train.height} test={test.height} "
              f"[{fold.test_start:%Y-%m-%d} .. {fold.test_end:%Y-%m-%d})")
    oos = pl.concat(parts, how="vertical")
    dupes = oos.height - oos.select("ts", "pair").unique().height
    if dupes:
        raise RuntimeError(f"OOS predictions contain {dupes} duplicate (ts, pair) rows — folds overlap")
    return oos, folds


def run_cv(cfg: Config) -> int:
    df = dataset(cfg)
    oos, folds = run_cv_frames(df, cfg)

    path = store.oos_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(path)
    write_folds(folds, store.folds_path(cfg))

    scores = oos["score"].to_numpy()
    y = oos["y_true"].to_numpy()
    pooled_ic = metrics.spearman_ic(scores, y)

    # Pre-cost signal SR per period, recorded for the DSR trial history:
    # equal-weight portfolio of per-name positions sized by sign(score).
    port = (
        oos.with_columns(r=pl.col("y_true") * pl.col("score").sign())
        .group_by("ts")
        .agg(pl.col("r").mean())
        .sort("ts")["r"]
        .to_numpy()
    )
    sr_pp = float(np.nanmean(port) / np.nanstd(port, ddof=1)) if len(port) > 2 else None

    TrialsLedger(store.trials_path(cfg)).append(
        {
            "stage": "cv",
            "config_hash": cfg.config_hash(),
            "n_folds": len(folds),
            "n_oos_rows": oos.height,
            "pooled_ic": None if np.isnan(pooled_ic) else pooled_ic,
            "sr_per_period": sr_pp,
        }
    )
    print(f"cv: {oos.height} OOS predictions across {len(folds)} folds -> {path}")
    print(f"cv: pooled Spearman IC = {pooled_ic:+.4f} (full stats: tml report)")
    return 0
