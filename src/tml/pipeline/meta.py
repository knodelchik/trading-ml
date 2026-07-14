"""Meta-labeling bet sizing (López de Prado).

A secondary classifier predicts P(a triggered trade is profitable) from the
same features plus the primary score. Position size then scales with that
probability, so capital concentrates where the model is confident and skips
low-conviction signals — directly attacking the ~7%-average under-deployment
the M5.2 review measured.

Leakage safety: the meta-model is trained walk-forward on the PRIMARY
model's OOS predictions. Fold k's meta-model trains only on rows from folds
< k (themselves out-of-sample), then predicts fold k. Fold 0 has no prior
OOS, so its trades size at 1.0 (neutral). The meta-label uses only the
realized target already present in oos.parquet — no new future data.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tml.config import Config
from tml.data import store
from tml.pipeline.research import dataset, feature_cols


def meta_path(cfg: Config):
    return cfg.artifacts / "predictions" / "meta.parquet"


def _triggered(df: pl.DataFrame, enter: float) -> pl.DataFrame:
    """Rows where the primary hysteresis would OPEN a position."""
    return df.filter(pl.col("score").abs() > enter)


def run_meta(cfg: Config) -> int:
    from lightgbm import LGBMClassifier

    oos = pl.read_parquet(store.oos_path(cfg))
    ds = dataset(cfg)
    cols = feature_cols(ds)
    # join primary OOS scores to the feature matrix (meta-features = model
    # features + the primary score itself)
    df = oos.join(ds.select("ts", "pair", *cols), on=["ts", "pair"], how="inner")
    enter = cfg.backtest.enter_threshold
    trig = _triggered(df, enter).with_columns(
        # profitable iff the vol-scaled realized return went the signal's way
        meta_y=(pl.col("y_true") * pl.col("score").sign() > 0).cast(pl.Int8),
    )
    meta_feats = [*cols, "score"]
    fold_ids = sorted(trig["fold_id"].unique().to_list())

    parts: list[pl.DataFrame] = []
    for k in fold_ids:
        test = trig.filter(pl.col("fold_id") == k)
        train = trig.filter(pl.col("fold_id") < k)
        if train.height < 500 or test.is_empty():
            # fold 0 (or too little history): neutral sizing
            parts.append(test.select("ts", "pair").with_columns(p_profit=pl.lit(0.5, dtype=pl.Float64)))
            continue
        clf = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.meta.seed,
            n_jobs=cfg.model.lgbm.num_threads,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        clf.fit(train.select(meta_feats).to_numpy().astype(np.float32), train["meta_y"].to_numpy())
        p = clf.predict_proba(test.select(meta_feats).to_numpy().astype(np.float32))[:, 1]
        parts.append(test.select("ts", "pair").with_columns(p_profit=pl.Series(p, dtype=pl.Float64)))
        base_rate = float(train["meta_y"].to_numpy().mean())
        hit = float((test["meta_y"].to_numpy() == (p > 0.5)).mean())
        print(f"meta fold {k}: train={train.height} test={test.height} base_rate={base_rate:.3f} acc={hit:.3f}")

    meta = pl.concat(parts, how="vertical")
    # size multiplier: 0 below min_p, then linearly up to max_mult at p=1
    span = max(1e-9, 1.0 - cfg.meta.min_p)
    meta = meta.with_columns(
        size_mult=pl.when(pl.col("p_profit") < cfg.meta.min_p)
        .then(0.0)
        .otherwise(((pl.col("p_profit") - cfg.meta.min_p) / span * cfg.meta.max_mult).clip(0.0, cfg.meta.max_mult))
    )
    meta.write_parquet(meta_path(cfg))

    traded = meta.filter(pl.col("size_mult") > 0)
    mean_mult = float(traded["size_mult"].to_numpy().mean()) if traded.height else 0.0
    print(
        f"meta: {meta.height} triggered signals, {traded.height} pass the "
        f"p>={cfg.meta.min_p} filter (mean mult {mean_mult:.2f}) -> {meta_path(cfg)}"
    )
    return 0


def size_multipliers(cfg: Config) -> dict[tuple[str, int], float] | None:
    """(pair_key, ts_ns) -> size multiplier, or None if meta is disabled/missing."""
    if not cfg.meta.enabled:
        return None
    p = meta_path(cfg)
    if not p.exists():
        raise RuntimeError("meta.enabled but predictions/meta.parquet missing — run `tml meta`")
    m = pl.read_parquet(p).with_columns(ts_ns=pl.col("ts").dt.epoch(time_unit="ns"))
    return {(row["pair"], int(row["ts_ns"])): float(row["size_mult"]) for row in m.iter_rows(named=True)}
