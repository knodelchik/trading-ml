"""Research pipeline: features + labels -> purged walk-forward OOS predictions.

The backtest consumes exactly the OOS predictions written here, so the
retraining schedule in the backtest is identical to CV by construction.

`run_cv_frames` is pure (frames in, frame out) so tests can drive it with
synthetic data; `run_cv` adds the IO shell and the trials-ledger record.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl

from tml.config import Config
from tml.data import store
from tml.ensemble.stack import mean_ensemble, rank_average
from tml.validation import metrics
from tml.validation.splits import Fold, PurgedWalkForwardCV, write_folds
from tml.validation.trials import TrialsLedger

KEY_COLS = ("ts", "pair", "close", "y", "label_end_ts")


def make_adapter(name: str, cfg: Config, lgbm_params: dict | None = None):
    """Fresh adapter per fold; lazy imports keep optional deps optional."""
    if name == "lgbm":
        from tml.models.lgbm import LGBMAdapter

        lc = cfg.model.lgbm.model_copy(update=lgbm_params) if lgbm_params else cfg.model.lgbm
        return LGBMAdapter(lc)
    if name == "catboost":
        from tml.models.cat import CatBoostAdapter

        return CatBoostAdapter(cfg.model.catboost)
    if name == "tabpfn":
        from tml.models.tabpfn_member import TabPFNAdapter

        return TabPFNAdapter(cfg.model.tabpfn)
    raise ValueError(f"unknown model member {name}")


def tuned_lgbm_path(cfg: Config):
    return cfg.artifacts / "tuned" / "lgbm.json"


def load_tuned_lgbm(cfg: Config) -> dict | None:
    """Optuna-tuned LightGBM overrides, if `tml tune` has been run."""
    p = tuned_lgbm_path(cfg)
    if not p.exists():
        return None
    return json.loads(p.read_text())["params"]


def params_hash(cfg: Config, lgbm_params: dict | None) -> str:
    """Effective-parameter identity for the trials ledger (config_hash does
    not see tuned overrides, so they must hash separately)."""
    payload = {"lgbm_overrides": lgbm_params or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def dataset(cfg: Config) -> pl.DataFrame:
    from tml.universe import membership_filter

    feats = pl.read_parquet(store.features_path(cfg))
    labels = pl.read_parquet(store.labels_path(cfg))
    df = feats.join(labels, on=["ts", "pair"], how="inner", validate="1:1")
    df = membership_filter(cfg, df)  # point-in-time universe (no-op when off)
    return df.filter(pl.col("y").is_not_null()).sort("ts", "pair")


def feature_cols(df: pl.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c not in KEY_COLS)


def run_cv_frames(
    df: pl.DataFrame, cfg: Config, lgbm_params: dict | None = None
) -> tuple[pl.DataFrame, list[Fold]]:
    """Fit every member per fold, predict its test window, concatenate OOS.

    The tradeable `score` is the mean of member predictions (all regress the
    same vol-scaled target, so units are preserved for the backtest
    thresholds); per-member columns and a rank-average diagnostic are kept
    for the report.
    """
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
        x_tr = train.select(cols).to_numpy().astype(np.float32)
        y_tr = train["y"].to_numpy().astype(np.float32)
        x_te = test.select(cols).to_numpy().astype(np.float32)

        scores: dict[str, np.ndarray] = {}
        for name in cfg.model.members:
            adapter = make_adapter(name, cfg, lgbm_params)
            adapter.fit(x_tr, y_tr, cols)
            scores[name] = np.asarray(adapter.predict(x_te), dtype=np.float64)
            del adapter  # free per-fold model memory (torch models especially)

        ensemble = mean_ensemble(scores)
        extra = [pl.Series(f"score_{n}", s, dtype=pl.Float32) for n, s in scores.items()]
        if len(scores) > 1:
            extra.append(pl.Series("score_rankavg", rank_average(scores), dtype=pl.Float32))
        parts.append(
            test.select("ts", "pair", pl.col("y").alias("y_true"))
            .with_columns(extra)
            .with_columns(
                score=pl.Series(ensemble, dtype=pl.Float32),
                fold_id=pl.lit(fold.fold_id, dtype=pl.Int32),
            )
        )
        print(f"fold {fold.fold_id}: train={train.height} test={test.height} "
              f"[{fold.test_start:%Y-%m-%d} .. {fold.test_end:%Y-%m-%d}) members={list(scores)}")
    oos = pl.concat(parts, how="vertical")
    dupes = oos.height - oos.select("ts", "pair").unique().height
    if dupes:
        raise RuntimeError(f"OOS predictions contain {dupes} duplicate (ts, pair) rows — folds overlap")
    return oos, folds


def run_cv(cfg: Config) -> int:
    df = dataset(cfg)
    lgbm_params = load_tuned_lgbm(cfg)
    if lgbm_params:
        print(f"cv: using Optuna-tuned LightGBM overrides {lgbm_params}")
    oos, folds = run_cv_frames(df, cfg, lgbm_params)

    path = store.oos_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    oos.write_parquet(path)
    write_folds(folds, store.folds_path(cfg))

    scores = oos["score"].to_numpy()
    y = oos["y_true"].to_numpy()
    pooled_ic = metrics.spearman_ic(scores, y)

    # Pre-cost signal SR per period, recorded for the DSR trial history:
    # equal-weight portfolio of per-name positions sized by sign(score).
    # Stride-H non-overlapping subsample — MUST match the estimator used for
    # the deflation candidate in reporting, or the trial-history variance
    # measures estimator mismatch instead of selection breadth.
    port = (
        oos.with_columns(r=pl.col("y_true") * pl.col("score").sign())
        .group_by("ts")
        .agg(pl.col("r").mean())
        .sort("ts")["r"]
        .gather_every(cfg.labels.horizon_bars)
        .to_numpy()
    )
    sr_pp = float(np.nanmean(port) / np.nanstd(port, ddof=1)) if len(port) > 2 else None

    member_ics = {
        c.removeprefix("score_"): metrics.spearman_ic(oos[c].to_numpy(), y)
        for c in oos.columns
        if c.startswith("score_")
    }
    TrialsLedger(store.trials_path(cfg)).append(
        {
            "stage": "cv",
            "config_hash": cfg.config_hash(),
            "params_hash": params_hash(cfg, lgbm_params),
            "members": list(cfg.model.members),
            "n_folds": len(folds),
            "n_oos_rows": oos.height,
            "pooled_ic": None if np.isnan(pooled_ic) else pooled_ic,
            "member_ics": {k: (None if np.isnan(v) else v) for k, v in member_ics.items()},
            "sr_per_period": sr_pp,
        }
    )
    print(f"cv: {oos.height} OOS predictions across {len(folds)} folds -> {path}")
    print(f"cv: pooled Spearman IC = {pooled_ic:+.4f} (full stats: tml report)")
    return 0
