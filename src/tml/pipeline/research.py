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
from tml.ensemble.stack import blend_horizons, horizon_scales, mean_ensemble, rank_average
from tml.labeling.labels import horizon_cols
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
    # y_h*/label_end_ts_h* are extra-horizon LABEL columns — never features
    return sorted(
        c
        for c in df.columns
        if c not in KEY_COLS and not c.startswith(("y_h", "label_end_ts_h"))
    )


def run_cv_frames(
    df: pl.DataFrame, cfg: Config, lgbm_params: dict | None = None
) -> tuple[pl.DataFrame, list[Fold]]:
    """Fit every member per fold — and per label horizon when the S4
    multi-horizon blend is on — predict its test window, concatenate OOS.

    Per horizon, the ensemble score is the mean of member predictions (all
    regress the same vol-scaled target, so units are preserved for the
    backtest thresholds); per-member columns and a rank-average diagnostic
    are kept for the report. With several horizons the tradeable `score` is
    their equal-weight blend (`blend_horizons`), each non-base horizon
    rescaled by its FROZEN train-window std ratio; the per-fold scales are
    stamped into `blend_scale_h*` columns so the blend is auditable from
    the artifact. A single horizon takes exactly the legacy path (bit-
    identical scores, unchanged schema).
    """
    horizons = cfg.labels.all_horizons
    base = horizons[0]
    multi = len(horizons) > 1
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
        test = df.filter(fold.test_expr())
        if test.is_empty():
            raise RuntimeError(f"fold {fold.fold_id} has empty test slice")
        x_te = test.select(cols).to_numpy().astype(np.float32)

        h_scores: dict[int, np.ndarray] = {}
        train_stds: dict[int, float] = {}
        n_train: dict[int, int] = {}
        extra: list[pl.Series] = []
        for h in horizons:
            ycol, endcol = horizon_cols(h, base)
            train = df.filter(fold.train_expr(endcol) & pl.col(ycol).is_not_null())
            if train.is_empty():
                raise RuntimeError(f"fold {fold.fold_id} has empty train slice (H={h})")
            x_tr = train.select(cols).to_numpy().astype(np.float32)
            y_tr = train[ycol].to_numpy().astype(np.float32)

            scores: dict[str, np.ndarray] = {}
            train_preds: list[np.ndarray] = []
            for name in cfg.model.members:
                adapter = make_adapter(name, cfg, lgbm_params)
                adapter.fit(x_tr, y_tr, cols)
                scores[name] = np.asarray(adapter.predict(x_te), dtype=np.float64)
                if multi:  # train-window dispersion -> frozen blend scale
                    train_preds.append(np.asarray(adapter.predict(x_tr), dtype=np.float64))
                del adapter  # free per-fold model memory (torch models especially)

            h_scores[h] = mean_ensemble(scores)
            n_train[h] = train.height
            if multi:
                train_stds[h] = float(np.std(np.mean(np.column_stack(train_preds), axis=1)))
            suffix = "" if h == base else f"_h{h}"
            extra += [pl.Series(f"score_{n}{suffix}", s, dtype=pl.Float32) for n, s in scores.items()]
            if h == base and len(scores) > 1:
                extra.append(pl.Series("score_rankavg", rank_average(scores), dtype=pl.Float32))

        scales = horizon_scales(train_stds, base) if multi else {base: 1.0}
        ensemble = blend_horizons(h_scores, scales)
        if multi:
            for h in horizons:
                extra.append(pl.Series(f"score_h{h}", h_scores[h], dtype=pl.Float32))
                extra.append(
                    pl.Series(f"blend_scale_h{h}", np.full(test.height, scales[h]), dtype=pl.Float32)
                )
        parts.append(
            test.select("ts", "pair", pl.col("y").alias("y_true"))
            .with_columns(extra)
            .with_columns(
                score=pl.Series(ensemble, dtype=pl.Float32),
                fold_id=pl.lit(fold.fold_id, dtype=pl.Int32),
            )
        )
        train_desc = (
            "/".join(f"H{h}={n_train[h]}" for h in horizons) if multi else str(n_train[base])
        )
        print(f"fold {fold.fold_id}: train={train_desc} test={test.height} "
              f"[{fold.test_start:%Y-%m-%d} .. {fold.test_end:%Y-%m-%d}) members={list(cfg.model.members)}")
        if multi:
            print(f"fold {fold.fold_id}: blend scales "
                  + ", ".join(f"H{h}={scales[h]:.4f}" for h in horizons))
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

    # mean |score| drives the hysteresis trigger rate; the multi-horizon
    # blend shifts it (imperfectly correlated horizons shrink |score|), so
    # record it — blended and per horizon — for before/after inspection.
    horizons = cfg.labels.all_horizons
    mean_abs_score = float(np.mean(np.abs(scores)))
    mean_abs_by_h = (
        {str(h): float(np.mean(np.abs(oos[f"score_h{h}"].to_numpy()))) for h in horizons}
        if len(horizons) > 1
        else None
    )

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
            "mean_abs_score": mean_abs_score,
            "mean_abs_score_by_horizon": mean_abs_by_h,
        }
    )
    print(f"cv: {oos.height} OOS predictions across {len(folds)} folds -> {path}")
    print(f"cv: pooled Spearman IC = {pooled_ic:+.4f} (full stats: tml report)")
    if mean_abs_by_h is None:
        print(f"cv: mean |score| = {mean_abs_score:.4f}")
    else:
        base_abs = mean_abs_by_h[str(horizons[0])]
        shift = (mean_abs_score / base_abs - 1.0) * 100 if base_abs else float("nan")
        per_h = ", ".join(f"H{h}={mean_abs_by_h[str(h)]:.4f}" for h in horizons)
        print(f"cv: mean |score| = {mean_abs_score:.4f} (blend; {per_h}; "
              f"{shift:+.1f}% vs H{horizons[0]} alone)")
    return 0
