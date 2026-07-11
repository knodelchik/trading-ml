"""Optuna tuning with nested purged validation — no outer-test leakage.

Tuning runs ONLY on the first outer training window (data strictly before
the first CV test fold), split internally by the same purged walk-forward
logic. The chosen parameters are then frozen for ALL outer folds, so outer
OOS estimates never see data that influenced the parameter choice.

Every trial is appended to the trials ledger with stage="optuna" for audit;
inner-split objectives are not comparable to outer SR estimates and do NOT
enter any DSR family (nested tuning inside the fold protocol does not add
outer trials — the one outer evaluation of the tuned config does).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from tml.config import Config, CVConfig
from tml.data import store
from tml.validation.metrics import spearman_ic
from tml.validation.splits import PurgedWalkForwardCV
from tml.validation.trials import TrialsLedger


def _inner_folds(cfg: Config, ts_index: pl.Series):
    window = ts_index.slice(0, cfg.cv.train_min_bars)
    n = len(window)
    span = int(n * (1 - cfg.optuna.inner_train_frac) / cfg.optuna.inner_splits)
    inner_cv = CVConfig(
        train_min_bars=int(n * cfg.optuna.inner_train_frac),
        test_span_bars=span,
        step_bars=span,
        purge_bars=cfg.cv.purge_bars,
        embargo_bars=cfg.cv.embargo_bars,
    )
    folds = PurgedWalkForwardCV(inner_cv, cfg.data.bar_ms).split(window)
    if len(folds) < cfg.optuna.inner_splits:
        raise RuntimeError(f"only {len(folds)} inner folds fit the first training window")
    return folds


def run_tune(cfg: Config) -> int:
    import optuna

    from tml.models.lgbm import LGBMAdapter
    from tml.pipeline.research import dataset, feature_cols

    df = dataset(cfg)
    cols = feature_cols(df)
    ts_index = df["ts"].unique().sort()
    folds = _inner_folds(cfg, ts_index)
    # restrict all data to the first outer training window
    window_end = ts_index[cfg.cv.train_min_bars - 1]
    df = df.filter(pl.col("ts") <= window_end)
    ledger = TrialsLedger(store.trials_path(cfg))

    # pre-materialize inner splits once (they do not depend on trial params)
    splits = []
    for fold in folds:
        train = df.filter(fold.train_expr())
        test = df.filter(fold.test_expr())
        splits.append(
            (
                train.select(cols).to_numpy().astype(np.float32),
                train["y"].to_numpy().astype(np.float32),
                test.select(cols).to_numpy().astype(np.float32),
                test["y"].to_numpy(),
            )
        )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        }
        ics = []
        for x_tr, y_tr, x_te, y_te in splits:
            adapter = LGBMAdapter(cfg.model.lgbm.model_copy(update=params))
            adapter.fit(x_tr, y_tr, cols)
            ics.append(spearman_ic(adapter.predict(x_te), y_te))
        mean_ic = float(np.nanmean(ics))
        ledger.append(
            {
                "stage": "optuna",
                "config_hash": cfg.config_hash(),
                "trial": trial.number,
                "params": params,
                "inner_mean_ic": None if np.isnan(mean_ic) else mean_ic,
                "sr_per_period": None,  # inner objective; never enters a DSR family
            }
        )
        return mean_ic

    sampler = optuna.samplers.TPESampler(seed=cfg.model.lgbm.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=False)

    out = cfg.artifacts / "tuned" / "lgbm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "params": study.best_params,
                "inner_mean_ic": study.best_value,
                "n_trials": cfg.optuna.n_trials,
                "tuned_on": f"first outer train window (<= {window_end.isoformat()})",
            },
            indent=2,
        )
    )
    print(f"tune: best inner IC {study.best_value:+.4f} with {study.best_params} -> {out}")
    return 0
