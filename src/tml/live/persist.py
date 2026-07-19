"""Model persistence for live trading.

`tml train-live` fits the traded ensemble members on ALL available history
(with tuned overrides, same adapters as research) and freezes them to disk.
The live tick loads frozen models — it never trains. Retraining is an
explicit act, mirroring the walk-forward cadence (cv.step_bars ~ 90 days),
and the tick refuses to trade on a model older than `live.model_max_age_days`.

Metadata pins the feature-column list and config hash so a drifted feature
pipeline cannot silently feed a stale model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import polars as pl

from tml.config import Config
from tml.labeling.labels import horizon_cols
from tml.pipeline.research import dataset, feature_cols, load_tuned_lgbm, make_adapter, params_hash


def live_dir(cfg: Config) -> Path:
    return cfg.artifacts / "live"


def models_dir(cfg: Config) -> Path:
    return live_dir(cfg) / "models"


def meta_path(cfg: Config) -> Path:
    return models_dir(cfg) / "meta.json"


def model_file(name: str, horizon: int, base: int) -> str:
    """Base-horizon members keep the legacy `{name}.joblib` filenames."""
    return f"{name}.joblib" if horizon == base else f"{name}_h{horizon}.joblib"


def run_train_live(cfg: Config) -> int:
    df = dataset(cfg)
    cols = feature_cols(df)
    lgbm_params = load_tuned_lgbm(cfg)
    horizons = cfg.labels.all_horizons
    base = horizons[0]
    multi = len(horizons) > 1

    mdir = models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    train_stds: dict[int, float] = {}
    n_base_rows = 0
    for h in horizons:
        ycol, _ = horizon_cols(h, base)
        sub = df.filter(pl.col(ycol).is_not_null())
        x = sub.select(cols).to_numpy().astype(np.float32)
        y = sub[ycol].to_numpy().astype(np.float32)
        if h == base:
            n_base_rows = len(x)
        train_preds: list[np.ndarray] = []
        for name in cfg.model.members:
            adapter = make_adapter(name, cfg, lgbm_params)
            adapter.fit(x, y, cols)
            fname = model_file(name, h, base)
            joblib.dump(adapter, mdir / fname)
            if multi:  # train-window dispersion -> frozen blend calibration
                train_preds.append(np.asarray(adapter.predict(x), dtype=np.float64))
            print(f"train-live: {name} (H={h}) fitted on {len(x)} rows -> {mdir / fname}")
        if multi:
            train_stds[h] = float(np.std(np.mean(np.column_stack(train_preds), axis=1)))

    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "members": list(cfg.model.members),
        "horizons": list(horizons),
        "feature_cols": cols,
        "model_hash": cfg.model_hash(),
        "params_hash": params_hash(cfg, lgbm_params),
        "lgbm_overrides": lgbm_params or {},
        "train_rows": n_base_rows,
        "train_ts_max": cast(datetime, df["ts"].max()).isoformat(),
    }
    if multi:
        # FROZEN blend calibration: live blending uses these constants only,
        # never runtime stats (parity with the per-fold research blend)
        meta["horizon_calib"] = {str(h): train_stds[h] for h in horizons}
    meta_path(cfg).write_text(json.dumps(meta, indent=2))
    print(f"train-live: meta -> {meta_path(cfg)}")
    return 0


def load_models(cfg: Config) -> tuple[dict, dict]:
    """-> ({horizon: {member: adapter}}, meta). Raises with guidance if stale."""
    mp = meta_path(cfg)
    if not mp.exists():
        raise RuntimeError("no live models found — run `tml train-live` first")
    meta = json.loads(mp.read_text())

    age_days = (
        datetime.now(UTC) - datetime.fromisoformat(meta["trained_at"])
    ).total_seconds() / 86400
    if age_days > cfg.live.model_max_age_days:
        raise RuntimeError(
            f"live models are {age_days:.0f} days old (> {cfg.live.model_max_age_days}) — "
            "rerun `tml train-live` (retraining cadence mirrors the walk-forward step)"
        )
    if meta.get("model_hash") != cfg.model_hash():
        raise RuntimeError(
            "model-relevant config (data/features/labels/model) changed since "
            f"models were trained ({meta.get('model_hash')} != {cfg.model_hash()}) — "
            "rerun `tml train-live`"
        )
    # model_hash covers the labels section, so meta horizons == cfg horizons;
    # .get is a fallback for pre-S4 metas only
    horizons = [int(h) for h in meta.get("horizons", [cfg.labels.horizon_bars])]
    models = {
        h: {
            name: joblib.load(models_dir(cfg) / model_file(name, h, horizons[0]))
            for name in meta["members"]
        }
        for h in horizons
    }
    return models, meta
