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

from tml.config import Config
from tml.pipeline.research import dataset, feature_cols, load_tuned_lgbm, make_adapter, params_hash


def live_dir(cfg: Config) -> Path:
    return cfg.artifacts / "live"


def models_dir(cfg: Config) -> Path:
    return live_dir(cfg) / "models"


def meta_path(cfg: Config) -> Path:
    return models_dir(cfg) / "meta.json"


def run_train_live(cfg: Config) -> int:
    df = dataset(cfg)
    cols = feature_cols(df)
    lgbm_params = load_tuned_lgbm(cfg)
    x = df.select(cols).to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.float32)

    mdir = models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    for name in cfg.model.members:
        adapter = make_adapter(name, cfg, lgbm_params)
        adapter.fit(x, y, cols)
        joblib.dump(adapter, mdir / f"{name}.joblib")
        print(f"train-live: {name} fitted on {len(x)} rows -> {mdir / f'{name}.joblib'}")

    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "members": list(cfg.model.members),
        "feature_cols": cols,
        "model_hash": cfg.model_hash(),
        "params_hash": params_hash(cfg, lgbm_params),
        "lgbm_overrides": lgbm_params or {},
        "train_rows": int(len(x)),
        "train_ts_max": cast(datetime, df["ts"].max()).isoformat(),
    }
    meta_path(cfg).write_text(json.dumps(meta, indent=2))
    print(f"train-live: meta -> {meta_path(cfg)}")
    return 0


def load_models(cfg: Config) -> tuple[dict, dict]:
    """-> ({member: adapter}, meta). Raises with clear guidance if stale."""
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
    models = {name: joblib.load(models_dir(cfg) / f"{name}.joblib") for name in meta["members"]}
    return models, meta
