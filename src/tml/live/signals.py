"""Live signal computation — the train/serve-skew killer.

The live tick calls the EXACT research pipeline (`build_features`, full
history, same config) and takes the latest completed bar's rows. This costs
~a minute per tick — trivially affordable at 1h cadence — and buys bit-level
parity with research features: the prefix-invariance tests guarantee that
features at ts are identical whether or not later data exists, and full-
history computation removes any EWM warm-start drift a tail-slice shortcut
would introduce.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from tml.config import Config
from tml.ensemble.stack import blend_horizons, horizon_scales, mean_ensemble
from tml.features.engine import build_features
from tml.live.persist import load_models


def latest_scores(cfg: Config) -> tuple[pl.DataFrame, datetime]:
    """-> ((pair, score, per-member scores) at the latest common bar, its ts).

    Pairs whose freshest bar lags the common latest timestamp are dropped
    for this tick (logged) — better no decision than a stale-feature one.

    Multi-horizon (S4): `score` is the SAME equal-weight horizon blend the
    research OOS trades, using the FROZEN train-time calibration persisted
    in meta.json — never statistics computed at tick time.
    """
    feats = build_features(cfg)
    latest_ts = cast(datetime, feats["ts"].max())
    rows = feats.filter(pl.col("ts") == latest_ts).sort("pair")
    missing = set(p.replace("/", "") for p in cfg.data.pairs) - set(rows["pair"].to_list())
    if missing:
        print(f"tick: pairs lagging at {latest_ts}: {sorted(missing)} — skipped this tick")

    models, meta = load_models(cfg)
    cols = meta["feature_cols"]
    current_cols = sorted(c for c in rows.columns if c not in ("ts", "pair", "close"))
    if current_cols != sorted(cols):
        raise RuntimeError(
            "live feature columns differ from the model's training schema — "
            f"missing={set(cols) - set(current_cols)} extra={set(current_cols) - set(cols)}; "
            "rerun `tml train-live`"
        )
    x = rows.select(cols).to_numpy().astype(np.float32)

    horizons = [int(h) for h in meta.get("horizons", [cfg.labels.horizon_bars])]
    base = horizons[0]
    multi = len(horizons) > 1
    member_scores: dict[str, np.ndarray] = {}
    h_scores: dict[int, np.ndarray] = {}
    for h in horizons:
        preds = {name: np.asarray(m.predict(x), dtype=np.float64) for name, m in models[h].items()}
        h_scores[h] = mean_ensemble(preds)
        suffix = "" if h == base else f"_h{h}"
        member_scores.update({f"{n}{suffix}": s for n, s in preds.items()})
    if multi:
        calib = {int(k): float(v) for k, v in meta["horizon_calib"].items()}
        scales = horizon_scales(calib, base)
    else:
        scales = {base: 1.0}
    ensemble = blend_horizons(h_scores, scales)

    out = rows.select("pair", "close").with_columns(
        score=pl.Series(ensemble),
        **{f"score_{n}": pl.Series(s) for n, s in member_scores.items()},
        **({f"score_h{h}": pl.Series(s) for h, s in h_scores.items()} if multi else {}),
    )
    return out, latest_ts
