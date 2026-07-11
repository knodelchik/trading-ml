"""Structural leakage tests over the WHOLE pipeline (traps 5 and 6).

test_null_labels: replace targets with pure noise and run the full CV.
If out-of-sample IC is distinguishable from zero, information is flowing
from test targets into training somewhere — a structural leak no amount
of modeling skill explains.

test_future_perturbation_isolation: corrupting data inside the LAST fold's
test window must not change predictions of earlier folds (retraining
schedule causality).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from tml.features.engine import build_features
from tml.labeling.labels import build_labels
from tml.pipeline.research import feature_cols, run_cv_frames
from tml.validation.metrics import spearman_ic


def _dataset(cfg) -> pl.DataFrame:
    feats = build_features(cfg)
    labels = build_labels(cfg)
    return (
        feats.join(labels, on=["ts", "pair"], how="inner")
        .filter(pl.col("y").is_not_null())
        .sort("ts", "pair")
    )


def test_null_labels(synth_store):
    df = _dataset(synth_store)
    rng = np.random.default_rng(123)
    df = df.with_columns(y=pl.Series(rng.normal(0, 1, df.height), dtype=pl.Float32))

    oos, _ = run_cv_frames(df, synth_store)
    ic = spearman_ic(oos["score"].to_numpy(), oos["y_true"].to_numpy())
    bound = 2 / np.sqrt(oos.height)
    assert abs(ic) < bound, (
        f"OOS |IC|={abs(ic):.4f} >= {bound:.4f} on pure-noise labels — structural leakage"
    )
    # fold-localized leaks can pool away in the global IC — check each fold
    for (fold_id,), fold_df in oos.group_by("fold_id"):
        fic = spearman_ic(fold_df["score"].to_numpy(), fold_df["y_true"].to_numpy())
        fbound = 3 / np.sqrt(fold_df.height)
        assert abs(fic) < fbound, (
            f"fold {fold_id}: |IC|={abs(fic):.4f} >= {fbound:.4f} on noise — localized leakage"
        )


def test_future_perturbation_isolation(synth_store):
    df = _dataset(synth_store)
    oos_before, folds = run_cv_frames(df, synth_store)
    last = folds[-1]

    cols = feature_cols(df)
    in_last_test = (pl.col("ts") >= last.test_start) & (pl.col("ts") < last.test_end)
    df_perturbed = df.with_columns(
        [pl.when(in_last_test).then(pl.col(c) * -5.0).otherwise(pl.col(c)).alias(c) for c in cols]
    )
    oos_after, _ = run_cv_frames(df_perturbed, synth_store)

    early_before = oos_before.filter(pl.col("fold_id") < last.fold_id)
    early_after = oos_after.filter(pl.col("fold_id") < last.fold_id)
    assert_frame_equal(early_before, early_after)
