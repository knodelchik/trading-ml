"""Purged walk-forward CV invariants (leakage trap 4)."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from tests.conftest import make_raw
from tml.labeling.labels import pair_labels
from tml.validation.splits import PurgedWalkForwardCV


def _labeled(cfg):
    raw = make_raw("AAA/USDT", n=2000)
    return pair_labels(raw, cfg.labels.horizon_bars, cfg.labels.vol_span, cfg.data.bar_ms).filter(
        pl.col("y").is_not_null()
    )


def test_purge_gap_invariant(synth_cfg):
    df = _labeled(synth_cfg)
    cv = PurgedWalkForwardCV(synth_cfg.cv, synth_cfg.data.bar_ms)
    folds = cv.split(df["ts"].unique())
    assert len(folds) == 6  # (2000 - 24 warmup - 6 tail nulls - 600) // 200

    bar = timedelta(milliseconds=synth_cfg.data.bar_ms)
    for fold in folds:
        train = df.filter(fold.train_expr())
        test = df.filter(fold.test_expr())
        assert train.height > 0 and test.height > 0
        # no train label may end at/after (test_start - embargo)
        max_end = train["label_end_ts"].max()
        cutoff = fold.test_start - synth_cfg.cv.embargo_bars * bar
        assert max_end < cutoff
        # train strictly precedes test in observation time as well
        assert train["ts"].max() < fold.test_start


def test_no_ts_in_two_test_folds(synth_cfg):
    df = _labeled(synth_cfg)
    cv = PurgedWalkForwardCV(synth_cfg.cv, synth_cfg.data.bar_ms)
    folds = cv.split(df["ts"].unique())
    seen: set = set()
    for fold in folds:
        ts_vals = set(df.filter(fold.test_expr())["ts"].to_list())
        assert not (seen & ts_vals), f"fold {fold.fold_id} test overlaps a previous fold"
        seen |= ts_vals


def test_split_determinism(synth_cfg):
    df = _labeled(synth_cfg)
    cv = PurgedWalkForwardCV(synth_cfg.cv, synth_cfg.data.bar_ms)
    assert cv.split(df["ts"].unique()) == cv.split(df["ts"].unique())
