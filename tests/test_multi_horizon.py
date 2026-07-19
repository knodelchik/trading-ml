"""S4 multi-horizon blend: two-horizon label correctness, per-horizon
purging, frozen-constant blend math, and flag-off byte-identity with the
single-horizon path."""

from __future__ import annotations

import hashlib
import io
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from tests.conftest import make_config, make_raw
from tml.data import store
from tml.ensemble.stack import blend_horizons, horizon_scales
from tml.features.engine import build_features
from tml.labeling.labels import build_labels, horizon_cols, pair_labels
from tml.pipeline.research import feature_cols, run_cv_frames
from tml.validation.splits import PurgedWalkForwardCV

H1, H2 = 6, 12  # base and second horizon (mirrors 24 -> 48 in production)
VOL_SPAN = 24  # conftest labels.vol_span
PAIRS3 = ("AAA/USDT", "BBB/USDT", "CCC/USDT")


def _populate(cfg, pairs=PAIRS3, n=1200) -> None:
    for pair in pairs:
        df = make_raw(pair, n=n)
        for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
            "year", maintain_order=True
        ):
            store.write_raw_year(cfg, pair, int(year), part.drop("year"))


def _multi_cfg(root, pairs=PAIRS3):
    return make_config(
        root,
        pairs=pairs,
        overrides={
            "labels": {"horizons": [H1, H2]},
            "cv": {"purge_bars": H2, "embargo_bars": H2},
        },
    )


def _dataset(cfg) -> pl.DataFrame:
    feats = build_features(cfg)
    labels = build_labels(cfg)
    return (
        feats.join(labels, on=["ts", "pair"], how="inner")
        .filter(pl.col("y").is_not_null())
        .sort("ts", "pair")
    )


def _sha(df: pl.DataFrame) -> str:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# --- config invariants -------------------------------------------------------


def test_config_rejects_underpurged_second_horizon(tmp_path):
    with pytest.raises(ValueError, match="purge_bars"):
        make_config(tmp_path, overrides={"labels": {"horizons": [H1, H2]}})  # purge stays 6 < 12


def test_config_horizons_must_start_with_base(tmp_path):
    with pytest.raises(ValueError, match="horizons\\[0\\]"):
        make_config(
            tmp_path,
            overrides={"labels": {"horizons": [H2, H1]}, "cv": {"purge_bars": H2}},
        )


def test_config_horizons_strictly_increasing(tmp_path):
    with pytest.raises(ValueError, match="strictly increasing"):
        make_config(
            tmp_path,
            overrides={"labels": {"horizons": [H1, H1]}, "cv": {"purge_bars": H2}},
        )


def test_all_horizons_default_is_single(tmp_path):
    assert make_config(tmp_path).labels.all_horizons == (H1,)
    assert _multi_cfg(tmp_path).labels.all_horizons == (H1, H2)


# --- labels ------------------------------------------------------------------


def test_second_horizon_label_columns_and_end_ts_math(tmp_path):
    cfg = _multi_cfg(tmp_path, pairs=("AAA/USDT",))
    _populate(cfg, pairs=("AAA/USDT",), n=400)
    labels = build_labels(cfg)
    assert labels.columns == ["ts", "pair", "y", "label_end_ts", "y_h12", "label_end_ts_h12"]

    # gap-free series: end_ts is exactly H bars ahead, per horizon
    d1 = (labels["label_end_ts"] - labels["ts"]).drop_nulls().unique()
    d2 = (labels["label_end_ts_h12"] - labels["ts"]).drop_nulls().unique()
    assert d1.len() == 1 and d1[0] == timedelta(hours=H1)
    assert d2.len() == 1 and d2[0] == timedelta(hours=H2)
    assert labels["label_end_ts_h12"].null_count() == H2  # last H2 rows have no future bar
    assert labels["y_h12"].null_count() == VOL_SPAN + H2  # vol warmup + trailing

    # the extra columns are the exact single-horizon primitive at H2
    ref = pair_labels(make_raw("AAA/USDT", n=400), H2, VOL_SPAN, cfg.data.bar_ms)
    assert_series_equal(labels["y_h12"], ref["y"], check_names=False)
    assert_series_equal(labels["label_end_ts_h12"], ref["label_end_ts"], check_names=False)


def test_horizon_cols_base_keeps_legacy_names():
    assert horizon_cols(H1, H1) == ("y", "label_end_ts")
    assert horizon_cols(H2, H1) == ("y_h12", "label_end_ts_h12")


# --- purge -------------------------------------------------------------------


def test_purge_respects_second_horizon(tmp_path):
    cfg = _multi_cfg(tmp_path)
    raw = make_raw("AAA/USDT", n=2000)
    lab = pair_labels(raw, H1, VOL_SPAN, cfg.data.bar_ms).join(
        pair_labels(raw, H2, VOL_SPAN, cfg.data.bar_ms).rename(
            {"y": "y_h12", "label_end_ts": "label_end_ts_h12"}
        ),
        on=["ts", "pair"],
        validate="1:1",
    )
    df = lab.filter(pl.col("y").is_not_null())
    folds = PurgedWalkForwardCV(cfg.cv, cfg.data.bar_ms).split(df["ts"].unique())
    assert folds
    for fold in folds:
        # the H2 training slice may not contain any label ending at/after the cutoff
        train_h2 = df.filter(fold.train_expr("label_end_ts_h12") & pl.col("y_h12").is_not_null())
        assert train_h2.height > 0
        assert train_h2["label_end_ts_h12"].max() < fold.purge_embargo_cutoff
        # purging H2 rows on the BASE end column would under-purge — the rows
        # the per-horizon filter removes are exactly the leaky ones
        under = df.filter(fold.train_expr() & pl.col("y_h12").is_not_null())
        leaky = under.filter(pl.col("label_end_ts_h12") >= fold.purge_embargo_cutoff)
        assert leaky.height > 0


# --- blend math --------------------------------------------------------------


def test_blend_math_with_frozen_constants():
    scales = horizon_scales({H1: 2.0, H2: 4.0}, base=H1)
    assert scales == {H1: 1.0, H2: 0.5}  # base is exactly 1.0

    a = np.array([1.0, -2.0, 0.5, 0.0])
    b = np.array([2.0, 2.0, -1.0, 3.0])
    out = blend_horizons({H1: a, H2: b}, scales)
    np.testing.assert_allclose(out, 0.5 * a + 0.5 * 0.5 * b)

    # single horizon: bit-identical passthrough (flag-off path)
    assert blend_horizons({H1: a}, {H1: 1.0}) is a
    # degenerate member std must not divide by zero
    assert np.isfinite(horizon_scales({H1: 2.0, H2: 0.0}, base=H1)[H2])


# --- pipeline ----------------------------------------------------------------


def test_flag_off_byte_identical_scores(tmp_path):
    cfg_default = make_config(tmp_path, pairs=PAIRS3)
    _populate(cfg_default)
    # horizons=(6,) is the explicit spelling of the default single-horizon path
    cfg_explicit = make_config(tmp_path, pairs=PAIRS3, overrides={"labels": {"horizons": [H1]}})

    oos_a, _ = run_cv_frames(_dataset(cfg_default), cfg_default)
    oos_b, _ = run_cv_frames(_dataset(cfg_explicit), cfg_explicit)
    # legacy schema: no horizon columns leak into the flag-off artifact
    assert oos_a.columns == ["ts", "pair", "y_true", "score_lgbm", "score", "fold_id"]
    assert_frame_equal(oos_a, oos_b)
    assert _sha(oos_a) == _sha(oos_b)


def test_multi_horizon_blend_end_to_end(tmp_path):
    cfg = _multi_cfg(tmp_path)
    _populate(cfg)
    df = _dataset(cfg)

    # extra-horizon label columns must never enter the feature matrix
    cols = feature_cols(df)
    assert "y_h12" not in cols and "label_end_ts_h12" not in cols

    oos, _ = run_cv_frames(df, cfg)
    for c in ("score_h6", "score_h12", "blend_scale_h6", "blend_scale_h12",
              "score_lgbm", "score_lgbm_h12"):
        assert c in oos.columns, c
    assert (oos["blend_scale_h6"] == 1.0).all()

    # the traded score is exactly the frozen-constant blend, auditable
    # from the artifact's own columns
    recon = (
        0.5 * oos["score_h6"].to_numpy() * oos["blend_scale_h6"].to_numpy()
        + 0.5 * oos["score_h12"].to_numpy() * oos["blend_scale_h12"].to_numpy()
    )
    np.testing.assert_allclose(oos["score"].to_numpy(), recon, rtol=1e-5, atol=1e-6)
    # and the blend actually differs from the base horizon alone
    assert not np.allclose(oos["score"].to_numpy(), oos["score_h6"].to_numpy())

    # base-horizon member is untouched by the second horizon's presence:
    # same folds (same purge/embargo), same features, same training rows
    cfg_single = make_config(
        tmp_path, pairs=PAIRS3, overrides={"cv": {"purge_bars": H2, "embargo_bars": H2}}
    )
    oos_single, _ = run_cv_frames(_dataset(cfg_single), cfg_single)
    assert_series_equal(oos["score_lgbm"], oos_single["score_lgbm"])
    assert_series_equal(oos["score_h6"], oos_single["score"], check_names=False)
