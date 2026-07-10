"""Two identical runs must produce bit-identical OOS predictions."""

from __future__ import annotations

import hashlib

import polars as pl
from polars.testing import assert_frame_equal

from tml.features.engine import build_features
from tml.labeling.labels import build_labels
from tml.pipeline.research import run_cv_frames


def _oos(cfg) -> pl.DataFrame:
    feats = build_features(cfg)
    labels = build_labels(cfg)
    df = (
        feats.join(labels, on=["ts", "pair"], how="inner")
        .filter(pl.col("y").is_not_null())
        .sort("ts", "pair")
    )
    oos, _ = run_cv_frames(df, cfg)
    return oos


def _sha(df: pl.DataFrame) -> str:
    import io

    buf = io.BytesIO()
    df.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def test_full_cv_deterministic(synth_store):
    a = _oos(synth_store)
    b = _oos(synth_store)
    assert_frame_equal(a, b)
    assert _sha(a) == _sha(b)
