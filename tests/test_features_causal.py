"""The load-bearing causality test: prefix invariance.

If any feature uses information from after its own row, truncating the
input series (or perturbing data after the truncation point) changes the
feature values before that point — and these tests fail. This covers
centered windows, negative shifts, and full-history normalization in one
mechanical property.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from tests.conftest import make_config, make_raw
from tml.features.cross_sectional import add_cs_ranks
from tml.features.engine import build_features, pair_features

FC = make_config("/tmp").features  # only .features is used; root is irrelevant here


def test_prefix_invariance_truncation():
    raw = make_raw("AAA/USDT", n=1200)
    full = pair_features(raw, FC)
    cut = 800
    truncated = pair_features(raw.slice(0, cut), FC)
    assert_frame_equal(full.slice(0, truncated.height), truncated)


@settings(max_examples=10, deadline=None)
@given(
    cut=st.integers(min_value=400, max_value=900),
    scale=st.floats(min_value=1.5, max_value=10.0),
)
def test_prefix_invariance_perturbation(cut: int, scale: float):
    """Perturbing raw data AFTER the cut must not change features AT or before it."""
    raw = make_raw("AAA/USDT", n=1000)
    before = pair_features(raw, FC)

    perturbed = raw.with_columns(
        [
            pl.when(pl.arange(0, raw.height) >= cut)
            .then(pl.col(c) * scale)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("open", "high", "low", "close", "volume")
        ]
    )
    after = pair_features(perturbed, FC)

    n_prefix = cut - FC.max_lookback  # feature rows are shifted by the warmup drop
    assert n_prefix > 50
    assert_frame_equal(before.slice(0, n_prefix), after.slice(0, n_prefix))


def test_cs_ranks_use_only_same_timestamp():
    frames = [pair_features(make_raw(p, n=600), FC) for p in ("AAA/USDT", "BBB/USDT", "CCC/USDT")]
    df = pl.concat(frames).sort("ts", "pair")
    ranked = add_cs_ranks(df, ["mom_24"])

    ts_split = df["ts"].unique().sort()[300]
    perturbed = df.with_columns(
        pl.when(pl.col("ts") > ts_split).then(pl.col("mom_24") * -3.0).otherwise(pl.col("mom_24")).alias("mom_24")
    )
    ranked_p = add_cs_ranks(perturbed, ["mom_24"])

    a = ranked.filter(pl.col("ts") <= ts_split)
    b = ranked_p.filter(pl.col("ts") <= ts_split)
    assert_frame_equal(a, b)


def test_no_nulls_after_warmup(synth_store):
    df = build_features(synth_store)
    assert int(df.null_count().sum_horizontal().item()) == 0
    assert df.height > 0
    # float32 discipline for model inputs
    for c in df.columns:
        if c not in ("ts", "pair", "close"):
            assert df.schema[c] == pl.Float32, f"{c} is {df.schema[c]}"


def test_features_deterministic():
    raw = make_raw("AAA/USDT", n=800)
    f1 = pair_features(raw, FC)
    f2 = pair_features(raw.clone(), FC)
    assert_frame_equal(f1, f2)
    assert not np.isnan(f1.select(pl.exclude("ts", "pair")).to_numpy()).any()
