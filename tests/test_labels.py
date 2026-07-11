"""Label alignment: the engineered spike must appear in y exactly H bars early."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from tests.conftest import make_raw
from tml.labeling.labels import pair_labels

H = 6
VOL_SPAN = 24
BAR_MS = 3_600_000


def test_spike_visible_exactly_in_forward_window():
    raw = make_raw("AAA/USDT", n=500, seed=7)
    k = 300
    spiked = raw.with_columns(
        pl.when(pl.arange(0, raw.height) >= k)
        .then(pl.col("close") * 3.0)  # permanent +200% level shift at bar k
        .otherwise(pl.col("close"))
        .alias("close")
    )
    labels = pair_labels(spiked, H, VOL_SPAN, BAR_MS)
    y = labels["y"].to_numpy()
    # rows whose forward window (t, t+H] contains the jump at k: t in [k-H, k-1]
    spike_window = y[k - H : k]
    outside = np.concatenate([y[VOL_SPAN : k - H], y[k + VOL_SPAN : -H]])  # skip post-jump vol warmup
    assert np.nanmin(spike_window) > np.nanmax(outside), (
        "the spike must dominate y exactly for the H rows preceding it and nowhere else"
    )
    # one bar before the window the spike is invisible
    assert abs(y[k - H - 1]) < np.nanmin(spike_window) / 5


def test_label_end_ts_offset():
    raw = make_raw("AAA/USDT", n=100)
    labels = pair_labels(raw, H, VOL_SPAN, BAR_MS)
    # gap-free series: the H-th bar ahead closes exactly H hours later;
    # the last H rows have no future bar -> null label_end_ts
    delta = (labels["label_end_ts"] - labels["ts"]).drop_nulls().unique()
    assert delta.len() == 1
    assert delta[0] == timedelta(hours=H)
    assert labels["label_end_ts"].null_count() == H


def test_label_end_ts_spans_gaps():
    raw = make_raw("AAA/USDT", n=100)
    gapped = pl.concat([raw.slice(0, 50), raw.slice(60, None)])  # 10 missing bars
    labels = pair_labels(gapped, H, VOL_SPAN, BAR_MS)
    # for the row right before the gap, the H-th bar ahead lies beyond it:
    # wall-clock distance must be H + 10 hours, NOT H — purging relies on this
    row = labels.row(49, named=True)
    assert row["label_end_ts"] - row["ts"] == timedelta(hours=H + 10)


def test_null_pattern_warmup_and_tail():
    raw = make_raw("AAA/USDT", n=200)
    labels = pair_labels(raw, H, VOL_SPAN, BAR_MS)
    y = labels["y"]
    # leading nulls: vol needs VOL_SPAN observations; trailing nulls: no forward close
    assert int(y.null_count()) == VOL_SPAN + H
    assert y.slice(0, VOL_SPAN).null_count() == VOL_SPAN
    assert y.slice(200 - H, H).null_count() == H
    assert y.slice(VOL_SPAN, 200 - VOL_SPAN - H).null_count() == 0
