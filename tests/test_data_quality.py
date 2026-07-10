"""Ingest transformation and quality-gate checks (leakage traps 1, 7, 8)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from tml.data import store
from tml.data.ingest import rows_to_frame
from tml.data.quality import check_pair

BAR_MS = 3_600_000


def _ccxt_rows(n: int, start_open_ms: int = 1_600_000_000_000) -> list[list]:
    return [[start_open_ms + i * BAR_MS, 100.0, 101.0, 99.0, 100.5, 10.0] for i in range(n)]


def test_ts_is_close_time():
    rows = _ccxt_rows(5)
    now_ms = rows[-1][0] + 10 * BAR_MS  # far in the future: nothing dropped
    df = rows_to_frame(rows, BAR_MS, "TESTUSDT", now_ms)
    expected_first_close = datetime.fromtimestamp((rows[0][0] + BAR_MS) / 1000, UTC)
    assert df["ts"][0] == expected_first_close
    assert df.height == 5


def test_no_partial_candle():
    rows = _ccxt_rows(5)
    # "now" lands in the middle of the last bar -> its close is in the future
    now_ms = rows[-1][0] + BAR_MS // 2
    df = rows_to_frame(rows, BAR_MS, "TESTUSDT", now_ms)
    assert df.height == 4
    assert df["ts"].max() < datetime.fromtimestamp(now_ms / 1000, UTC)


def test_quality_detects_gap_and_does_not_fill(synth_cfg):
    from tests.conftest import make_raw

    df = make_raw("AAA/USDT", n=500)
    with_gap = pl.concat([df.slice(0, 200), df.slice(210, None)])  # 10 bars missing
    for (year,), part in with_gap.with_columns(year=pl.col("ts").dt.year()).group_by(
        "year", maintain_order=True
    ):
        store.write_raw_year(synth_cfg, "AAA/USDT", int(year), part.drop("year"))

    rep = check_pair(synth_cfg, "AAA/USDT")
    assert not rep.hard_fail
    assert len(rep.gaps) == 1
    assert rep.gaps[0]["missing_bars"] == 10
    # the gap is journaled, NOT filled: row count unchanged
    assert store.scan_raw(synth_cfg, "AAA/USDT").collect().height == 490


def test_quality_hard_fails_on_future_candle(synth_cfg):
    from tests.conftest import make_raw

    df = make_raw("BBB/USDT", n=100, start=datetime.now(UTC) + timedelta(days=2))
    for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
        "year", maintain_order=True
    ):
        store.write_raw_year(synth_cfg, "BBB/USDT", int(year), part.drop("year"))
    rep = check_pair(synth_cfg, "BBB/USDT")
    assert rep.hard_fail
    assert rep.partial_candles > 0


def test_write_raw_year_is_idempotent(synth_store):
    before = store.scan_raw(synth_store, "AAA/USDT").collect()
    df = before.clone()
    for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
        "year", maintain_order=True
    ):
        store.write_raw_year(synth_store, "AAA/USDT", int(year), part.drop("year"))
    after = store.scan_raw(synth_store, "AAA/USDT").collect()
    assert before.equals(after)
