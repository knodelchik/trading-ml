"""Funding features: point-in-time correctness of the asof alignment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from tests.conftest import make_config
from tml.data.funding import funding_path
from tml.features.funding import add_funding_features

T0 = datetime(2021, 1, 1, 0, 0, tzinfo=UTC)


def _write_funding(cfg, pair: str, rows: list[tuple[datetime, float]]) -> None:
    df = pl.DataFrame(
        {"ts": [r[0] for r in rows], "rate": [r[1] for r in rows], "pair": pair.replace("/", "")},
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    ).sort("ts")
    p = funding_path(cfg, pair)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)


def _bars(pair: str, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [T0 + timedelta(hours=i + 1) for i in range(n)],  # closes 01:00, 02:00, ...
            "pair": pair.replace("/", ""),
            "close": [100.0] * n,
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )


def test_asof_uses_latest_settlement_at_or_before_bar_close(tmp_path):
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), funding=True)
    # settlement with millisecond jitter AFTER the 08:00 bar close: the rate
    # must only become visible from the 09:00 bar onward
    _write_funding(
        cfg,
        "LTC/USDT",
        [
            (T0, 0.01),
            (T0 + timedelta(hours=8, milliseconds=2), 0.02),
            (T0 + timedelta(hours=16), 0.03),
        ],
    )
    out = add_funding_features(_bars("LTC/USDT", 20), cfg)

    def rate_at(hour: int) -> float:
        return out.filter(pl.col("ts") == T0 + timedelta(hours=hour))["fund_rate"][0]

    assert rate_at(1) == pl.Series([0.01], dtype=pl.Float32)[0]
    assert rate_at(8) == pl.Series([0.01], dtype=pl.Float32)[0]  # 08:00.002 settles AFTER close
    assert rate_at(9) == pl.Series([0.02], dtype=pl.Float32)[0]
    assert rate_at(16) == pl.Series([0.03], dtype=pl.Float32)[0]
    assert out["fund_known"].sum() == 20  # every bar has some prior settlement


def test_prelisting_bars_are_neutral_and_flagged(tmp_path):
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), funding=True)
    _write_funding(cfg, "LTC/USDT", [(T0 + timedelta(hours=10), 0.05)])
    out = add_funding_features(_bars("LTC/USDT", 20), cfg)
    # a settlement at exactly 10:00:00.000 IS known at the 10:00 bar close
    pre = out.filter(pl.col("ts") < T0 + timedelta(hours=10))
    post = out.filter(pl.col("ts") >= T0 + timedelta(hours=10))
    assert (pre["fund_known"] == 0.0).all()
    assert (pre["fund_rate"] == 0.0).all()
    assert (post["fund_known"] == 1.0).all()


def test_future_funding_perturbation_does_not_change_past(tmp_path):
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), funding=True)
    rows = [(T0 + timedelta(hours=8 * i), 0.001 * i) for i in range(12)]
    _write_funding(cfg, "LTC/USDT", rows)
    before = add_funding_features(_bars("LTC/USDT", 96), cfg)

    cut = T0 + timedelta(hours=48)
    perturbed = [(ts, r if ts <= cut else r * -7.0) for ts, r in rows]
    _write_funding(cfg, "LTC/USDT", perturbed)
    after = add_funding_features(_bars("LTC/USDT", 96), cfg)

    assert_frame_equal(
        before.filter(pl.col("ts") <= cut), after.filter(pl.col("ts") <= cut)
    )
