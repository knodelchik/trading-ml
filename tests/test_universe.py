"""Point-in-time universe: archive parsing traps and membership rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from tests.conftest import make_config
from tml.data.archives import _epoch_to_us, month_range
from tml.universe import build_membership


def test_epoch_unit_trap():
    # 13-digit ms (pre-2025 archives) vs 16-digit us (2025+)
    assert _epoch_to_us(1619827200000) == 1619827200000000
    assert _epoch_to_us(1735689600000000) == 1735689600000000


def test_month_range():
    assert month_range("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]


def _daily(pair: str, start: datetime, end: datetime, vol: float, skip: tuple = ()) -> pl.DataFrame:
    days = []
    d = start
    while d <= end:
        if not any(a <= d <= b for a, b in skip):
            days.append(d)
        d += timedelta(days=1)
    return pl.DataFrame(
        {
            "ts": days,
            "close": [1.0] * len(days),
            "quote_volume": [vol] * len(days),
            "pair": pair,
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )


def test_membership_history_gap_and_topn(tmp_path):
    cfg = make_config(
        tmp_path,
        universe={"enabled": True, "size": 2, "ranking_window_days": 30, "min_history_days": 60, "max_gap_days": 3},
    )
    y = 2021
    jan1, dec31 = datetime(y, 1, 1, tzinfo=UTC), datetime(y, 12, 31, tzinfo=UTC)
    daily = pl.concat(
        [
            _daily("AAAUSDT", jan1, dec31, vol=1000),  # steady incumbent
            _daily("BBBUSDT", datetime(y, 6, 1, tzinfo=UTC), dec31, vol=2000),  # late lister
            _daily(  # LUNA-style: 18-day hole in July
                "CCCUSDT", jan1, dec31, vol=1500,
                skip=((datetime(y, 7, 1, tzinfo=UTC), datetime(y, 7, 18, tzinfo=UTC)),),
            ),
        ]
    )
    (tmp_path / "artifacts" / "universe").mkdir(parents=True)
    m = build_membership(cfg, daily)

    def members(month: str) -> set[str]:
        return set(m.filter(pl.col("month") == month)["pair"].to_list())

    # April: A and C have 60d history; B doesn't exist yet
    assert members("2021-04") == {"AAAUSDT", "CCCUSDT"}
    # August: C's July hole (>3d) is inside its history window -> ineligible;
    # B has just cleared 60 days -> in
    assert members("2021-08") == {"AAAUSDT", "BBBUSDT"}
    # November: C's hole has aged out of the 60d window -> re-qualifies,
    # and top-2 by volume is {B: 2000, C: 1500} — A (1000) is cut
    assert members("2021-11") == {"BBBUSDT", "CCCUSDT"}
