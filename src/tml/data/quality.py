"""Data quality gate.

Hard failures (nonzero exit): duplicate timestamps, non-monotonic timestamps,
non-positive prices, a partial (future-closing) candle, OHLC inconsistency.
Soft findings (recorded, never fixed): gaps. Binance has real maintenance
windows; interpolating them would smuggle future information into features,
so gaps are journaled and left in place — feature windows are time-aware.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl

from tml.config import Config
from tml.data import store

# a pair must actually contain >= this fraction of the bars its own
# [ts_min, ts_max] span implies — otherwise a mostly-empty series would
# silently pass as "ok" with a long gap list
MIN_COVERAGE = 0.95


@dataclass
class PairQuality:
    pair: str
    rows: int
    ts_min: str
    ts_max: str
    coverage: float
    duplicate_ts: int
    non_monotonic: int
    non_positive_prices: int
    ohlc_violations: int
    partial_candles: int
    gaps: list[dict] = field(default_factory=list)

    @property
    def hard_fail(self) -> bool:
        return bool(
            self.duplicate_ts
            or self.non_monotonic
            or self.non_positive_prices
            or self.ohlc_violations
            or self.partial_candles
            or self.coverage < MIN_COVERAGE
        )


def check_pair(cfg: Config, pair: str) -> PairQuality:
    df = store.scan_raw(cfg, pair).collect()
    bar = timedelta(milliseconds=cfg.data.bar_ms)
    ts = df["ts"]

    dup = df.height - df["ts"].n_unique()
    non_mono = int((ts.diff().drop_nulls() <= timedelta(0)).sum())
    non_pos = int(
        df.select(
            ((pl.col("open") <= 0) | (pl.col("high") <= 0) | (pl.col("low") <= 0) | (pl.col("close") <= 0))
            .sum()
        ).item()
    )
    ohlc_bad = int(
        df.select(
            (
                (pl.col("high") < pl.col("low"))
                | (pl.col("high") < pl.max_horizontal("open", "close"))
                | (pl.col("low") > pl.min_horizontal("open", "close"))
            ).sum()
        ).item()
    )
    partial = int((ts > datetime.now(UTC)).sum())

    gap_df = (
        df.select("ts")
        .with_columns(prev=pl.col("ts").shift(1))
        .with_columns(delta=pl.col("ts") - pl.col("prev"))
        .filter(pl.col("delta") > bar)
    )
    gaps = [
        {
            "after": row["prev"].isoformat(),
            "resume": row["ts"].isoformat(),
            "missing_bars": int(row["delta"] / bar) - 1,
        }
        for row in gap_df.iter_rows(named=True)
    ]

    span_bars = int((cast(datetime, ts.max()) - cast(datetime, ts.min())) / bar) + 1
    return PairQuality(
        pair=store.pair_key(pair),
        rows=df.height,
        ts_min=cast(datetime, ts.min()).isoformat(),
        ts_max=cast(datetime, ts.max()).isoformat(),
        coverage=round(df.height / span_bars, 6),
        duplicate_ts=dup,
        non_monotonic=non_mono,
        non_positive_prices=non_pos,
        ohlc_violations=ohlc_bad,
        partial_candles=partial,
        gaps=gaps,
    )


def run_quality(cfg: Config) -> int:
    reports = [check_pair(cfg, pair) for pair in cfg.data.pairs]
    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hard_fail": any(r.hard_fail for r in reports),
        "pairs": [asdict(r) for r in reports],
    }
    path = store.quality_report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    for r in reports:
        status = "FAIL" if r.hard_fail else "ok"
        print(
            f"{r.pair}: {r.rows} rows [{r.ts_min} .. {r.ts_max}] "
            f"coverage={r.coverage:.4f} gaps={len(r.gaps)} {status}"
        )
    if out["hard_fail"]:
        print("quality: HARD FAILURES present, see quality_report.json")
        return 1
    print(f"quality: all pairs clean (report: {path})")
    return 0
