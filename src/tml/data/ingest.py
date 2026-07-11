"""OHLCV ingestion from exchange public endpoints via ccxt.

Leakage-critical invariants enforced here, at the source:
  * ccxt returns bar OPEN time; we store bar CLOSE time (`ts = open + bar`),
    because a bar is only knowable once it closes.
  * The still-forming final candle is dropped (its close lies in the future).
Incremental: resumes from the last stored close timestamp per pair.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import ccxt
import polars as pl

from tml.config import Config
from tml.data import store

MAX_RETRIES = 6
PAGE_LIMIT = 1000

RETRYABLE = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.RateLimitExceeded,
    ccxt.DDoSProtection,
)


def _fetch_page(exchange: ccxt.Exchange, pair: str, timeframe: str, since_ms: int) -> list[list]:
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=PAGE_LIMIT)
        except RETRYABLE as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after {type(e).__name__}: sleeping {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_pair(exchange: ccxt.Exchange, cfg: Config, pair: str) -> int:
    """Fetch all missing bars for one pair; returns number of new rows stored."""
    bar_ms = cfg.data.bar_ms
    last = store.last_raw_ts(cfg, pair)
    if last is None:
        # config `since` is the earliest OPEN time we want
        since_ms = int(cfg.data.since.timestamp() * 1000)
    else:
        # stored ts is a CLOSE time == OPEN time of the next bar
        since_ms = int(last.timestamp() * 1000)

    rows: list[list] = []
    while True:
        page = _fetch_page(exchange, pair, cfg.data.timeframe, since_ms)
        if not page:
            break
        raw_len = len(page)  # termination must use the UNfiltered length
        next_since = page[-1][0] + bar_ms
        # guard against exchanges ignoring `since` and looping
        rows.extend(r for r in page if r[0] >= since_ms)
        since_ms = next_since
        if raw_len < PAGE_LIMIT:
            break

    if not rows:
        return 0

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    df = rows_to_frame(rows, bar_ms, store.pair_key(pair), now_ms)
    for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
        "year", maintain_order=True
    ):
        store.write_raw_year(cfg, pair, int(year), part.drop("year"))
    return len(df)


def rows_to_frame(rows: list[list], bar_ms: int, pair_key: str, now_ms: int) -> pl.DataFrame:
    """ccxt OHLCV rows -> canonical frame.

    Converts bar OPEN time to CLOSE time (`ts = open + bar`) and drops any
    candle whose close lies in the future (the still-forming one).
    """
    return (
        pl.DataFrame(
            {
                "open_ms": [int(r[0]) for r in rows],
                "open": [float(r[1]) for r in rows],
                "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows],
                "close": [float(r[4]) for r in rows],
                "volume": [float(r[5]) for r in rows],
            }
        )
        .with_columns(close_ms=pl.col("open_ms") + bar_ms)
        .filter(pl.col("close_ms") <= now_ms)
        .with_columns(
            ts=pl.from_epoch("close_ms", time_unit="ms").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
            pair=pl.lit(pair_key),
        )
        .select("ts", "open", "high", "low", "close", "volume", "pair")
        .unique(subset="ts", keep="last")
        .sort("ts")
    )


def run_ingest(cfg: Config) -> int:
    exchange_cls = getattr(ccxt, cfg.data.exchange)
    exchange = exchange_cls({"enableRateLimit": True})
    total = 0
    for pair in cfg.data.pairs:
        t0 = time.time()
        n = fetch_pair(exchange, cfg, pair)
        total += n
        print(f"{pair}: +{n} bars in {time.time() - t0:.1f}s")
    print(f"ingest done: {total} new bars across {len(cfg.data.pairs)} pairs")
    return 0
