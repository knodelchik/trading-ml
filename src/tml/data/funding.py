"""Funding-rate ingestion for Binance USDT perpetuals (public, no keys).

Facts verified against ccxt 4.5.64 and live probes:
  * endpoint returns settlement records {fundingTime ms, fundingRate float},
    ascending, max 1000/page; cursor = last timestamp + 1 ms, stop on a
    short page (the built-in paginate=True assumes a fixed 8h step and can
    DROP records during 2h/4h funding regimes — e.g. SOL, Nov 2022);
  * fundingTime has millisecond jitter off the 8h grid — timestamps are
    stored raw and consumed via backward asof joins, never grid-compared;
  * perp listings postdate spot listings (BTC 2019-09 .. SOL 2020-09), so
    early bars legitimately have no funding — handled at the feature layer.
"""

from __future__ import annotations

import time

import ccxt
import polars as pl

from tml.config import Config
from tml.data import store
from tml.data.ingest import MAX_RETRIES, RETRYABLE

PAGE_LIMIT = 1000

FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),  # settlement time: the rate is known from here on
    "rate": pl.Float64(),
    "pair": pl.String(),
}


def funding_path(cfg: Config, pair: str):
    return cfg.artifacts / "data" / "funding" / f"pair={store.pair_key(pair)}" / "part.parquet"


def scan_funding(cfg: Config, pair: str) -> pl.LazyFrame | None:
    p = funding_path(cfg, pair)
    if not p.exists():
        return None
    return pl.scan_parquet(str(p)).sort("ts")


def perp_symbol(pair: str) -> str:
    base = pair.split("/")[0]
    return f"{base}/USDT:USDT"


def _fetch_page(exchange, symbol: str, since_ms: int) -> list[dict]:
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            return exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=PAGE_LIMIT)
        except RETRYABLE as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after {type(e).__name__}: sleeping {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_funding_pair(exchange, cfg: Config, pair: str) -> int:
    path = funding_path(cfg, pair)
    existing = pl.read_parquet(path) if path.exists() else None
    if existing is not None and existing.height:
        from datetime import datetime
        from typing import cast

        since_ms = int(cast(datetime, existing["ts"].max()).timestamp() * 1000) + 1
    else:
        since_ms = 1
    records: list[dict] = []
    while True:
        page = _fetch_page(exchange, perp_symbol(pair), since_ms)
        if not page:
            break
        records.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        since_ms = page[-1]["timestamp"] + 1

    if not records:
        return 0
    new = (
        pl.DataFrame(
            {
                "ts_ms": [int(r["timestamp"]) for r in records],
                "rate": [float(r["fundingRate"]) for r in records],
            }
        )
        .with_columns(
            ts=pl.from_epoch("ts_ms", time_unit="ms").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
            pair=pl.lit(store.pair_key(pair)),
        )
        .select("ts", "rate", "pair")
    )
    df = pl.concat([existing, new]) if existing is not None else new
    df = df.unique(subset="ts", keep="last").sort("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return new.height


def run_funding_ingest(cfg: Config) -> int:
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    total = 0
    from tml.universe import pairs_for

    for pair in pairs_for(cfg):
        try:
            n = fetch_funding_pair(exchange, cfg, pair)
        except ccxt.BadSymbol:
            # delisted perp (e.g. old LUNA): no live endpoint; funding
            # features fall back to fund_known=0 for such pairs
            print(f"funding {pair}: perp not listed — skipped (features 0-fill)")
            continue
        total += n
        print(f"funding {pair}: +{n} settlements")
    print(f"funding ingest done: {total} new settlement records")
    return 0
