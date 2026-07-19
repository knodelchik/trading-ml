"""Perp-basis ingestion from premium-index klines (public, no keys).

The premium index tracks the fractional perp-spot basis continuously; its
1h kline "close" is the basis at the bar close — a carry/positioning signal
that leads funding. Two sources serve the SAME series (verified 2026-07
against BTCUSDT: identical 12-column kline shape):
  * history: data.binance.vision monthly zips
    data/futures/um/monthly/premiumIndexKlines/{SYM}/1h/{SYM}-1h-{YYYY-MM}.zip
    — but the freshest month's zip appears only days AFTER month end;
  * recency: GET /fapi/v1/premiumIndexKlines?symbol=&interval=1h — JSON
    kline arrays, ms epochs, max 1500/page, ascending; INCLUDES the
    still-open kline (its close_time is in the future — dropped here: a row
    may exist only once its bar has closed);
  * an unlisted symbol answers HTTP 400 code -1121 — the ccxt.BadSymbol
    analogue, mapped to PerpUnlisted so the ingest loop can skip it.
Perp listings postdate spot, and delisted perps stop printing while spot
continues — both handled at the feature layer via basis_known.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import cast

import polars as pl

from tml.config import Config
from tml.data import archives, store

FAPI = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
REST_PAGE_LIMIT = 1500
MAX_RETRIES = 4

BASIS_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),  # kline CLOSE time: the basis is known from here on
    "basis": pl.Float64(),
    "pair": pl.String(),
}


class PerpUnlisted(Exception):
    """Symbol has no USDT perp (REST code -1121) — ccxt.BadSymbol analogue."""


def basis_path(cfg: Config, pair: str):
    return cfg.artifacts / "data" / "basis" / f"pair={store.pair_key(pair)}" / "part.parquet"


def scan_basis(cfg: Config, pair: str) -> pl.LazyFrame | None:
    p = basis_path(cfg, pair)
    if not p.exists():
        return None
    return pl.scan_parquet(str(p)).sort("ts")


def rest_url(symbol: str, start_ms: int, limit: int = REST_PAGE_LIMIT) -> str:
    return f"{FAPI}?symbol={symbol}&interval=1h&startTime={start_ms}&limit={limit}"


def _rest_page(symbol: str, start_ms: int) -> list:
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(rest_url(symbol, start_ms), timeout=30) as resp:
                return cast(list, json.loads(resp.read()))
        except urllib.error.HTTPError as e:
            if e.code == 400 and b"-1121" in e.read():
                raise PerpUnlisted(symbol) from e
            if attempt == MAX_RETRIES - 1:
                raise
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def fetch_premium_rest(symbol: str, since_ms: int) -> pl.DataFrame:
    """Premium klines with open_time >= since_ms via the live endpoint.
    Closed bars only; empty frame when nothing new. Raises PerpUnlisted."""
    rows: list[list] = []
    cursor = since_ms
    while True:
        page = _rest_page(symbol, cursor)
        if not page:
            break
        rows.extend(page)
        if len(page) < REST_PAGE_LIMIT:
            break
        cursor = int(page[-1][0]) + 1
    now_ms = int(time.time() * 1000)
    rows = [r for r in rows if int(r[6]) <= now_ms]  # drop the still-open kline
    if not rows:
        return pl.DataFrame(schema=BASIS_SCHEMA)
    return (
        pl.DataFrame(
            {
                "open_ms": [int(r[0]) for r in rows],
                "basis": [float(r[4]) for r in rows],
            }
        )
        .with_columns(
            ts=(pl.from_epoch("open_ms", time_unit="ms") + pl.duration(hours=1))
            .dt.cast_time_unit("us")
            .dt.replace_time_zone("UTC"),
            pair=pl.lit(symbol),
        )
        .select("ts", "basis", "pair")
        .sort("ts")
    )


def _merge_write(path, frames: list[pl.DataFrame]) -> pl.DataFrame:
    out = pl.concat(frames).unique(subset="ts", keep="last").sort("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


def fetch_basis_pair(cfg: Config, pair: str, months: list[str]) -> int:
    """Archive backfill (cached months resume) + REST top-up to now: the
    freshest monthly zip lags days behind, so without the top-up the live
    tail would always sit beyond the staleness cutoff."""
    symbol = store.pair_key(pair)
    path = basis_path(cfg, pair)
    existing = pl.read_parquet(path) if path.exists() else None
    have: set[str] = (
        set(existing["ts"].dt.strftime("%Y-%m").unique().to_list()) if existing is not None else set()
    )
    refetch = sorted({m for m in months if m not in have} | set(sorted(have)[-1:]))
    frames = [] if existing is None else [existing.filter(~pl.col("ts").dt.strftime("%Y-%m").is_in(refetch))]
    for month in refetch:
        y, m = month.split("-")
        df = archives.fetch_premium_month(symbol, int(y), int(m))
        if df is not None:
            frames.append(df)
    frames = [f for f in frames if f.height]
    if frames:
        last = max(cast(datetime, f["ts"].max()) for f in frames)
        since_ms = int(last.timestamp() * 1000) - 3_600_000  # re-verify the seam bar
    else:
        since_ms = int(cfg.data.since.timestamp() * 1000)
    try:
        rest = fetch_premium_rest(symbol, since_ms)
    except PerpUnlisted:
        if not frames:
            raise  # never listed at all — caller prints the skip
        rest = None  # delisted: the series legitimately ends; staleness cutoff flags later bars
    if rest is not None and rest.height:
        frames.append(rest)
    if not frames:
        return 0
    out = _merge_write(path, frames)
    return out.height - (existing.height if existing is not None else 0)


def refresh_basis_pair(cfg: Config, pair: str) -> int:
    """REST-only top-up from the last stored kline (tick-time fast path)."""
    path = basis_path(cfg, pair)
    existing = pl.read_parquet(path) if path.exists() else None
    if existing is not None and existing.height:
        since_ms = int(cast(datetime, existing["ts"].max()).timestamp() * 1000) - 3_600_000
    else:
        since_ms = int(cfg.data.since.timestamp() * 1000)
    new = fetch_premium_rest(store.pair_key(pair), since_ms)
    if not new.height:
        return 0
    _merge_write(path, [existing, new] if existing is not None else [new])
    return new.height


def run_basis_ingest(cfg: Config) -> int:
    from tml.universe import pairs_for

    months = archives.month_range(cfg.data.since.strftime("%Y-%m"), archives.now_month())
    total = 0
    for pair in pairs_for(cfg):
        try:
            n = fetch_basis_pair(cfg, pair, months)
        except PerpUnlisted:
            # no perp for this spot symbol (e.g. old LUNA post-delist with no
            # archives): basis features fall back to basis_known=0
            print(f"basis {pair}: perp not listed — skipped (features 0-fill)")
            continue
        total += n
        print(f"basis {pair}: +{n} premium bars")
    print(f"basis ingest done: {total} new premium-kline rows")
    return 0
