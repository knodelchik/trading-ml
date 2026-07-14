"""Taker buy/sell flow ingestion from Binance archives (data.binance.vision).

Order-flow signal: what fraction of each bar's volume lifted the offer
(aggressive buying) vs hit the bid. ccxt fetch_ohlcv drops this column, so
it comes from the 12-column kline archives. Stored per pair on the same
bar-close grid as the OHLCV, cached per symbol so a crashed bulk build
resumes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

from tml.config import Config
from tml.data import archives, store


def taker_path(cfg: Config, pair: str):
    return cfg.artifacts / "data" / "taker" / f"pair={store.pair_key(pair)}" / "part.parquet"


def scan_taker(cfg: Config, pair: str) -> pl.LazyFrame | None:
    p = taker_path(cfg, pair)
    if not p.exists():
        return None
    return pl.scan_parquet(str(p)).sort("ts")


def _fetch_symbol(cfg: Config, pair: str, months: list[str]) -> int:
    symbol = store.pair_key(pair)
    path = taker_path(cfg, pair)
    have: set[str] = set()
    existing = None
    if path.exists():
        existing = pl.read_parquet(path)
        have = set(existing["ts"].dt.strftime("%Y-%m").unique().to_list())
    refetch = sorted(({m for m in months if m not in have}) | set(sorted(have)[-1:]))
    frames = [] if existing is None else [existing.filter(~pl.col("ts").dt.strftime("%Y-%m").is_in(refetch))]
    for month in refetch:
        df = archives.fetch_spot_taker_month(symbol, cfg.data.timeframe, month)
        if df is not None:
            frames.append(df)
    frames = [f for f in frames if f.height]
    if not frames:
        return 0
    out = pl.concat(frames).unique(subset="ts").sort("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out.height


def run_taker_ingest(cfg: Config) -> int:
    from tml.universe import pairs_for

    pairs = pairs_for(cfg)
    months = archives.month_range(cfg.data.since.strftime("%Y-%m"), archives.now_month())
    total = done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_symbol, cfg, p, months): p for p in pairs}
        for fut in as_completed(futures):
            total += fut.result()
            done += 1
            if done % 25 == 0:
                print(f"taker: {done}/{len(pairs)} symbols")
    print(f"taker ingest done: {total} bar-rows across {len(pairs)} symbols")
    return 0
