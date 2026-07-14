"""Point-in-time universe: monthly top-N by trailing liquidity.

At the start of each month M, membership is decided ONLY from data
available before M:
  * rank all USDT spot symbols (including delisted ones — archives survive
    delisting) by median daily quote volume over the trailing
    ranking_window_days;
  * eligibility: >= min_history_days of history AND no gap longer than
    max_gap_days within that window — the continuity rule doubles as the
    ticker-reuse guard (LUNA's 18-day halt in 2022-05 and FTT's 9-month
    suspension reset their clocks, so the reborn tickers re-qualify only
    after building fresh continuous history);
  * take the top `size`.

Backtest membership is enforced at the dataset level (a pair's rows exist
only for months it was in the universe) plus a forced position exit on a
pair's final in-universe bar. Live trades the CURRENT month's membership
intersected with Bybit availability.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

import polars as pl

from tml.config import Config
from tml.data import archives

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
STABLE_BASES = {
    "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "PAX", "USDP", "SUSD",
    "EUR", "GBP", "AEUR", "USDS", "USD1", "EURI", "XUSD", "USDE",
}


def universe_dir(cfg: Config):
    return cfg.artifacts / "universe"


def daily_path(cfg: Config):
    return universe_dir(cfg) / "daily.parquet"


def membership_path(cfg: Config):
    return universe_dir(cfg) / "membership.parquet"


def current_pairs_path(cfg: Config):
    return universe_dir(cfg) / "current_pairs.json"


def discover_symbols() -> list[str]:
    """All USDT spot symbols Binance has ever listed (active + BREAK),
    minus leveraged tokens and fiat-stables."""
    import ccxt

    ex = ccxt.binance()
    markets = ex.load_markets()
    out = []
    for m in markets.values():
        if not m.get("spot") or m.get("quote") != "USDT":
            continue
        base = m.get("base") or ""
        if base in STABLE_BASES:
            continue
        if any(base.endswith(suf) and len(base) > len(suf) + 1 for suf in LEVERAGED_SUFFIXES):
            continue
        symbol = m["id"]  # exchange symbol, e.g. BTCUSDT
        if not re.fullmatch(r"[A-Z0-9]+USDT", symbol):
            continue  # exotic/non-ASCII ids break archive URLs
        out.append(symbol)
    return sorted(set(out))


def _fetch_symbol_daily(cfg: Config, symbol: str, months: list[str]) -> pl.DataFrame | None:
    """Per-symbol daily series with an on-disk cache: a crashed bulk build
    resumes instead of restarting 40k downloads."""
    cache = universe_dir(cfg) / "daily_cache" / f"{symbol}.parquet"
    if cache.exists():
        cached = pl.read_parquet(cache)
        have = set(cached["ts"].dt.strftime("%Y-%m").unique().to_list())
        # last cached month may be partial — always refetch it and current
        refetch = sorted({m for m in months if m not in have} | set(sorted(have)[-1:]))
    else:
        cached = None
        refetch = months
    frames = [] if cached is None else [cached.filter(~pl.col("ts").dt.strftime("%Y-%m").is_in(refetch))]
    for month in refetch:
        df = archives.fetch_spot_klines_month(symbol, "1d", month)
        if df is not None:
            frames.append(df.select("ts", "close", "quote_volume", "pair"))
    frames = [f for f in frames if f.height]
    if not frames:
        return None
    out = pl.concat(frames).unique(subset="ts").sort("ts")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache)
    return out


def build_daily(cfg: Config, symbols: list[str]) -> pl.DataFrame:
    """Daily quote-volume panel for ranking (bulk archive download,
    parallel; ~500 symbols x ~78 months of tiny zips)."""
    months = archives.month_range(cfg.data.since.strftime("%Y-%m"), archives.now_month())
    frames: list[pl.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_symbol_daily, cfg, s, months): s for s in symbols}
        for fut in as_completed(futures):
            df = fut.result()
            done += 1
            if df is not None:
                frames.append(df)
            if done % 50 == 0:
                print(f"universe: daily data {done}/{len(symbols)} symbols")
    daily = pl.concat(frames).sort("pair", "ts")
    p = daily_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(p)
    print(f"universe: daily panel {daily.height} rows, {daily['pair'].n_unique()} symbols -> {p}")
    return daily


def build_membership(cfg: Config, daily: pl.DataFrame) -> pl.DataFrame:
    """(month, pair) membership decided point-in-time at each month start."""
    u = cfg.universe
    first_month = (cfg.data.since + timedelta(days=u.min_history_days)).strftime("%Y-%m")
    months = archives.month_range(first_month, archives.now_month())
    rows: list[dict] = []
    for month in months:
        m_start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
        win_start = m_start - timedelta(days=u.ranking_window_days)
        hist_start = m_start - timedelta(days=u.min_history_days)

        window = daily.filter((pl.col("ts") > win_start) & (pl.col("ts") <= m_start))
        history = daily.filter((pl.col("ts") > hist_start) & (pl.col("ts") <= m_start))

        # continuity over the full history window: max day-gap per pair
        gaps = (
            history.sort("pair", "ts")
            .with_columns(gap=(pl.col("ts") - pl.col("ts").shift(1).over("pair")).dt.total_days())
            .group_by("pair")
            .agg(
                max_gap=pl.col("gap").max(),
                days=pl.len(),
                first_ts=pl.col("ts").min(),
            )
        )
        eligible = gaps.filter(
            (pl.col("first_ts") <= hist_start + timedelta(days=u.max_gap_days))
            & (pl.col("max_gap").fill_null(1) <= u.max_gap_days)
            & (pl.col("days") >= u.min_history_days * 0.9)
        )["pair"]

        ranked = (
            window.filter(pl.col("pair").is_in(eligible.implode()))
            .group_by("pair")
            .agg(vol=pl.col("quote_volume").median(), days=pl.len())
            .filter(pl.col("days") >= u.ranking_window_days * 0.9)
            .sort("vol", descending=True)
            .head(u.size)
        )
        for pair in ranked["pair"]:
            rows.append({"month": month, "pair": pair})

    membership = pl.DataFrame(rows)
    membership.write_parquet(membership_path(cfg))
    latest = months[-1]
    current = membership.filter(pl.col("month") == latest)["pair"].to_list()
    current_pairs_path(cfg).write_text(json.dumps({"month": latest, "pairs": current}, indent=2))
    print(
        f"universe: {membership['month'].n_unique()} months, "
        f"{membership['pair'].n_unique()} distinct symbols ever member, "
        f"current ({latest}): {len(current)} pairs -> {membership_path(cfg)}"
    )
    return membership


def current_members(cfg: Config) -> list[str]:
    """The CURRENT month's universe membership as 'BASE/USDT' pairs — the
    set the live bot is allowed to actively enter. Written by `tml universe`."""
    data = json.loads(current_pairs_path(cfg).read_text())
    return sorted(f"{s[:-4]}/USDT" for s in data["pairs"])


def pairs_for(cfg: Config) -> list[str]:
    """Data-stage pair list: static config pairs, or (universe mode) every
    symbol that was EVER a member — history for ex-members is still needed
    to train and to backtest their in-membership periods."""
    if not cfg.universe.enabled:
        return list(cfg.data.pairs)
    m = pl.read_parquet(membership_path(cfg))
    return sorted(f"{s[:-4]}/USDT" for s in m["pair"].unique().to_list())


def membership_filter(cfg: Config, df: pl.DataFrame) -> pl.DataFrame:
    """Keep only (ts, pair) rows where the pair was a member during ts's
    month. No-op when universe mode is off."""
    if not cfg.universe.enabled:
        return df
    m = pl.read_parquet(membership_path(cfg)).with_columns(member=pl.lit(True))
    return (
        df.with_columns(month=pl.col("ts").dt.strftime("%Y-%m"))
        .join(m, left_on=["month", "pair"], right_on=["month", "pair"], how="left")
        .filter(pl.col("member"))
        .drop("month", "member")
    )


def run_universe(cfg: Config) -> int:
    symbols = discover_symbols()
    print(f"universe: {len(symbols)} candidate USDT symbols (incl. delisted)")
    daily = build_daily(cfg, symbols)
    build_membership(cfg, daily)
    return 0
