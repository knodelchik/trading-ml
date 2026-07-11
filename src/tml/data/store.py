"""Canonical storage layout and schema.

All timestamps in storage are **bar close times**, UTC. That convention is
established at ingestion and every downstream stage relies on it: a row keyed
by `ts` may only use information available at `ts`.

Layout (all under artifacts/, gitignored):
    data/raw/exchange=<ex>/pair=<PAIR>/year=<YYYY>/part.parquet
    data/features/features.parquet
    data/labels/labels.parquet
    predictions/{oos.parquet, folds.json}
    backtest/{fills.parquet, equity.parquet, stats.json}
    reports/
    trials.jsonl
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from tml.config import Config

RAW_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "pair": pl.String(),
}


def pair_key(pair: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT' (filesystem- and nautilus-friendly)."""
    return pair.replace("/", "")


def raw_pair_dir(cfg: Config, pair: str) -> Path:
    return cfg.artifacts / "data" / "raw" / f"exchange={cfg.data.exchange}" / f"pair={pair_key(pair)}"


def features_path(cfg: Config) -> Path:
    return cfg.artifacts / "data" / "features" / "features.parquet"


def labels_path(cfg: Config) -> Path:
    return cfg.artifacts / "data" / "labels" / "labels.parquet"


def oos_path(cfg: Config) -> Path:
    return cfg.artifacts / "predictions" / "oos.parquet"


def folds_path(cfg: Config) -> Path:
    return cfg.artifacts / "predictions" / "folds.json"


def backtest_dir(cfg: Config) -> Path:
    return cfg.artifacts / "backtest"


def reports_dir(cfg: Config) -> Path:
    return cfg.artifacts / "reports"


def trials_path(cfg: Config) -> Path:
    return cfg.artifacts / "trials.jsonl"


def quality_report_path(cfg: Config) -> Path:
    return cfg.artifacts / "data" / "quality_report.json"


def write_raw_year(cfg: Config, pair: str, year: int, df: pl.DataFrame) -> None:
    """Merge-write one pair-year partition (idempotent: dedupes on ts)."""
    part_dir = raw_pair_dir(cfg, pair) / f"year={year}"
    part_dir.mkdir(parents=True, exist_ok=True)
    path = part_dir / "part.parquet"
    if path.exists():
        existing = pl.read_parquet(path)
        df = pl.concat([existing, df], how="vertical")
    df = df.unique(subset="ts", keep="last").sort("ts")
    df.write_parquet(path)


def scan_raw(cfg: Config, pair: str | None = None) -> pl.LazyFrame:
    """Lazy scan of raw bars for one pair or all pairs."""
    # exact filename, not *.parquet: macOS tar transfers can drop AppleDouble
    # junk ("._part.parquet") next to real files, and a wildcard would eat it
    if pair is not None:
        pattern = raw_pair_dir(cfg, pair) / "**" / "part.parquet"
    else:
        pattern = cfg.artifacts / "data" / "raw" / f"exchange={cfg.data.exchange}" / "**" / "part.parquet"
    return pl.scan_parquet(str(pattern)).sort("ts")


def last_raw_ts(cfg: Config, pair: str) -> datetime | None:
    """Latest stored bar-close timestamp for incremental resume."""
    d = raw_pair_dir(cfg, pair)
    if not d.exists() or not any(d.rglob("*.parquet")):
        return None
    out = scan_raw(cfg, pair).select(pl.col("ts").max()).collect()
    return out.item()
