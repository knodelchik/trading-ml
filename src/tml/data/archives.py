"""Binance public data archives (data.binance.vision) — the only source
that survives delistings, which a point-in-time universe depends on.

Verified facts this module encodes:
  * monthly zips: spot klines
    data/spot/monthly/klines/{SYM}/{interval}/{SYM}-{interval}-{YYYY-MM}.zip
    and USDT-perp funding
    data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip
  * spot kline CSVs have NO header, 12 columns; funding CSVs HAVE a header;
  * TIMESTAMP TRAP: kline open/close times are epoch MILLISECONDS through
    2024-12 and MICROSECONDS from 2025-01 — branch on digit count;
  * archives persist after delisting (LUNA, FTT verified); a 404 for a
    month simply means the symbol didn't trade that month;
  * month enumeration via the public S3 listing endpoint.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime

import polars as pl

BASE = "https://data.binance.vision/data"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _get(url: str, retries: int = 4) -> bytes | None:
    """GET with retries; None on 404 (month simply absent)."""
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, OSError, TimeoutError):
            # ConnectionResetError etc. — inevitable across tens of
            # thousands of requests; retry with backoff
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    return None


def _epoch_to_us(value: int) -> int:
    """ms (13 digits, pre-2025) or us (16 digits, 2025+) -> microseconds."""
    return value * 1000 if value < 10**15 else value


def month_range(start: str, end: str) -> list[str]:
    """['2020-01', ...] inclusive."""
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch_spot_klines_month(symbol: str, interval: str, month: str) -> pl.DataFrame | None:
    """One month of spot klines -> canonical frame (ts = bar CLOSE, UTC),
    or None if the archive doesn't exist (symbol not trading that month)."""
    url = f"{BASE}/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    blob = _get(url)
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    rows = list(csv.reader(io.StringIO(raw)))
    if rows and not rows[0][0].isdigit():  # defensive: some datasets grew headers
        rows = rows[1:]
    if not rows:
        return None
    return (
        pl.DataFrame(
            {
                "open_us": [_epoch_to_us(int(r[0])) for r in rows],
                "open": [float(r[1]) for r in rows],
                "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows],
                "close": [float(r[4]) for r in rows],
                "volume": [float(r[5]) for r in rows],
                "quote_volume": [float(r[7]) for r in rows],
            }
        )
        # close time in the file is open + interval - 1(ms|us); we normalize
        # to our exact bar-close convention from the open time instead
        .with_columns(
            ts=(pl.from_epoch("open_us", time_unit="us") + _interval_duration(interval))
            .dt.replace_time_zone("UTC"),
            pair=pl.lit(symbol),
        )
        .drop("open_us")
        .sort("ts")
    )


def _interval_duration(interval: str) -> pl.Expr:
    ms = {"1h": 3_600_000, "1d": 86_400_000, "15m": 900_000}[interval]
    return pl.duration(milliseconds=ms)


def fetch_spot_taker_month(symbol: str, interval: str, month: str) -> pl.DataFrame | None:
    """One month of spot klines reduced to the taker buy/sell split:
    taker_buy_ratio = taker_buy_quote / quote_volume (fraction of volume
    that lifted the offer — aggressive buying). ts = bar CLOSE, UTC.
    Columns 7=quote_volume, 10=taker_buy_quote in the 12-col kline CSV.
    """
    url = f"{BASE}/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    blob = _get(url)
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    rows = list(csv.reader(io.StringIO(raw)))
    if rows and not rows[0][0].isdigit():
        rows = rows[1:]
    if not rows:
        return None
    return (
        pl.DataFrame(
            {
                "open_us": [_epoch_to_us(int(r[0])) for r in rows],
                "quote_volume": [float(r[7]) for r in rows],
                "taker_buy_quote": [float(r[10]) for r in rows],
            }
        )
        .with_columns(
            ts=(pl.from_epoch("open_us", time_unit="us") + _interval_duration(interval))
            .dt.replace_time_zone("UTC"),
            # neutral 0.5 when a bar had zero volume (no information)
            taker_buy_ratio=pl.when(pl.col("quote_volume") > 0)
            .then(pl.col("taker_buy_quote") / pl.col("quote_volume"))
            .otherwise(0.5),
            pair=pl.lit(symbol),
        )
        .select("ts", "taker_buy_ratio", "pair")
        .sort("ts")
    )


def fetch_funding_month(symbol: str, month: str) -> pl.DataFrame | None:
    """One month of perp funding settlements (HAS a header row)."""
    url = f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    blob = _get(url)
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows:
        return None
    return (
        pl.DataFrame(
            {
                "ts_us": [_epoch_to_us(int(r["calc_time"])) for r in rows],
                "rate": [float(r["last_funding_rate"]) for r in rows],
            }
        )
        .with_columns(
            ts=pl.from_epoch("ts_us", time_unit="us").dt.replace_time_zone("UTC"),
            pair=pl.lit(symbol),
        )
        .drop("ts_us")
        .sort("ts")
    )


def list_available_months(symbol: str, interval: str = "1h") -> list[str]:
    """Enumerate archive months for a symbol via the S3 listing (public)."""
    url = (
        f"{S3}?delimiter=/&prefix=data/spot/monthly/klines/{symbol}/{interval}/"
    )
    blob = _get(url)
    if blob is None:
        return []
    text = blob.decode()
    months = []
    marker = f"{symbol}-{interval}-"
    i = 0
    while (i := text.find(marker, i)) != -1:
        chunk = text[i + len(marker) : i + len(marker) + 7]
        if len(chunk) == 7 and chunk[4] == "-" and chunk.endswith(("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")):
            if chunk not in months and not text[i:].startswith(f"{marker}{chunk}.zip.CHECKSUM", 0):
                months.append(chunk)
        i += len(marker)
    return sorted(set(months))


def now_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")
