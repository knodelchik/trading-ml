"""Instrument definitions and bar preparation for nautilus_trader.

Cost model note: nautilus's FillModel slippage is exactly one price tick,
which is not bps-controllable, so we set prob_slippage=0 and bake the
configured slippage into the instrument's taker fee instead:
    effective_taker = (taker_fee_bps + slippage_bps) / 1e4
MakerTakerFeeModel charges notional * taker_fee on market-order fills, so
this yields exact, deterministic bps-level costs. stats.json discloses it.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import polars as pl

from tml.config import Config
from tml.data import store

VENUE_NAME = "BINANCE"

from nautilus_trader.model.currencies import USDT  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue  # noqa: E402
from nautilus_trader.model.instruments import CryptoPerpetual  # noqa: E402
from nautilus_trader.model.objects import Currency, Price, Quantity  # noqa: E402

PRICE_PRECISION = 6
SIZE_PRECISION = 6


def make_instrument(cfg: Config, pair: str) -> CryptoPerpetual:
    """Linear USDT perpetual (matches what the live bot trades on Bybit).

    Symbol stays exactly `BTCUSDT`: signals and bar types key on
    `{symbol}.{venue}`, so renaming (e.g. BTCUSDT-PERP) would silently turn
    every bar into a missing-signal no-op.
    """
    key = store.pair_key(pair)  # e.g. BTCUSDT
    base = pair.split("/")[0]
    effective_taker = Decimal(cfg.backtest.taker_fee_bps + cfg.backtest.slippage_bps) / Decimal(10_000)
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(key), Venue(VENUE_NAME)),
        raw_symbol=Symbol(key),
        base_currency=Currency.from_str(base),
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=PRICE_PRECISION,
        size_precision=SIZE_PRECISION,
        price_increment=Price(10**-PRICE_PRECISION, precision=PRICE_PRECISION),
        size_increment=Quantity(10**-SIZE_PRECISION, precision=SIZE_PRECISION),
        ts_event=0,
        ts_init=0,
        max_quantity=Quantity(1e9, precision=SIZE_PRECISION),
        min_quantity=Quantity(10**-SIZE_PRECISION, precision=SIZE_PRECISION),
        max_notional=None,
        min_notional=None,
        max_price=Price(100_000_000, precision=PRICE_PRECISION),
        min_price=Price(10**-PRICE_PRECISION, precision=PRICE_PRECISION),
        margin_init=Decimal(0),  # engine skips margin risk checks anyway; blow-up guard is ours
        margin_maint=Decimal(0),
        maker_fee=effective_taker,  # we only send market orders; keep symmetric
        taker_fee=effective_taker,
    )


def bar_type_for(instrument: CryptoPerpetual, timeframe: str) -> BarType:
    spec = {"1h": "1-HOUR", "1d": "1-DAY", "1m": "1-MINUTE"}[timeframe]
    return BarType.from_str(f"{instrument.id}-{spec}-LAST-EXTERNAL")


def bars_for_pair(
    cfg: Config, pair: str, instrument: CryptoPerpetual, start: datetime, end: datetime
) -> list[Bar]:
    """Raw parquet -> nautilus Bar list, window-filtered [start, end).

    Bars are built directly (not via BarDataWrangler: its cython memoryview
    requires writable arrays, which pandas 3 copy-on-write no longer
    provides). ts_event = ts_init = bar CLOSE time in ns, matching our
    storage convention.
    """
    df = (
        store.scan_raw(cfg, pair)
        .filter((pl.col("ts") >= start) & (pl.col("ts") < end))
        .select("ts", "open", "high", "low", "close", "volume")
        .collect()
        .with_columns(ts_ns=pl.col("ts").dt.epoch(time_unit="ns"))
    )
    bar_type = bar_type_for(instrument, cfg.data.timeframe)
    pp, sp = instrument.price_precision, instrument.size_precision
    return [
        Bar(
            bar_type=bar_type,
            open=Price(o, pp),
            high=Price(h, pp),
            low=Price(lo, pp),
            close=Price(c, pp),
            volume=Quantity(v, sp),
            ts_event=t,
            ts_init=t,
        )
        for o, h, lo, c, v, t in zip(
            df["open"], df["high"], df["low"], df["close"], df["volume"], df["ts_ns"], strict=True
        )
    ]


def closes_frame(cfg: Config, start: datetime, end: datetime) -> pl.DataFrame:
    """(ts, pair, close) for mark-to-market equity computation."""
    return (
        store.scan_raw(cfg)
        .filter((pl.col("ts") >= start) & (pl.col("ts") < end))
        .select("ts", "pair", "close")
        .collect()
        .sort("ts", "pair")
    )
