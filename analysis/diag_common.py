"""Shared loader for the diag_* diagnostic scripts.

Same data pattern as analysis/sltp_sweep.py: OOS scores joined with raw
high/low/close, membership-filtered via the research dataset so rows match
the traded point-in-time universe. Optional cache (TML_DIAG_CACHE=<path>)
speeds up running several diag scripts back to back; without it every
script loads fresh and is fully self-contained.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from tml.config import load_config
from tml.data import store
from tml.pipeline.research import dataset

warnings.filterwarnings("ignore")

ROUND_TRIP_COST = 0.0013  # ~13 bps round trip, matching sltp_sweep
NOTIONAL = 100_000.0 / 30  # equal-notional per pair, matching the backtest


def load_bars() -> pl.DataFrame:
    """(ts, pair, score, high, low, close) on membership-filtered rows,
    sorted by (pair, ts)."""
    cache = os.environ.get("TML_DIAG_CACHE")
    if cache and Path(cache).exists():
        return pl.read_parquet(cache)

    cfg = load_config()
    oos = pl.read_parquet(store.oos_path(cfg)).select("ts", "pair", "score")
    ds = dataset(cfg)  # membership-filtered => pairs match the traded universe
    raw = store.scan_raw(cfg).select("ts", "pair", "high", "low", "close").collect()
    bars = (
        oos.join(raw, on=["ts", "pair"], how="inner")
        .join(ds.select("ts", "pair"), on=["ts", "pair"], how="inner")
        .sort("pair", "ts")
    )
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        bars.write_parquet(cache)
    return bars


def load_raw_closes(pair: str = "BTCUSDT") -> pl.DataFrame:
    """(ts, close) for one pair from the raw store (not membership-filtered)."""
    cfg = load_config()
    return (
        store.scan_raw(cfg)
        .filter(pl.col("pair") == pair)
        .select("ts", "close")
        .sort("ts")
        .collect()
    )


def hysteresis_trades(
    bars: pl.DataFrame, enter: float, exit_: float, allow_short: bool = True
) -> pl.DataFrame:
    """Reconstruct the hysteresis trade list per pair (same state machine as
    sltp_sweep with SL/TP disabled). Entry/exit at bar close.

    Returns one row per closed trade:
      pair, side (+1/-1), entry_ts, exit_ts, entry_score, gross_ret, net_ret,
      n_bars (bars held).  Open positions at the end of data are dropped,
      matching sltp_sweep.
    """
    rows: list[tuple] = []
    for (pk,), g in bars.group_by("pair", maintain_order=True):
        score = g["score"].to_numpy()
        close = g["close"].to_numpy()
        ts = g["ts"].to_list()  # python datetimes so polars keeps a Datetime dtype
        n = len(score)
        state = 0
        entry_px = 0.0
        entry_i = -1
        for i in range(n):
            s = score[i]
            if state == 0:
                if s > enter:
                    state, entry_px, entry_i = 1, close[i], i
                elif allow_short and s < -enter:
                    state, entry_px, entry_i = -1, close[i], i
                continue
            if (state == 1 and s < exit_) or (state == -1 and s > -exit_):
                gross = (close[i] / entry_px - 1) * state
                rows.append(
                    (pk, state, ts[entry_i], ts[i], score[entry_i], gross,
                     gross - ROUND_TRIP_COST, i - entry_i)
                )
                state = 0
    return pl.DataFrame(
        rows,
        schema=["pair", "side", "entry_ts", "exit_ts", "entry_score",
                "gross_ret", "net_ret", "n_bars"],
        orient="row",
    )


def hysteresis_states(
    bars: pl.DataFrame, enter: float, exit_: float, allow_short: bool = True
) -> pl.DataFrame:
    """Per-bar position state (+1/0/-1) under hysteresis, one row per
    (ts, pair) in `bars`.  A bar's state is the position held AFTER its
    close is processed (entry bar counts as open, exit bar as flat)."""
    parts: list[pl.DataFrame] = []
    for (pk,), g in bars.group_by("pair", maintain_order=True):
        score = g["score"].to_numpy()
        n = len(score)
        st = np.zeros(n, dtype=np.int8)
        state = 0
        for i in range(n):
            s = score[i]
            if state == 0:
                if s > enter:
                    state = 1
                elif allow_short and s < -enter:
                    state = -1
            elif (state == 1 and s < exit_) or (state == -1 and s > -exit_):
                state = 0
            st[i] = state
        parts.append(g.select("ts", "pair", "score").with_columns(pl.Series("state", st)))
    return pl.concat(parts)


def max_drawdown_usd(trades: pl.DataFrame, notional: float = NOTIONAL) -> float:
    """Rough max drawdown in $ from the cumsum of per-trade PnL ordered by
    exit time (first-order; ignores intratrade marks)."""
    if trades.is_empty():
        return 0.0
    pnl = trades.sort("exit_ts")["net_ret"].to_numpy() * notional
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())
