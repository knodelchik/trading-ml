"""Stop-loss / take-profit overlay experiment (research spike).

A controlled experiment: take the EXACT trade sequence the production model
produced out-of-sample (the hysteresis long/flat/short decisions on
oos.parquet), then re-run only the EXIT rule with different SL/TP levels.
Everything else — entries, universe, costs — is held identical, so any
difference is attributable to SL/TP alone.

Fills are intrabar off the real high/low: a stop fills at its level when the
bar trades through it, a take-profit at its level. When a bar breaches both,
the stop is assumed first (worst case). Costs are a fixed round-trip applied
to every trade uniformly, so the (SL, TP) grid is compared apples-to-apples.
This is a first-order verdict; if it clearly helps we implement it properly
in nautilus + live, if not we reject it cheaply.

Run: uv run python analysis/sltp_sweep.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tml.config import load_config
from tml.data import store
from tml.data.store import pair_key
from tml.pipeline.research import dataset

ROUND_TRIP_COST = 0.0013  # ~13 bps per round trip, uniform across variants
NOTIONAL = 100_000.0 / 30  # equal-notional per pair, matching the backtest


def reconstruct_and_apply(bars: pl.DataFrame, enter: float, exit_: float, allow_short: bool,
                          sl: float, tp: float) -> list[float]:
    """bars: per-pair rows (ts, score, high, low, close) sorted by ts.
    Returns net per-trade returns under the given SL/TP (0 = disabled)."""
    ts = bars["score"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(ts)

    rets: list[float] = []
    state = 0  # 0 flat, +1 long, -1 short
    entry_px = 0.0
    for i in range(n):
        score = ts[i]
        if state == 0:
            if score > enter:
                state, entry_px = 1, close[i]
            elif allow_short and score < -enter:
                state, entry_px = -1, close[i]
            continue
        # position open — check SL/TP intrabar first, then signal exit at close
        exit_px = None
        if state == 1:
            sl_lvl = entry_px * (1 - sl) if sl > 0 else None
            tp_lvl = entry_px * (1 + tp) if tp > 0 else None
            if sl_lvl is not None and low[i] <= sl_lvl:
                exit_px = sl_lvl
            elif tp_lvl is not None and high[i] >= tp_lvl:
                exit_px = tp_lvl
            elif score < exit_:
                exit_px = close[i]
        else:  # short
            sl_lvl = entry_px * (1 + sl) if sl > 0 else None
            tp_lvl = entry_px * (1 - tp) if tp > 0 else None
            if sl_lvl is not None and high[i] >= sl_lvl:
                exit_px = sl_lvl
            elif tp_lvl is not None and low[i] <= tp_lvl:
                exit_px = tp_lvl
            elif score > -exit_:
                exit_px = close[i]
        if exit_px is not None:
            gross = (exit_px / entry_px - 1) if state == 1 else (1 - exit_px / entry_px)
            rets.append(gross - ROUND_TRIP_COST)
            state = 0
    return rets


def main() -> None:
    cfg = load_config()
    enter, exit_ = cfg.backtest.enter_threshold, cfg.backtest.exit_threshold
    allow_short = cfg.backtest.allow_short

    oos = pl.read_parquet(store.oos_path(cfg)).select("ts", "pair", "score")
    ds = dataset(cfg)  # membership-filtered, so pairs match the traded universe
    # bring in high/low/close per (ts, pair) from raw bars
    raw = store.scan_raw(cfg).select("ts", "pair", "high", "low", "close").collect()
    bars = (
        oos.join(raw, on=["ts", "pair"], how="inner")
        .join(ds.select("ts", "pair"), on=["ts", "pair"], how="inner")  # keep only traded rows
        .sort("pair", "ts")
    )

    sl_grid = [0.0, 0.02, 0.03, 0.05, 0.08]
    tp_grid = [0.0, 0.05, 0.10, 0.20]

    print(f"universe rows: {bars.height}, pairs: {bars['pair'].n_unique()}")
    print(f"cost/trip: {ROUND_TRIP_COST * 1e4:.0f}bps, notional/pair: {NOTIONAL:.0f}\n")
    print(f"{'SL':>5} {'TP':>5} | {'trades':>7} {'win%':>6} {'avg$':>8} {'total$':>10} {'~ret%':>7}")
    print("-" * 60)

    baseline_total = None
    for sl in sl_grid:
        for tp in tp_grid:
            all_rets: list[float] = []
            for (_pk,), g in bars.group_by("pair", maintain_order=True):
                all_rets.extend(reconstruct_and_apply(g, enter, exit_, allow_short, sl, tp))
            if not all_rets:
                continue
            arr = np.array(all_rets)
            pnl = arr * NOTIONAL
            total = float(pnl.sum())
            if sl == 0.0 and tp == 0.0:
                baseline_total = total
            win = float((arr > 0).mean()) * 100
            tag = f"{sl * 100:>4.0f}% {tp * 100:>4.0f}%" if (sl or tp) else "  base   "
            print(
                f"{tag} | {len(arr):>7} {win:>5.1f} {pnl.mean():>8.1f} {total:>10.0f} "
                f"{total / 100_000 * 100:>6.1f}"
            )
    if baseline_total is not None:
        print(f"\nbaseline (no SL/TP) aggregate P&L: {baseline_total:+.0f} USDT")


if __name__ == "__main__":
    main()
