"""Trailing-stop / wide-disaster-stop overlay (research spike, follow-up).

The fixed SL/TP sweep showed tight stops whipsaw on noise and take-profits
cap winners. This tests the two exit rules that specifically target the
user's actual fear — "a winning position reverses and gives it all back" —
WITHOUT the tight-stop whipsaw:

  * trailing stop: let the winner run, but exit if price retraces `trail`
    from its best point since entry (locks in a chunk of a reversed winner).
  * wide disaster stop: a fixed stop so far away (10-15%) it only triggers
    on a genuine crash, never on ordinary noise.

Same controlled setup as sltp_sweep: identical entries/universe/costs, only
the exit rule changes. Intrabar off real high/low. Run:
    uv run python analysis/trailing_sweep.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tml.config import load_config
from tml.data import store
from tml.pipeline.research import dataset

ROUND_TRIP_COST = 0.0013
NOTIONAL = 100_000.0 / 30


def apply_trailing(bars: pl.DataFrame, enter: float, exit_: float, allow_short: bool,
                   trail: float, disaster: float) -> list[float]:
    score = bars["score"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(score)

    rets: list[float] = []
    state = 0
    entry_px = 0.0
    peak = 0.0  # best favorable price since entry (max for long, min for short)
    for i in range(n):
        s = score[i]
        if state == 0:
            if s > enter:
                state, entry_px, peak = 1, close[i], close[i]
            elif allow_short and s < -enter:
                state, entry_px, peak = -1, close[i], close[i]
            continue
        exit_px = None
        if state == 1:
            peak = max(peak, high[i])
            trail_lvl = peak * (1 - trail) if trail > 0 else None
            dis_lvl = entry_px * (1 - disaster) if disaster > 0 else None
            if dis_lvl is not None and low[i] <= dis_lvl:
                exit_px = dis_lvl
            elif trail_lvl is not None and low[i] <= trail_lvl:
                exit_px = trail_lvl
            elif s < exit_:
                exit_px = close[i]
        else:
            peak = min(peak, low[i])
            trail_lvl = peak * (1 + trail) if trail > 0 else None
            dis_lvl = entry_px * (1 + disaster) if disaster > 0 else None
            if dis_lvl is not None and high[i] >= dis_lvl:
                exit_px = dis_lvl
            elif trail_lvl is not None and high[i] >= trail_lvl:
                exit_px = trail_lvl
            elif s > -exit_:
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
    ds = dataset(cfg)
    raw = store.scan_raw(cfg).select("ts", "pair", "high", "low", "close").collect()
    bars = (
        oos.join(raw, on=["ts", "pair"], how="inner")
        .join(ds.select("ts", "pair"), on=["ts", "pair"], how="inner")
        .sort("pair", "ts")
    )

    # (label, trail%, disaster%)
    variants = [
        ("base", 0.0, 0.0),
        ("trail 3%", 0.03, 0.0),
        ("trail 5%", 0.05, 0.0),
        ("trail 8%", 0.08, 0.0),
        ("trail 12%", 0.12, 0.0),
        ("disaster 10%", 0.0, 0.10),
        ("disaster 15%", 0.0, 0.15),
        ("trail 8% + dis 15%", 0.08, 0.15),
    ]
    print(f"universe rows: {bars.height}, pairs: {bars['pair'].n_unique()}\n")
    print(f"{'variant':>20} | {'trades':>7} {'win%':>6} {'avg$':>8} {'total$':>10} {'~ret%':>7} {'maxDD%':>7}")
    print("-" * 78)
    for label, trail, dis in variants:
        rets: list[float] = []
        for (_pk,), g in bars.group_by("pair", maintain_order=True):
            rets.extend(apply_trailing(g, enter, exit_, allow_short, trail, dis))
        arr = np.array(rets)
        pnl = arr * NOTIONAL
        cum = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        dd = float(((cum - peak)).min()) / 100_000 * 100
        print(
            f"{label:>20} | {len(arr):>7} {(arr > 0).mean() * 100:>5.1f} {pnl.mean():>8.1f} "
            f"{pnl.sum():>10.0f} {pnl.sum() / 100_000 * 100:>6.1f} {dd:>7.1f}"
        )


if __name__ == "__main__":
    main()
