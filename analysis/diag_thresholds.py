"""Diag 2 — THRESHOLD SENSITIVITY: is 0.5/0.0 near-optimal or conservative?

Sweeps enter in {0.30..0.60} x exit in {0.0, 0.1, 0.2} with the same overlay
machinery as sltp_sweep (close-to-close hysteresis trades, 13 bps round
trip, notional 100000/30).  maxDD is the rough cumsum-of-trades drawdown
ordered by exit time.

Run: uv run python analysis/diag_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import NOTIONAL, hysteresis_trades, load_bars, max_drawdown_usd


def main() -> None:
    bars = load_bars()
    enter_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    exit_grid = [0.0, 0.1, 0.2]

    print(f"universe rows: {bars.height}, pairs: {bars['pair'].n_unique()}")
    print(f"cost/trip: 13bps, notional/pair: {NOTIONAL:.0f}\n")
    print(f"{'enter':>6} {'exit':>5} | {'trades':>7} {'win%':>6} {'avg$':>8} "
          f"{'total$':>10} {'~ret%':>7} {'maxDD$':>9}")
    print("-" * 70)

    for exit_ in exit_grid:
        for enter in enter_grid:
            tr = hysteresis_trades(bars, enter=enter, exit_=exit_, allow_short=True)
            if tr.is_empty():
                continue
            ret = tr["net_ret"].to_numpy()
            pnl = ret * NOTIONAL
            dd = max_drawdown_usd(tr)
            tag = " <- prod" if (enter == 0.50 and exit_ == 0.0) else ""
            print(
                f"{enter:>6.2f} {exit_:>5.1f} | {len(ret):>7} {(ret > 0).mean() * 100:>5.1f} "
                f"{pnl.mean():>8.1f} {pnl.sum():>10.0f} {pnl.sum() / 100_000 * 100:>6.1f} "
                f"{dd:>9.0f}{tag}"
            )
        print()


if __name__ == "__main__":
    main()
