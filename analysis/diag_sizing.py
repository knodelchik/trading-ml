"""Diag 3 — SIZING BY SCORE: does conviction sizing add anything?

Takes the EXACT production trade list (enter 0.5 / exit 0.0, shorts) and
re-weights each trade's notional by conviction at entry:
    w = min(|entry_score| / 0.5, 2.0)
Weights are also reported normalized (scaled so the mean weight is 1.0) so
total average deployed notional matches flat sizing — otherwise conviction
sizing just trades bigger on average and the comparison is unfair.

Run: uv run python analysis/diag_sizing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import NOTIONAL, hysteresis_trades, load_bars, max_drawdown_usd


def report(label: str, pnl: np.ndarray) -> None:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())
    print(f"{label:>24} | {len(pnl):>6} {(pnl > 0).mean() * 100:>5.1f} "
          f"{pnl.mean():>8.1f} {pnl.sum():>10.0f} {dd:>9.0f}")


def main() -> None:
    bars = load_bars()
    tr = hysteresis_trades(bars, enter=0.5, exit_=0.0, allow_short=True).sort("exit_ts")

    ret = tr["net_ret"].to_numpy()
    conv = np.abs(tr["entry_score"].to_numpy())
    w_raw = np.minimum(conv / 0.5, 2.0)
    w_norm = w_raw / w_raw.mean()

    print(f"trades: {len(ret)},  mean raw conviction weight: {w_raw.mean():.3f}\n")
    print(f"{'variant':>24} | {'trades':>6} {'win%':>6} {'avg$':>8} {'total$':>10} {'maxDD$':>9}")
    print("-" * 72)
    report("flat (prod)", ret * NOTIONAL)
    report("conviction raw", ret * w_raw * NOTIONAL)
    report("conviction normalized", ret * w_norm * NOTIONAL)

    # is conviction at entry informative about the trade outcome at all?
    rho, p = stats.spearmanr(conv, ret)
    print(f"\nSpearman(|entry_score|, net_ret): rho={rho:+.4f}  p={p:.3g}")

    # quartiles of conviction -> avg trade return
    q = np.quantile(conv, [0.25, 0.5, 0.75])
    bins = np.digitize(conv, q)
    print("\nby conviction quartile (Q1 lowest |score| at entry):")
    print(f"{'quartile':>9} {'n':>6} {'avg |score|':>12} {'avg net ret bps':>16} {'win%':>6}")
    for b in range(4):
        m = bins == b
        print(f"{'Q' + str(b + 1):>9} {m.sum():>6} {conv[m].mean():>12.3f} "
              f"{ret[m].mean() * 1e4:>16.1f} {(ret[m] > 0).mean() * 100:>6.1f}")


if __name__ == "__main__":
    main()
