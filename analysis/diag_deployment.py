"""Diag 1 — DEPLOYMENT: how much of the book is actually working?

Reconstructs the production hysteresis positions per bar (enter |score|>0.5,
exit at 0-crossing, shorts allowed) and reports the distribution of
concurrent open positions over time, plus the implied capital utilization
under equal-notional 1/30 sizing.

Run: uv run python analysis/diag_deployment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import hysteresis_states, load_bars


def main() -> None:
    bars = load_bars()
    st = hysteresis_states(bars, enter=0.5, exit_=0.0, allow_short=True)

    per_ts = (
        st.group_by("ts")
        .agg(
            n_open=(pl.col("state") != 0).sum(),
            n_long=(pl.col("state") == 1).sum(),
            n_short=(pl.col("state") == -1).sum(),
            n_universe=pl.len(),
        )
        .sort("ts")
    )
    n = per_ts["n_open"].to_numpy()
    nl = per_ts["n_long"].to_numpy()
    ns = per_ts["n_short"].to_numpy()

    print(f"bars (hours) in sample: {len(n)},  universe rows: {bars.height}")
    print(f"avg universe size per bar: {per_ts['n_universe'].mean():.1f}\n")

    print("concurrent open positions per bar:")
    print(f"  mean   : {n.mean():.2f}   (long {nl.mean():.2f} / short {ns.mean():.2f})")
    print(f"  median : {np.median(n):.0f}")
    print(f"  p75    : {np.percentile(n, 75):.0f}")
    print(f"  p90    : {np.percentile(n, 90):.0f}")
    print(f"  p99    : {np.percentile(n, 99):.0f}")
    print(f"  max    : {n.max()}")
    print()
    for k in range(0, 6):
        print(f"  fraction of bars with exactly {k} open: {(n == k).mean() * 100:5.1f}%")
    print(f"  fraction of bars with >=6 open        : {(n >= 6).mean() * 100:5.1f}%")
    print()
    util = n / 30.0
    print(f"capital utilization (n_open/30, equal-notional):")
    print(f"  mean {util.mean() * 100:.1f}%  median {np.median(util) * 100:.1f}%  "
          f"p90 {np.percentile(util, 90) * 100:.1f}%  max {util.max() * 100:.1f}%")
    print(f"  fraction of bars fully idle (0 positions): {(n == 0).mean() * 100:.1f}%")

    # how much |score| mass sits just below the entry gate while idle-ish
    yearly = (
        per_ts.with_columns(pl.col("ts").dt.year().alias("year"))
        .group_by("year")
        .agg(mean_open=pl.col("n_open").mean(), idle=(pl.col("n_open") == 0).mean() * 100)
        .sort("year")
    )
    print("\nby year (mean open / % idle bars):")
    for r in yearly.iter_rows(named=True):
        print(f"  {r['year']}: {r['mean_open']:.2f} open,  {r['idle']:.1f}% idle")


if __name__ == "__main__":
    main()
