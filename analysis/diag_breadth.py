"""Diag 6 — MISSED BREADTH: near-miss signals while the book sits idle.

On bars where the production policy (0.5/0.0) holds 0-2 positions, counts
OTHER pairs whose |score| is in [0.4, 0.5) — signals the entry gate rejects.
Then a rough upside estimate: enter=0.4 / exit=0.0 with per-position vol
targeting (weight ~ 1/trailing-168h realized vol at entry, capped at 2x,
normalized to mean weight 1 so average deployed notional matches flat).

Run: uv run python analysis/diag_breadth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import NOTIONAL, hysteresis_states, hysteresis_trades, load_bars


def report(label: str, pnl: np.ndarray) -> None:
    cum = np.cumsum(pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    print(f"{label:>34} | {len(pnl):>6} {(pnl > 0).mean() * 100:>5.1f} "
          f"{pnl.mean():>8.1f} {pnl.sum():>10.0f} {dd:>9.0f}")


def main() -> None:
    bars = load_bars()

    # ---- near-miss census under the production policy ----
    st = hysteresis_states(bars, enter=0.5, exit_=0.0, allow_short=True)
    st = st.with_columns(
        near=((pl.col("score").abs() >= 0.4) & (pl.col("score").abs() < 0.5)
              & (pl.col("state") == 0)).cast(pl.Int32)
    )
    per_ts = (
        st.group_by("ts")
        .agg(n_open=(pl.col("state") != 0).sum(), n_near=pl.col("near").sum())
        .sort("ts")
    )
    low = per_ts.filter(pl.col("n_open") <= 2)
    n_near = low["n_near"].to_numpy()
    print(f"bars total: {per_ts.height};  bars holding 0-2 positions: {low.height} "
          f"({low.height / per_ts.height * 100:.1f}%)")
    print(f"near-misses (|score| in [0.4,0.5) on non-held pairs) on those bars:")
    print(f"  mean {n_near.mean():.2f}  median {np.median(n_near):.0f}  "
          f"p90 {np.percentile(n_near, 90):.0f}  max {n_near.max()}")
    print(f"  bars with >=1 near-miss: {(n_near >= 1).mean() * 100:.1f}%   "
          f"total near-miss pair-bars: {n_near.sum()}")

    # ---- rough upside: enter=0.4 with vol targeting ----
    # trailing 168h realized vol of hourly log returns, per pair
    volf = (
        bars.sort("pair", "ts")
        .with_columns(lr=(pl.col("close") / pl.col("close").shift(1).over("pair")).log())
        .with_columns(vol=pl.col("lr").rolling_std(168).over("pair"))
        .select("ts", "pair", "vol")
    )

    tr50 = hysteresis_trades(bars, enter=0.5, exit_=0.0, allow_short=True).sort("exit_ts")
    tr40 = hysteresis_trades(bars, enter=0.4, exit_=0.0, allow_short=True).sort("exit_ts")
    tr40v = tr40.join(volf, left_on=["entry_ts", "pair"], right_on=["ts", "pair"], how="left")

    vol = tr40v["vol"].to_numpy()
    med = np.nanmedian(vol)
    w = np.where(np.isnan(vol), 1.0, np.clip(med / vol, 0.25, 2.0))
    w = w / w.mean()

    print(f"\ntrailing-vol stats at entry (168h std of 1h log rets): "
          f"median {med * 100:.2f}%/h, weights capped [0.25, 2.0], normalized to mean 1")
    print(f"\n{'variant':>34} | {'trades':>6} {'win%':>6} {'avg$':>8} {'total$':>10} {'maxDD$':>9}")
    print("-" * 84)
    report("enter 0.5 flat (prod)", tr50["net_ret"].to_numpy() * NOTIONAL)
    report("enter 0.4 flat", tr40["net_ret"].to_numpy() * NOTIONAL)
    report("enter 0.4 + vol targeting", tr40v["net_ret"].to_numpy() * w * NOTIONAL)

    # vol targeting on the production threshold too, for attribution
    tr50v = tr50.join(volf, left_on=["entry_ts", "pair"], right_on=["ts", "pair"], how="left")
    v50 = tr50v["vol"].to_numpy()
    w50 = np.where(np.isnan(v50), 1.0, np.clip(np.nanmedian(v50) / v50, 0.25, 2.0))
    w50 = w50 / w50.mean()
    report("enter 0.5 + vol targeting", tr50v["net_ret"].to_numpy() * w50 * NOTIONAL)


if __name__ == "__main__":
    main()
