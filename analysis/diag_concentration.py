"""Diag 4 — CONCENTRATION: where does the PnL actually come from?

Per-pair contribution (top/bottom 10), per-calendar-month PnL, and PnL split
by BTC-up vs BTC-down months (BTC monthly return from raw closes).  Uses the
production trade list (0.5/0.0, shorts, 13 bps, 100000/30 notional); trades
are stamped to the month of their exit.

Run: uv run python analysis/diag_concentration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import NOTIONAL, hysteresis_trades, load_bars, load_raw_closes


def main() -> None:
    bars = load_bars()
    tr = hysteresis_trades(bars, enter=0.5, exit_=0.0, allow_short=True)
    tr = tr.with_columns(
        pnl=pl.col("net_ret") * NOTIONAL,
        month=pl.col("exit_ts").dt.strftime("%Y-%m"),
    )
    total = tr["pnl"].sum()
    print(f"trades: {tr.height}, total PnL: {total:+.0f} USDT\n")

    # ---- per pair ----
    per_pair = (
        tr.group_by("pair")
        .agg(pnl=pl.col("pnl").sum(), n=pl.len(),
             win=(pl.col("pnl") > 0).mean() * 100)
        .sort("pnl", descending=True)
    )
    print(f"pairs traded: {per_pair.height}")
    print("\ntop 10 pairs by PnL contribution:")
    print(f"{'pair':>14} {'pnl$':>9} {'%of total':>9} {'trades':>7} {'win%':>6}")
    for r in per_pair.head(10).iter_rows(named=True):
        print(f"{r['pair']:>14} {r['pnl']:>9.0f} {r['pnl'] / total * 100:>8.1f}% "
              f"{r['n']:>7} {r['win']:>6.1f}")
    print("\nbottom 10 pairs:")
    for r in per_pair.tail(10).iter_rows(named=True):
        print(f"{r['pair']:>14} {r['pnl']:>9.0f} {r['pnl'] / total * 100:>8.1f}% "
              f"{r['n']:>7} {r['win']:>6.1f}")
    top10_share = per_pair.head(10)["pnl"].sum() / total * 100
    pos_pairs = (per_pair["pnl"] > 0).sum()
    print(f"\ntop-10 pairs = {top10_share:.1f}% of total PnL; "
          f"{pos_pairs}/{per_pair.height} pairs net positive")

    # ---- per calendar month ----
    per_month = tr.group_by("month").agg(pnl=pl.col("pnl").sum(), n=pl.len()).sort("month")

    # BTC monthly return from raw closes (last close per month)
    btc = load_raw_closes("BTCUSDT").with_columns(
        month=pl.col("ts").dt.strftime("%Y-%m")
    )
    btc_m = (
        btc.group_by("month").agg(close=pl.col("close").last()).sort("month")
        .with_columns(btc_ret=pl.col("close") / pl.col("close").shift(1) - 1)
    )
    pm = per_month.join(btc_m.select("month", "btc_ret"), on="month", how="left")

    print("\nper-calendar-month PnL (with BTC month return):")
    print(f"{'month':>8} {'pnl$':>8} {'trades':>7} {'btc%':>7}")
    for r in pm.iter_rows(named=True):
        b = f"{r['btc_ret'] * 100:+.1f}" if r["btc_ret"] is not None else "   n/a"
        print(f"{r['month']:>8} {r['pnl']:>8.0f} {r['n']:>7} {b:>7}")

    mp = pm.drop_nulls("btc_ret")
    up = mp.filter(pl.col("btc_ret") > 0)
    dn = mp.filter(pl.col("btc_ret") <= 0)
    print(f"\nmonths: {mp.height}  (BTC-up {up.height}, BTC-down {dn.height})")
    print(f"strategy PnL in BTC-up months  : {up['pnl'].sum():>+8.0f}  "
          f"(mean {up['pnl'].mean():+.0f}/mo, {(up['pnl'] > 0).mean() * 100:.0f}% positive)")
    print(f"strategy PnL in BTC-down months: {dn['pnl'].sum():>+8.0f}  "
          f"(mean {dn['pnl'].mean():+.0f}/mo, {(dn['pnl'] > 0).mean() * 100:.0f}% positive)")
    corr = float(np.corrcoef(mp["pnl"].to_numpy(), mp["btc_ret"].to_numpy())[0, 1])
    print(f"corr(monthly strategy PnL, BTC monthly return): {corr:+.3f}")

    # long vs short split for regime color
    ls = tr.group_by("side").agg(pnl=pl.col("pnl").sum(), n=pl.len()).sort("side")
    for r in ls.iter_rows(named=True):
        side = "long" if r["side"] == 1 else "short"
        print(f"{side} trades: {r['n']}, PnL {r['pnl']:+.0f}")


if __name__ == "__main__":
    main()
