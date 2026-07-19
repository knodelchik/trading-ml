"""S2 — NOTIONAL SCALING 1/15 spike (risk decision, pre-registered gates).

Production sizes at equity/30 notional but deploys only ~13% of the book on
average (see diag_deployment).  This spike asks a pure RISK question: is
equity/15 sizing feasible within margin/DD constraints?  Per-trade returns
are unchanged — only the notional multiplier (and, in the capped variant,
which entries get a slot) differs.

Variants simulated with a joint cross-pair event loop (same hysteresis state
machine as diag_common, enter |score|>0.5 / exit at 0-crossing, shorts on,
13 bps round-trip cost, entries/exits at bar close).  Unlike the per-pair
trade-list reconstruction, positions are FLATTENED at a pair's last
in-universe bar and no entry is taken on that bar, matching the production
backtest (src/tml/backtest/runner.py force_close + strategy.on_bar) — a
leverage study must not carry zombie positions on pairs that left the
point-in-time universe.  The freeze-through-gaps convention is kept only as
a machinery cross-check against diag_common.hysteresis_trades.
  (a) notional 100000/15, hard cap of 15 concurrent positions; when there are
      more entry candidates than free slots, best-|score| wins, deterministic
      tie-break by pair name.  Open positions are never force-closed by the
      cap — it only gates new entries.
  (b) notional 100000/15, uncapped (every signal taken).
  baseline: notional 100000/30, uncapped == production overlay.

Per bar we track mark-to-market gross notional (sum of |qty|*price, i.e.
notional * close/entry per open position), MTM equity (realized + unrealized
PnL) and drawdown vs the running equity peak.  Equity denominator for the
gates is the STATIC initial 100k, matching how the gates were registered.

Pre-registered gates (do not move):
  gate A: p99 of gross leverage (gross notional / 100k) <= 1.5x
  gate B: overlay maxDD <= 15% of initial 100k (i.e. >= -15_000 USDT)
  gate C: tail-stress months 2022-11 (FTX), 2024-04, 2025-02 — with baseline
          month PnL b and variant month PnL v:
            b > 0: PASS iff v >= 0.8 * 2 * b   (retains >=80% of the 2x-scaled
                                                baseline contribution)
            b < 0: PASS iff v >= 2.2 * b       (loss no worse than 2.2x)
  sanity (not a gate): total PnL ~ 2x baseline (report the ratio).

Run: uv run python analysis/diag_sizing15.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import ROUND_TRIP_COST, hysteresis_trades, load_bars

EQUITY = 100_000.0
ENTER, EXIT = 0.5, 0.0
STRESS_MONTHS = ["2022-11", "2024-04", "2025-02"]


def simulate(
    bars: pl.DataFrame, notional: float, cap: int | None,
    flatten_membership_exit: bool = True,
) -> pl.DataFrame:
    """Joint cross-pair hysteresis simulation.

    Returns one row per bar: ts, n_open, gross_lev (MTM gross notional / 100k),
    equity (cumulative MTM PnL in USDT), realized (cumulative realized PnL).
    Deterministic: rows processed in (ts, pair) order; capped entry selection
    sorts candidates by (-|score|, pair).

    flatten_membership_exit=True (production semantics): on a pair's last
    in-universe bar (next bar missing or >1h away) any open position is
    closed at that bar's close and no entry is taken.  False reproduces the
    per-pair trade list of diag_common.hysteresis_trades exactly (positions
    frozen through membership gaps) — used only as a cross-check.
    """
    b = bars.sort("pair", "ts").with_columns(
        mexit=(
            pl.col("ts").shift(-1).over("pair").is_null()
            | (pl.col("ts").shift(-1).over("pair") > pl.col("ts") + pl.duration(hours=1))
        )
    ).sort("ts", "pair")
    ts_all = b["ts"].to_numpy()
    pair_all = b["pair"].to_numpy()
    score_all = b["score"].to_numpy()
    close_all = b["close"].to_numpy()
    mexit_all = b["mexit"].to_numpy() if flatten_membership_exit else np.zeros(b.height, bool)

    # bar boundaries: contiguous runs of equal ts
    change = np.flatnonzero(ts_all[1:] != ts_all[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(ts_all)]))

    # open positions: pair -> [side(+1/-1), entry_px, last_close]
    pos: dict[str, list] = {}
    realized = 0.0
    out_nopen, out_lev, out_eq, out_real = [], [], [], []

    for s, e in zip(starts, ends):
        # ---- exits first (frees slots), no same-bar re-entry for that pair ----
        exited: set[str] = set()
        candidates: list[tuple[float, str, int, float]] = []  # (-|score|, pair, side, close)
        for i in range(s, e):
            p = pair_all[i]
            sc = score_all[i]
            cl = close_all[i]
            st = pos.get(p)
            if st is not None:
                st[2] = cl  # refresh mark
                if (
                    mexit_all[i]  # last in-universe bar: flatten, no decision
                    or (st[0] == 1 and sc < EXIT)
                    or (st[0] == -1 and sc > -EXIT)
                ):
                    gross = (cl / st[1] - 1.0) * st[0]
                    realized += (gross - ROUND_TRIP_COST) * notional
                    del pos[p]
                    exited.add(p)
            elif mexit_all[i]:
                continue  # no entries on a membership-exit bar
            elif sc > ENTER:
                candidates.append((-sc, p, 1, cl))
            elif sc < -ENTER:
                candidates.append((sc, p, -1, cl))

        # ---- entries: best |score| first, tie-break by pair name ----
        candidates = [c for c in candidates if c[1] not in exited]
        candidates.sort()
        slots = len(candidates) if cap is None else max(cap - len(pos), 0)
        for _, p, side, cl in candidates[:slots]:
            pos[p] = [side, cl, cl]

        # ---- per-bar marks ----
        unreal = 0.0
        gross_val = 0.0
        for side, entry_px, last_cl in pos.values():
            ratio = last_cl / entry_px
            unreal += (ratio - 1.0) * side * notional
            gross_val += ratio * notional
        out_nopen.append(len(pos))
        out_lev.append(gross_val / EQUITY)
        out_eq.append(realized + unreal)
        out_real.append(realized)

    return pl.DataFrame(
        {
            "ts": b["ts"].gather(starts),
            "n_open": out_nopen,
            "gross_lev": out_lev,
            "equity": out_eq,
            "realized": out_real,
        }
    )


def monthly_pnl(perbar: pl.DataFrame) -> pl.DataFrame:
    """MTM PnL per calendar month = sum of per-bar equity deltas."""
    return (
        perbar.sort("ts")
        .with_columns(delta=pl.col("equity").diff().fill_null(pl.col("equity")))
        .group_by(month=pl.col("ts").dt.strftime("%Y-%m"))
        .agg(pnl=pl.col("delta").sum())
        .sort("month")
    )


def dd_curve(perbar: pl.DataFrame) -> tuple[float, str]:
    """Max drawdown of the per-bar MTM equity curve (peak seeded at 0 —
    equity starts flat at 100k, so a negative first bar is already a DD)."""
    eq = perbar["equity"].to_numpy()
    peak = np.maximum.accumulate(np.maximum(eq, 0.0))
    dd = eq - peak
    i = int(dd.argmin())
    return float(dd[i]), str(perbar["ts"][i])


def lev_stats(perbar: pl.DataFrame) -> dict[str, float]:
    lv = perbar["gross_lev"].to_numpy()
    return {
        "p50": float(np.percentile(lv, 50)),
        "p90": float(np.percentile(lv, 90)),
        "p99": float(np.percentile(lv, 99)),
        "max": float(lv.max()),
    }


def gate_c_row(base: float, var: float) -> tuple[bool, str]:
    if base > 0:
        ok = var >= 0.8 * 2.0 * base
        rule = f"need >= {0.8 * 2.0 * base:+,.0f}"
    elif base < 0:
        ok = var >= 2.2 * base
        rule = f"need >= {2.2 * base:+,.0f}"
    else:
        ok, rule = True, "baseline 0"
    return ok, rule


def main() -> None:
    bars = load_bars()

    # machinery cross-check: with membership-exit flattening OFF, the joint
    # sim must reproduce the per-pair trade list of diag_common exactly
    tr = hysteresis_trades(bars, enter=ENTER, exit_=EXIT, allow_short=True)
    trade_total = float(tr["net_ret"].sum()) * (EQUITY / 30)
    frozen = simulate(bars, EQUITY / 30, cap=None, flatten_membership_exit=False)
    print(f"cross-check (freeze-gap mode) realized PnL: joint sim "
          f"{float(frozen['realized'][-1]):+,.0f} vs per-pair trade list "
          f"{trade_total:+,.0f} (diff {float(frozen['realized'][-1]) - trade_total:+,.2f})\n")

    # gated runs use production semantics: flatten at membership exit
    runs = {
        "baseline 1/30": simulate(bars, EQUITY / 30, cap=None),
        "(a) 1/15 cap15": simulate(bars, EQUITY / 15, cap=15),
        "(b) 1/15 uncapped": simulate(bars, EQUITY / 15, cap=None),
    }

    print(f"{'run':>18} | {'final MTM$':>11} {'maxDD$':>9} {'maxDD%':>7} {'dd date':>17} "
          f"| {'lev p50':>7} {'p90':>6} {'p99':>6} {'max':>6}")
    print("-" * 106)
    stats: dict[str, dict] = {}
    for name, pb in runs.items():
        mdd, mdd_ts = dd_curve(pb)
        ls = lev_stats(pb)
        stats[name] = {"final": float(pb["equity"][-1]), "mdd": mdd, "lev": ls, "pb": pb}
        print(f"{name:>18} | {stats[name]['final']:>11,.0f} {mdd:>9,.0f} "
              f"{mdd / EQUITY * 100:>6.2f}% {mdd_ts[:16]:>17} "
              f"| {ls['p50']:>7.3f} {ls['p90']:>6.3f} {ls['p99']:>6.3f} {ls['max']:>6.3f}")

    base_final = stats["baseline 1/30"]["final"]
    print(f"\nsanity — total MTM PnL ratio vs baseline: "
          f"(a) {stats['(a) 1/15 cap15']['final'] / base_final:.3f}x, "
          f"(b) {stats['(b) 1/15 uncapped']['final'] / base_final:.3f}x  (target ~2x)")

    # ---- stress months ----
    mp = {n: monthly_pnl(s["pb"]) for n, s in stats.items()}
    print(f"\nstress months (MTM monthly PnL, USDT):")
    print(f"{'month':>8} | {'base 1/30':>10} {'gateC need':>10} | {'(a) cap15':>10} {'gateC':>6} "
          f"| {'(b) uncap':>10} {'gateC':>6}")
    print("-" * 78)
    gate_c = {"(a) 1/15 cap15": True, "(b) 1/15 uncapped": True}
    for m in STRESS_MONTHS:
        b = float(mp["baseline 1/30"].filter(pl.col("month") == m)["pnl"][0])
        va = float(mp["(a) 1/15 cap15"].filter(pl.col("month") == m)["pnl"][0])
        vb = float(mp["(b) 1/15 uncapped"].filter(pl.col("month") == m)["pnl"][0])
        need = 0.8 * 2.0 * b if b > 0 else 2.2 * b
        ok_a, _ = gate_c_row(b, va)
        ok_b, _ = gate_c_row(b, vb)
        gate_c["(a) 1/15 cap15"] &= ok_a
        gate_c["(b) 1/15 uncapped"] &= ok_b
        print(f"{m:>8} | {b:>10,.1f} {need:>10,.1f} | {va:>10,.1f} "
              f"{'PASS' if ok_a else 'FAIL':>6} | {vb:>10,.1f} {'PASS' if ok_b else 'FAIL':>6}")

    # ---- gate summary ----
    print(f"\ngates (pre-registered):")
    print(f"{'variant':>18} | {'A: p99 lev<=1.5':>16} | {'B: maxDD<=15%':>14} | {'C: stress':>10}")
    print("-" * 70)
    for name in ["(a) 1/15 cap15", "(b) 1/15 uncapped"]:
        s = stats[name]
        ga = s["lev"]["p99"] <= 1.5
        gb = s["mdd"] >= -0.15 * EQUITY
        gc = gate_c[name]
        print(f"{name:>18} | {s['lev']['p99']:>7.4f} {'PASS' if ga else 'FAIL':>8} "
              f"| {s['mdd'] / EQUITY * 100:>6.2f}% {'PASS' if gb else 'FAIL':>6} "
              f"| {'PASS' if gc else 'FAIL':>10}")

    # ---- extra context: how often does the cap bind? ----
    pb_b = stats["(b) 1/15 uncapped"]["pb"]
    n = pb_b["n_open"].to_numpy()
    print(f"\ncap-15 binding (uncapped n_open): >15 on {(n > 15).mean() * 100:.1f}% of bars, "
          f"p99 {np.percentile(n, 99):.0f}, max {n.max()}")


if __name__ == "__main__":
    main()
