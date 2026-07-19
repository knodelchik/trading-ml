"""S2b — NOTIONAL SCALING 1/21 spike (risk decision, pre-registered gates).

The equity/15 spike (analysis/diag_sizing15.py) FAILED its pre-registered
gates: capped-15 broke gate B (maxDD -16.68% vs -15%) and a stress month;
uncapped-15 broke gate A (p99 leverage 1.53 vs 1.5).  Post-hoc arithmetic
suggested equity/21 uncapped (multiplier m = 30/21 ~ 1.43x) clears every
gate — but post-hoc is not a result.  This spike PRE-REGISTERS equity/21
uncapped as ONE new trial with the SAME gates, unchanged except that the
scale factor 2x is replaced by the variant's own multiplier m:

  gate A: p99 of gross leverage (gross notional / 100k) <= 1.5x
  gate B: overlay maxDD <= 15% of the STATIC initial 100k (>= -15_000 USDT)
  gate C: tail-stress months 2022-11 (FTX), 2024-04, 2025-02 — with baseline
          month PnL b and variant month PnL v, m = 30/21:
            b > 0: PASS iff v >= 0.8 * m * b   (retains >=80% of the m-scaled
                                                baseline contribution)
            b < 0: PASS iff v >= (m + 0.2) * b (loss no worse than (m+0.2)x)
  sanity (not a gate): total PnL ~ m x baseline (report the ratio).

Machinery is diag_sizing15's simulate() imported UNCHANGED — same joint
cross-pair hysteresis event loop (enter |score|>0.5 / exit at 0-crossing,
shorts on, 13 bps round trip, entries/exits at bar close), same
membership-exit flattening, same MTM gross-notional / equity / drawdown
marks — parameterized at notional 100000/21, uncapped.  Bars are loaded
ONCE at script start; nothing under artifacts/backtest/* is read, so a
concurrently running nautilus backtest cannot corrupt this analysis.

The trial is charged to artifacts/trials.jsonl exactly once (idempotent),
reusing the ledger convention of analysis/charge_threshold_sweep.py:
stage="sizing_spike_m7", estimator stride_h_nonoverlap_v2 computed on the
VARIANT per-bar MTM equity curve (100k initial cash, stride-H
non-overlapping simple SR, H = labels.horizon_bars).  The record carries
notional_divisor=21 / capped=False so any future combined-family DSR can
dedupe sizing trials by their true configuration identity.

Run: cd /Users/bogdan/trading-ml && uv run python analysis/diag_sizing21.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import hysteresis_trades, load_bars
from diag_sizing15 import (
    ENTER,
    EQUITY,
    EXIT,
    STRESS_MONTHS,
    dd_curve,
    lev_stats,
    monthly_pnl,
    simulate,
)

from tml.config import load_config
from tml.data import store
from tml.validation.trials import TrialsLedger

STAGE = "sizing_spike_m7"
DIVISOR = 21
M = 30.0 / DIVISOR  # PnL multiplier vs the 1/30 production baseline
INITIAL_CASH = 100_000.0


def gate_c_row(base: float, var: float) -> tuple[bool, float]:
    """(pass, threshold) for one stress month under the m-scaled rule."""
    if base > 0:
        return var >= 0.8 * M * base, 0.8 * M * base
    if base < 0:
        return var >= (M + 0.2) * base, (M + 0.2) * base
    return True, 0.0


def variant_sr(perbar: pl.DataFrame, horizon: int) -> tuple[float | None, float]:
    """(sr_per_period, net_total_return_pct) on the variant MTM equity curve —
    same estimator construction as charge_threshold_sweep.cell_sr / runner.py:
    100k initial cash, stride-H non-overlapping simple SR."""
    equity = perbar.sort("ts").with_columns(equity=INITIAL_CASH + pl.col("equity"))
    eq_h = equity["equity"].gather_every(horizon).to_numpy()
    r_h = np.diff(eq_h) / eq_h[:-1]
    sr_h = float(np.nanmean(r_h) / np.nanstd(r_h, ddof=1)) if len(r_h) > 2 else None
    total_ret_pct = float((equity["equity"][-1] / INITIAL_CASH - 1.0) * 100.0)
    return sr_h, round(total_ret_pct, 2)


def main() -> None:
    cfg = load_config()
    # the trial identity is the PRODUCTION overlay (enter 0.5 / exit 0.0) at
    # notional/21 — pin the hashed config to those thresholds so a concurrent
    # threshold experiment editing config/research.toml cannot leak into the
    # ledger record (the simulation itself is hardwired to ENTER/EXIT anyway)
    if (cfg.backtest.enter_threshold, cfg.backtest.exit_threshold) != (ENTER, EXIT):
        print(f"NOTE: working config has thresholds "
              f"{cfg.backtest.enter_threshold}/{cfg.backtest.exit_threshold}; "
              f"this spike evaluates and records the overlay at {ENTER}/{EXIT}\n")
    cfg_pinned = cfg.model_copy(update={
        "backtest": cfg.backtest.model_copy(
            update={"enter_threshold": ENTER, "exit_threshold": EXIT}
        )
    })
    bars = load_bars()  # single read at script start; never touches artifacts/backtest/*

    # machinery cross-check (as in diag_sizing15): with membership-exit
    # flattening OFF, the joint sim must reproduce the per-pair trade list
    tr = hysteresis_trades(bars, enter=ENTER, exit_=EXIT, allow_short=True)
    trade_total = float(tr["net_ret"].sum()) * (EQUITY / DIVISOR)
    frozen = simulate(bars, EQUITY / DIVISOR, cap=None, flatten_membership_exit=False)
    print(f"cross-check (freeze-gap mode) realized PnL: joint sim "
          f"{float(frozen['realized'][-1]):+,.0f} vs per-pair trade list "
          f"{trade_total:+,.0f} (diff {float(frozen['realized'][-1]) - trade_total:+,.2f})\n")

    # gated runs use production semantics: flatten at membership exit
    runs = {
        "baseline 1/30": simulate(bars, EQUITY / 30, cap=None),
        "1/21 uncapped": simulate(bars, EQUITY / DIVISOR, cap=None),
    }

    print(f"{'run':>14} | {'final MTM$':>11} {'maxDD$':>9} {'maxDD%':>7} {'dd date':>17} "
          f"| {'lev p50':>7} {'p90':>6} {'p99':>6} {'max':>6}")
    print("-" * 102)
    stats: dict[str, dict] = {}
    for name, pb in runs.items():
        mdd, mdd_ts = dd_curve(pb)
        ls = lev_stats(pb)
        stats[name] = {"final": float(pb["equity"][-1]), "mdd": mdd, "lev": ls, "pb": pb}
        print(f"{name:>14} | {stats[name]['final']:>11,.0f} {mdd:>9,.0f} "
              f"{mdd / EQUITY * 100:>6.2f}% {mdd_ts[:16]:>17} "
              f"| {ls['p50']:>7.3f} {ls['p90']:>6.3f} {ls['p99']:>6.3f} {ls['max']:>6.3f}")

    base_final = stats["baseline 1/30"]["final"]
    var = stats["1/21 uncapped"]
    print(f"\nsanity — total MTM PnL ratio vs baseline: "
          f"{var['final'] / base_final:.3f}x  (target ~{M:.3f}x)")

    # ---- gate C: stress months ----
    mp = {n: monthly_pnl(s["pb"]) for n, s in stats.items()}
    print(f"\nstress months (MTM monthly PnL, USDT, m={M:.4f}):")
    print(f"{'month':>8} | {'base 1/30':>10} {'gateC need':>10} | {'1/21 uncap':>10} {'gateC':>6}")
    print("-" * 56)
    gate_c = True
    for m in STRESS_MONTHS:
        b = float(mp["baseline 1/30"].filter(pl.col("month") == m)["pnl"][0])
        v = float(mp["1/21 uncapped"].filter(pl.col("month") == m)["pnl"][0])
        ok, need = gate_c_row(b, v)
        gate_c &= ok
        print(f"{m:>8} | {b:>10,.1f} {need:>10,.1f} | {v:>10,.1f} {'PASS' if ok else 'FAIL':>6}")

    # ---- gate summary + verdict (AND-only, pre-registered) ----
    ga = var["lev"]["p99"] <= 1.5
    gb = var["mdd"] >= -0.15 * EQUITY
    verdict = "GO" if (ga and gb and gate_c) else "NO-GO"
    print("\ngates (pre-registered):")
    print(f"  A: p99 gross lev <= 1.5x   : {var['lev']['p99']:.4f}  {'PASS' if ga else 'FAIL'}")
    print(f"  B: maxDD <= 15% of 100k    : {var['mdd'] / EQUITY * 100:.2f}%  {'PASS' if gb else 'FAIL'}")
    print(f"  C: stress months retain    : {'PASS' if gate_c else 'FAIL'}")
    print(f"\nVERDICT: {verdict}")

    # ---- charge the trial to the ledger (idempotent, ONE trial only) ----
    ledger = TrialsLedger(store.trials_path(cfg))
    already = [
        r for r in ledger.read()
        if r.get("stage") == STAGE
        and r.get("notional_divisor") == DIVISOR
        and r.get("capped") is False
    ]
    if already:
        print(f"\nledger: trial already charged under stage={STAGE} "
              f"(notional_divisor={DIVISOR}, uncapped) — nothing appended (idempotent)")
    else:
        sr_h, ret_pct = variant_sr(var["pb"], cfg.labels.horizon_bars)
        ledger.append(
            {
                "stage": STAGE,
                "config_hash": cfg_pinned.config_hash(),
                "enter_threshold": ENTER,
                "exit_threshold": EXIT,
                "execution_delay_bars": 0,  # overlay fills at the signal bar close
                "notional_divisor": DIVISOR,
                "capped": False,
                "sr_per_period": sr_h,
                "net_total_return_pct": ret_pct,
                "gates_pass": bool(ga and gb and gate_c),
                "source": "analysis/diag_sizing21.py pre-registered equity/21 uncapped sizing spike (S2b)",
            }
        )
        print(f"\nledger: appended 1 trial (stage={STAGE}, sr_per_period="
              f"{(sr_h if sr_h is not None else float('nan')):.5f}, "
              f"net_total_return_pct={ret_pct:.2f}) to {store.trials_path(cfg)}")


if __name__ == "__main__":
    main()
