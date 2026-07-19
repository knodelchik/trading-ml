"""H0 — retro-charge the 21-cell threshold sweep to the trials ledger + DSR.

analysis/diag_thresholds.py ran a 7x3 grid (enter in {0.30..0.60} x exit in
{0.0, 0.1, 0.2}) over the hysteresis overlay but never charged those 21
configuration evaluations to artifacts/trials.jsonl.  That is unpaid data
mining: the Deflated Sharpe Ratio's expected-max benchmark must count every
configuration ever looked at.

What this script does, deterministically:
  1. Re-runs the exact 21-cell overlay sweep (same machinery as
     diag_thresholds / diag_common: close-to-close hysteresis trades, 13 bps
     round trip, notional 100000/30, allow_short=True).
  2. Converts each cell's per-trade returns into the ledger's estimator
     convention (`stride_h_nonoverlap_v2`) the same way run_backtest does in
     src/tml/backtest/runner.py: build a net equity curve on the hourly bar
     grid (100_000 initial cash, per-trade PnL booked at exit_ts), subsample
     every H=labels.horizon_bars bars, and take the simple SR of those
     non-overlapping per-period returns.
  3. Appends one record per cell with stage="overlay_sweep_m7", mirroring the
     backtest-stage field schema (config_hash, enter_threshold,
     exit_threshold, execution_delay_bars, sr_per_period,
     net_total_return_pct).  IDEMPOTENT: cells whose
     (enter_threshold, exit_threshold) already exist under this stage are not
     re-appended.
  4. Recomputes the net DSR with the repo's own code
     (tml.validation.metrics.deflated_sharpe), exactly as
     tml.reporting.report does for "DSR (net)": candidate = stride-H
     non-overlapping SR of artifacts/backtest/equity.parquet.
       BEFORE = deflated against the distinct backtest-stage family only
                (what the report currently uses).
       AFTER  = deflated against the combined net-of-cost family
                (backtest + overlay_sweep_m7), deduped by the ledger's
                configuration-identity key (config_hash, enter, exit, delay)
                with the nautilus backtest value winning on key collisions
                (the prod 0.50/0.0 cell is a re-measurement, not a new trial).

The cv-stage (pre-cost) DSR is untouched: the sweep is a net-of-cost,
backtest-family selection and must not enter the pre-cost family.

Run: cd /Users/bogdan/trading-ml && uv run python analysis/charge_threshold_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import NOTIONAL, hysteresis_trades, load_bars

from tml.config import load_config
from tml.data import store
from tml.validation import metrics
from tml.validation.trials import ESTIMATOR, TrialsLedger

STAGE = "overlay_sweep_m7"
ENTER_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
EXIT_GRID = [0.0, 0.1, 0.2]
INITIAL_CASH = 100_000.0


def cell_sr(bars: pl.DataFrame, grid: pl.DataFrame, enter: float, exit_: float,
            horizon: int) -> tuple[float | None, float, int]:
    """(sr_per_period on the stride-H non-overlapping net equity returns,
    net_total_return_pct, n_trades) for one sweep cell — same estimator
    construction as run_backtest's trial record."""
    tr = hysteresis_trades(bars, enter=enter, exit_=exit_, allow_short=True)
    if tr.is_empty():
        return None, 0.0, 0
    pnl = tr.group_by("exit_ts").agg((pl.col("net_ret").sum() * NOTIONAL).alias("pnl"))
    equity = (
        grid.join(pnl, left_on="ts", right_on="exit_ts", how="left")
        .with_columns(pl.col("pnl").fill_null(0.0))
        .with_columns(equity=INITIAL_CASH + pl.col("pnl").cum_sum())
    )
    # mirror runner.py: eq_h = equity["equity"].gather_every(horizon); simple SR
    eq_h = equity["equity"].gather_every(horizon).to_numpy()
    r_h = np.diff(eq_h) / eq_h[:-1]
    sr_h = float(np.nanmean(r_h) / np.nanstd(r_h, ddof=1)) if len(r_h) > 2 else None
    total_ret_pct = float((equity["equity"][-1] / INITIAL_CASH - 1.0) * 100.0)
    return sr_h, round(total_ret_pct, 2), tr.height


def combined_net_history(ledger: TrialsLedger) -> list[float]:
    """Distinct-configuration SRs across BOTH net-of-cost families
    (backtest + overlay_sweep_m7), same dedupe key convention as
    TrialsLedger.sharpe_history; on key collisions the backtest-stage
    (nautilus) value wins — a re-measured config is not a new trial."""
    order = {STAGE: 0, "backtest": 1}
    recs = [
        r for r in ledger.read()
        if r.get("stage") in order
        and r.get("estimator") == ESTIMATOR
        and r.get("sr_per_period") is not None
    ]
    recs.sort(key=lambda r: order[r["stage"]])  # stable: overlay first, backtest overwrites
    seen: dict[tuple, float] = {}
    for r in recs:
        key = (
            r.get("config_hash"),
            r.get("enter_threshold"),
            r.get("exit_threshold"),
            r.get("execution_delay_bars"),
        )
        seen[key] = r["sr_per_period"]
    return list(seen.values())


def net_candidate(cfg) -> tuple[float, int, float, float]:
    """(sr, n_obs, skew, kurt) of the nautilus net equity curve on the
    stride-H non-overlapping subsample — verbatim the report.py recipe."""
    eqdf = pl.read_parquet(store.backtest_dir(cfg) / "equity.parquet")
    eq_h = eqdf.gather_every(cfg.labels.horizon_bars)["equity"].to_numpy()
    r_net = np.diff(eq_h) / eq_h[:-1]
    sr_net = float(np.mean(r_net) / np.std(r_net, ddof=1))
    skew_n, kurt_n = metrics.moments(r_net)
    return sr_net, len(r_net), skew_n, kurt_n


def main() -> None:
    cfg = load_config()
    ledger = TrialsLedger(store.trials_path(cfg))
    horizon = cfg.labels.horizon_bars

    # ---------- DSR BEFORE (current report convention: backtest family only)
    sr_net, n_obs, skew_n, kurt_n = net_candidate(cfg)
    hist_bt = ledger.sharpe_history("backtest")
    before = metrics.deflated_sharpe(sr_net, hist_bt, n_obs, skew_n, kurt_n)
    print(f"net candidate: SR/period={sr_net:.5f}  n_obs={n_obs}  "
          f"skew={skew_n:.2f}  kurt={kurt_n:.1f}")
    print(f"DSR BEFORE (backtest family only, n_trials={int(before['n_trials'])}, "
          f"E[maxSR]={before['sr_benchmark_expected_max']:.5f}): "
          f"{before['dsr']:.4f}\n")

    # ---------- idempotency: which cells are already charged?
    existing = {
        (r.get("enter_threshold"), r.get("exit_threshold"))
        for r in ledger.read()
        if r.get("stage") == STAGE
    }
    cells = [(e, x) for x in EXIT_GRID for e in ENTER_GRID]
    todo = [(e, x) for (e, x) in cells if (e, x) not in existing]
    print(f"sweep cells: {len(cells)} total, {len(existing)} already charged, "
          f"{len(todo)} to append")

    if todo:
        bars = load_bars()
        grid = bars.select("ts").unique().sort("ts")
        print(f"bars: {bars.height} rows, {bars['pair'].n_unique()} pairs, "
              f"grid: {grid.height} hourly ts\n")
        print(f"{'enter':>6} {'exit':>5} | {'trades':>7} {'ret%':>8} {'SR/period':>10}")
        print("-" * 45)
        for enter, exit_ in todo:
            sr_h, ret_pct, n_tr = cell_sr(bars, grid, enter, exit_, horizon)
            ledger.append(
                {
                    "stage": STAGE,
                    "config_hash": cfg.config_hash(),
                    "enter_threshold": enter,
                    "exit_threshold": exit_,
                    "execution_delay_bars": 0,  # overlay fills at the signal bar close
                    "sr_per_period": sr_h,
                    "net_total_return_pct": ret_pct,
                    "source": "retro-charge of analysis/diag_thresholds.py 7x3 sweep (M7 recon)",
                }
            )
            print(f"{enter:>6.2f} {exit_:>5.1f} | {n_tr:>7} {ret_pct:>8.2f} "
                  f"{(sr_h if sr_h is not None else float('nan')):>10.5f}")
        print(f"\nappended {len(todo)} record(s) to {store.trials_path(cfg)}")
    else:
        print("nothing to append — ledger already contains all 21 cells (idempotent)")

    # ---------- DSR AFTER (combined net family, repo DSR code)
    combined = combined_net_history(ledger)
    after = metrics.deflated_sharpe(sr_net, combined, n_obs, skew_n, kurt_n)
    print(f"\nDSR AFTER (backtest + {STAGE} combined, "
          f"n_trials={int(after['n_trials'])}, "
          f"E[maxSR]={after['sr_benchmark_expected_max']:.5f}): "
          f"{after['dsr']:.4f}")
    print(f"delta: {after['dsr'] - before['dsr']:+.4f}")


if __name__ == "__main__":
    main()
