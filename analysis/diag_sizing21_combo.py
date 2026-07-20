"""Pre-registered COMBO sizing trial: equity/21 uncapped AT enter=0.40.

The S2b spike passed all gates at enter=0.50, but 0.40 was adopted since —
more concurrent positions means more gross leverage and deeper drawdowns,
so the 0.50 result does not transfer automatically. Same gates, unchanged:
  A: p99 gross leverage <= 1.5x
  B: overlay maxDD <= 15% of the static 100k denominator
  C: stress months (2022-11, 2024-04, 2025-02) each retain >= 80% of the
     1/30-baseline month PnL scaled by m = 30/21 (loss months: <= (m+0.2)x)
  sanity: total ~= m x baseline.
One ledger trial (stage="sizing_spike_m7", enter 0.4). Run:
    uv run python analysis/diag_sizing21_combo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import diag_sizing15 as d15
from diag_common import load_bars

ENTER = 0.4
M = 30.0 / 21.0
STRESS = ["2022-11", "2024-04", "2025-02"]


def monthly(sim: pl.DataFrame) -> pl.DataFrame:
    return (
        sim.with_columns(month=pl.col("ts").dt.strftime("%Y-%m"))
        .group_by("month", maintain_order=True)
        .agg((pl.col("equity").last() - pl.col("equity").first()).alias("pnl"))
    )


def main() -> None:
    d15.ENTER = ENTER  # combo: sizing gates evaluated at the ADOPTED threshold
    bars = load_bars()
    base = d15.simulate(bars, 100_000.0 / 30, cap=None)
    var = d15.simulate(bars, 100_000.0 / 21, cap=None)

    p99 = float(var["gross_lev"].quantile(0.99))
    eq = var["equity"]
    dd = float((eq - eq.cum_max()).min())
    total_b = float(base["equity"][-1])
    total_v = float(var["equity"][-1])

    mb, mv = monthly(base), monthly(var)
    stress_rows = []
    c_pass = True
    for m in STRESS:
        b = float(mb.filter(pl.col("month") == m)["pnl"][0])
        v = float(mv.filter(pl.col("month") == m)["pnl"][0])
        ok = v >= 0.8 * M * b if b >= 0 else v >= (M + 0.2) * b
        c_pass &= ok
        stress_rows.append((m, b, v, ok))

    a_pass = p99 <= 1.5
    b_pass = dd >= -15_000
    print(f"combo equity/21 @ enter={ENTER}")
    print(f"  gate A p99 lev  : {p99:.4f}  {'PASS' if a_pass else 'FAIL'} (<=1.5)")
    print(f"  gate B maxDD    : {dd:,.0f} USDT ({dd / 1000:.2f}%)  {'PASS' if b_pass else 'FAIL'} (<= -15,000)")
    for m, b, v, ok in stress_rows:
        print(f"  gate C {m}   : base {b:>+9,.0f}  var {v:>+9,.0f}  {'PASS' if ok else 'FAIL'}")
    print(f"  sanity ratio    : {total_v / total_b:.4f} (target {M:.4f})")
    print(f"  totals          : base {total_b:>+11,.0f}  var {total_v:>+11,.0f}")
    verdict = "GO" if (a_pass and b_pass and c_pass) else "NO-GO"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
