"""Diag — ENTRY-THRESHOLD 0.40 AUDIT (pre-registered gates, S1 spike).

Recon (diag_thresholds) found lowering enter 0.50 -> 0.40 adds ~2.5k trades
and ~+$70k overlay PnL. Before trusting it, audit WHERE the incremental PnL
comes from:

  * INCREMENTAL set = trades opened in the 0.40 variant whose entry-bar
    |score| was in [0.40, 0.50).  Same overlay machinery as sltp_sweep /
    diag_common: close-to-close hysteresis, 13 bps round trip, notional
    100000/30.
  * Gate (a) liquidity concentration: per-pair maker fill share from
    artifacts/backtest/fills.parquet; pairs ranked into quartiles by
    maker share (bottom quartile = worst fill quality).  PASS if <30% of
    incremental trades are in bottom-quartile pairs.  Pairs absent from
    fills.parquet are conservatively treated as bottom quartile.
  * Gate (b) regime slice: incremental net PnL per calendar year (bucketed
    by exit ts; entry-year shown for reference).  PASS if no single year
    contributes >50% of total incremental net PnL AND incremental PnL is
    positive in >=3 of the 4 years 2022-2025 (2026 partial year is its own
    slice, excluded from the >=3-of-4 test).
  * Sanity bound (not a gate): re-price ALL incremental trades at full
    taker (26 bps RT instead of 13 bps).
  * Same audit repeated for enter=0.35 (band [0.35, 0.50)) — informational
    only.

Verdict: GO only if BOTH gates pass for the 0.40 variant.

Deterministic; no randomness. Run:
  cd /Users/bogdan/trading-ml && uv run python analysis/diag_threshold040_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import NOTIONAL, ROUND_TRIP_COST, hysteresis_trades, load_bars

FILLS_PATH = Path(__file__).parent.parent / "artifacts" / "backtest" / "fills.parquet"
TAKER_ROUND_TRIP = 0.0026  # full taker both legs (stress bound, not a gate)
GATE_A_MAX_BOTTOMQ_SHARE = 0.30  # <30% of incremental trades in bottom-quartile pairs
GATE_B_MAX_YEAR_SHARE = 0.50  # no single year >50% of total incremental PnL
GATE_B_YEARS = [2022, 2023, 2024, 2025]  # 2026 partial excluded from >=3-of-4
GATE_B_MIN_POS_YEARS = 3


def maker_share_quartiles() -> tuple[pl.DataFrame, set[str]]:
    """Per-pair maker fill share from the nautilus fills, with quartile rank
    (q=1 bottom = worst fill quality .. q=4 top).  Ties broken by pair name
    so the split is deterministic.  Returns (table, bottom-quartile pairs)."""
    fills = pl.read_parquet(FILLS_PATH)
    per_pair = (
        fills.with_columns(pl.col("instrument").str.split(".").list.first().alias("pair"))
        .group_by("pair")
        .agg(
            n_fills=pl.len(),
            maker_share=(pl.col("liquidity") == "MAKER").mean(),
        )
        .sort("maker_share", "pair")
        .with_row_index("rank")
    )
    n = per_pair.height
    per_pair = per_pair.with_columns(
        (pl.col("rank") * 4 // n + 1).cast(pl.Int8).alias("quartile")
    ).drop("rank")
    bottom = set(per_pair.filter(pl.col("quartile") == 1)["pair"])
    return per_pair, bottom


def incremental_trades(bars: pl.DataFrame, enter_low: float) -> pl.DataFrame:
    """Trades of the enter=enter_low/exit=0.0 variant whose entry-bar |score|
    fell in [enter_low, 0.50) — i.e. entries the production 0.50 gate would
    have skipped.  Tagged with |entry score|, entry/exit year and $ PnL."""
    tr = hysteresis_trades(bars, enter=enter_low, exit_=0.0, allow_short=True)
    return (
        tr.with_columns(pl.col("entry_score").abs().alias("abs_entry_score"))
        .filter((pl.col("abs_entry_score") >= enter_low) & (pl.col("abs_entry_score") < 0.50))
        .with_columns(
            pl.col("entry_ts").dt.year().alias("entry_year"),
            pl.col("exit_ts").dt.year().alias("exit_year"),
            (pl.col("net_ret") * NOTIONAL).alias("pnl_usd"),
            (pl.col("gross_ret") * NOTIONAL).alias("gross_pnl_usd"),
        )
    )


def audit_variant(
    bars: pl.DataFrame,
    enter_low: float,
    quart: pl.DataFrame,
    bottom_pairs: set[str],
    base: pl.DataFrame,
    is_gated: bool,
) -> dict:
    baseline_total = float((base["net_ret"] * NOTIONAL).sum())
    tr_var = hysteresis_trades(bars, enter=enter_low, exit_=0.0, allow_short=True)
    var_total = float((tr_var["net_ret"] * NOTIONAL).sum())
    inc = incremental_trades(bars, enter_low)
    inc_total = float(inc["pnl_usd"].sum())
    n_inc = inc.height

    print("=" * 78)
    print(f"VARIANT enter={enter_low:.2f}/exit=0.0   (band [{enter_low:.2f}, 0.50) = incremental)")
    print("=" * 78)
    print(f"{'':24} {'trades':>8} {'net PnL $':>12} {'~ret%':>7}")
    print(f"{'baseline 0.50/0.0':24} {base.height:>8} {baseline_total:>12,.0f} "
          f"{baseline_total / 100_000 * 100:>6.1f}")
    print(f"{f'variant {enter_low:.2f}/0.0':24} {tr_var.height:>8} {var_total:>12,.0f} "
          f"{var_total / 100_000 * 100:>6.1f}")
    print(f"{'variant - baseline':24} {tr_var.height - base.height:>8} "
          f"{var_total - baseline_total:>12,.0f}")
    print(f"{'incremental band':24} {n_inc:>8} {inc_total:>12,.0f}   "
          f"(win% {float((inc['net_ret'] > 0).mean()) * 100:.1f}, "
          f"avg ${inc_total / max(n_inc, 1):.1f}/trade)")
    resid = (var_total - baseline_total) - inc_total
    print(f"  NOTE substitution effect: band entries preempt baseline entries, so the")
    print(f"  band set is larger than the trade-count delta; band PnL minus true delta")
    print(f"  = {-resid:+,.0f} $ of PnL the band trades DISPLACE from >=0.50 entries.")

    # ---- Gate (a): liquidity concentration --------------------------------
    inc_q = inc.join(quart.select("pair", "maker_share", "quartile"), on="pair", how="left")
    missing = inc_q.filter(pl.col("quartile").is_null())
    # pairs never filled in the prod backtest: unknown fill quality -> worst
    inc_q = inc_q.with_columns(pl.col("quartile").fill_null(1))
    in_bottom = inc_q.filter(pl.col("quartile") == 1)
    share_bottom = in_bottom.height / max(n_inc, 1)
    pnl_bottom = float(in_bottom["pnl_usd"].sum())
    gate_a = share_bottom < GATE_A_MAX_BOTTOMQ_SHARE

    print(f"\nGate (a) — liquidity concentration "
          f"(bottom quartile = {len(bottom_pairs)} pairs by maker fill share)")
    print(f"{'quartile':>9} {'trades':>8} {'trade%':>8} {'PnL $':>12} {'PnL%':>7}")
    for q in (1, 2, 3, 4):
        g = inc_q.filter(pl.col("quartile") == q)
        print(f"{('Q%d%s' % (q, ' (worst)' if q == 1 else '')):>9} {g.height:>8} "
              f"{g.height / max(n_inc, 1) * 100:>7.1f} {float(g['pnl_usd'].sum()):>12,.0f} "
              f"{float(g['pnl_usd'].sum()) / inc_total * 100 if inc_total else 0:>6.1f}")
    if missing.height:
        print(f"  note: {missing.height} trades on {missing['pair'].n_unique()} pairs absent "
              f"from fills.parquet -> counted as bottom quartile (conservative): "
              f"{sorted(set(missing['pair']))}")
    print(f"  incremental trades in bottom-quartile pairs: {share_bottom * 100:.1f}% "
          f"(gate: <{GATE_A_MAX_BOTTOMQ_SHARE * 100:.0f}%)  "
          f"PnL there: {pnl_bottom:+,.0f} $")
    print(f"  Gate (a): {'PASS' if gate_a else 'FAIL'}")

    # ---- Gate (b): regime slice by calendar year --------------------------
    def pnl_by_exit_year(tr: pl.DataFrame) -> dict[int, float]:
        return dict(
            tr.with_columns(pl.col("exit_ts").dt.year().alias("y"))
            .group_by("y")
            .agg((pl.col("net_ret") * NOTIONAL).sum())
            .iter_rows()
        )

    by_exit = {
        int(y): (float(p), int(n))
        for y, p, n in inc.group_by("exit_year")
        .agg(pl.col("pnl_usd").sum(), pl.len().alias("n"))
        .iter_rows()
    }
    by_entry = {int(y): float(p) for y, p in
                inc.group_by("entry_year").agg(pl.col("pnl_usd").sum()).iter_rows()}
    var_y = pnl_by_exit_year(tr_var)
    base_y = pnl_by_exit_year(base)
    years = sorted(set(by_exit) | set(by_entry) | set(var_y) | set(base_y))
    print(f"\nGate (b) — regime slice (calendar years; PnL bucketed by exit ts)")
    print(f"{'year':>6} {'trades':>7} {'band PnL$':>11} {'share%':>7} "
          f"{'entry-yr PnL$':>14} {'var-base $':>11}")
    shares: dict[int, float] = {}
    for y in years:
        pnl_y, n_y = by_exit.get(y, (0.0, 0))
        share = pnl_y / inc_total if inc_total else 0.0
        shares[y] = share
        delta_y = var_y.get(y, 0.0) - base_y.get(y, 0.0)
        tag = "  (partial yr, excl. 3-of-4 test)" if y == 2026 else ""
        print(f"{y:>6} {n_y:>7} {pnl_y:>11,.0f} {share * 100:>6.1f} "
              f"{by_entry.get(y, 0.0):>14,.0f} {delta_y:>11,.0f}{tag}")
    print("  (band PnL/share = pre-registered gate basis; var-base = true marginal")
    print("   effect incl. substitution, informational)")
    max_share_year = max(shares, key=lambda y: shares[y])
    no_dominant = all(s <= GATE_B_MAX_YEAR_SHARE for s in shares.values())
    pos_years = [y for y in GATE_B_YEARS if by_exit.get(y, (0.0, 0))[0] > 0]
    enough_pos = len(pos_years) >= GATE_B_MIN_POS_YEARS
    gate_b = no_dominant and enough_pos
    print(f"  max single-year share: {shares[max_share_year] * 100:.1f}% ({max_share_year}) "
          f"(gate: <={GATE_B_MAX_YEAR_SHARE * 100:.0f}%) -> "
          f"{'ok' if no_dominant else 'VIOLATED'}")
    print(f"  positive years among {GATE_B_YEARS}: {pos_years} "
          f"({len(pos_years)}/4, need >={GATE_B_MIN_POS_YEARS}) -> "
          f"{'ok' if enough_pos else 'VIOLATED'}")
    print(f"  Gate (b): {'PASS' if gate_b else 'FAIL'}")

    # ---- Sanity bound (not a gate): full-taker repricing -------------------
    stressed = float(((inc["gross_ret"] - TAKER_ROUND_TRIP) * NOTIONAL).sum())
    extra_cost = n_inc * (TAKER_ROUND_TRIP - ROUND_TRIP_COST) * NOTIONAL
    delta_stressed = (var_total - baseline_total) - extra_cost
    print(f"\nSanity bound (not a gate): ALL {n_inc} incremental trades at full taker "
          f"({TAKER_ROUND_TRIP * 1e4:.0f} bps RT): {stressed:+,.0f} $ "
          f"(vs {inc_total:+,.0f} $ at {ROUND_TRIP_COST * 1e4:.0f} bps)")
    print(f"  delta view: variant-baseline with band trades at taker: "
          f"{delta_stressed:+,.0f} $ (extra cost {extra_cost:,.0f} $)")

    if is_gated:
        print(f"\nVERDICT (0.40 variant): {'GO' if gate_a and gate_b else 'NO-GO'} "
              f"(gate a {'PASS' if gate_a else 'FAIL'}, gate b {'PASS' if gate_b else 'FAIL'})")
    else:
        print("\n(informational only — no gate applies to this variant)")
    print()
    return {"gate_a": gate_a, "gate_b": gate_b, "inc_total": inc_total,
            "n_inc": n_inc, "stressed": stressed}


def main() -> None:
    bars = load_bars()
    print(f"universe rows: {bars.height}, pairs: {bars['pair'].n_unique()}")
    print(f"cost/trip: {ROUND_TRIP_COST * 1e4:.0f} bps, notional/pair: {NOTIONAL:.0f}\n")

    quart, bottom_pairs = maker_share_quartiles()
    print(f"fills.parquet: {quart['n_fills'].sum()} fills over {quart.height} pairs; "
          f"bottom-quartile maker-share range "
          f"[{quart.filter(pl.col('quartile') == 1)['maker_share'].min() * 100:.1f}%, "
          f"{quart.filter(pl.col('quartile') == 1)['maker_share'].max() * 100:.1f}%], "
          f"top-quartile min "
          f"{quart.filter(pl.col('quartile') == 4)['maker_share'].min() * 100:.1f}%\n")

    base = hysteresis_trades(bars, enter=0.50, exit_=0.0, allow_short=True)

    audit_variant(bars, 0.40, quart, bottom_pairs, base, is_gated=True)
    audit_variant(bars, 0.35, quart, bottom_pairs, base, is_gated=False)


if __name__ == "__main__":
    main()
