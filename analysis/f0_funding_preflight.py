"""F0: funding-fix pre-flight quantification.

The coverage bug in src/tml/features/funding.py (looped static cfg.data.pairs
[22 pairs] instead of pairs_for(cfg) [~131 ever-members]) was just fixed, but
the features parquet on disk is still the OLD build. This script quantifies
what a feature rebuild will change, BEFORE running it:

  1. per-pair funding-partition inventory (exists? min/max settlement ts);
  2. of the fund_known==0 rows in the old build, how many become RECOVERABLE
     (partition exists AND row ts >= partition min ts + 30d z-warmup),
     overall and on membership-filtered (actually-traded) rows;
  3. smoke-test of the fixed add_funding_features on 3 previously-all-zero
     pairs over one recent month;
  4. per-pair top-20 by recoverable rows.

Deterministic: no sampling, fixed pair selection rules, sorted outputs.

Run: cd /Users/bogdan/trading-ml && uv run python analysis/f0_funding_preflight.py
Requires the pre-fix snapshot at SNAPSHOT (ts, pair, fund_known, fund_rate).
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import polars as pl

from tml.config import load_config
from tml.data import store
from tml.data.funding import scan_funding
from tml.data.store import pair_key
from tml.features.funding import add_funding_features
from tml.pipeline.research import dataset
from tml.universe import pairs_for

warnings.filterwarnings("ignore")

SNAPSHOT = (
    "/private/tmp/claude-501/-Users-bogdan/52bd6bd2-e4b3-46cd-a923-ec966868e60e/"
    "scratchpad/fund_known_before_fix.parquet"
)
Z_WARMUP = timedelta(days=30)  # ~90 settlements at 8h — fund_z warm-up
VERIFY_MONTH = "2026-06"  # recent full month for the smoke test
N_VERIFY = 3


def partition_inventory(cfg) -> pl.DataFrame:
    """One row per ever-member pair: has_partition, n settlements, min/max ts."""
    rows = []
    for pair in pairs_for(cfg):
        pk = pair_key(pair)
        lf = scan_funding(cfg, pair)
        if lf is None:
            rows.append((pk, False, 0, None, None))
            continue
        s = lf.select(
            n=pl.len(), min_ts=pl.col("ts").min(), max_ts=pl.col("ts").max()
        ).collect()
        rows.append((pk, True, s["n"][0], s["min_ts"][0], s["max_ts"][0]))
    return pl.DataFrame(
        rows,
        schema={
            "pair": pl.String,
            "has_partition": pl.Boolean,
            "n_settlements": pl.Int64,
            "fund_min_ts": pl.Datetime("us", "UTC"),
            "fund_max_ts": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


def main() -> None:
    cfg = load_config()

    # ---- step 1: partition inventory -------------------------------------
    inv = partition_inventory(cfg)
    n_missing = (~inv["has_partition"]).sum()
    missing = inv.filter(~pl.col("has_partition"))["pair"].sort().to_list()
    print("## Step 1 — funding partition inventory")
    print(f"ever-member pairs (pairs_for): {inv.height}")
    print(f"partitions on disk: {inv['has_partition'].sum()}   missing: {n_missing}")
    print(f"pairs with NO partition (stay fund_known=0 after fix): {missing}")
    print()

    # ---- step 2: recoverable rows -----------------------------------------
    snap = pl.read_parquet(SNAPSHOT)
    total_rows = snap.height
    zero = snap.filter(pl.col("fund_known") == 0)
    n_zero = zero.height

    zero = zero.join(inv, on="pair", how="left").with_columns(
        recoverable=(
            pl.col("has_partition")
            & (pl.col("ts") >= pl.col("fund_min_ts") + Z_WARMUP)
        ).fill_null(False),
        recoverable_any=(  # fund_known flips from the FIRST settlement (no warmup)
            pl.col("has_partition") & (pl.col("ts") >= pl.col("fund_min_ts"))
        ).fill_null(False),
    )
    n_rec = int(zero["recoverable"].sum())
    n_rec_any = int(zero["recoverable_any"].sum())

    print("## Step 2 — recoverable fund_known==0 rows")
    print(f"snapshot rows: {total_rows:,}   fund_known==0: {n_zero:,} ({n_zero / total_rows:.1%})")
    print(
        f"RECOVERABLE (partition exists AND ts >= min_ts + 30d warmup): "
        f"{n_rec:,} = {n_rec / total_rows:.1%} of all rows, "
        f"{n_rec / n_zero:.1%} of fund_known==0 rows"
    )
    print(
        f"  (without warmup, i.e. ts >= first settlement — fund_known itself flips: "
        f"{n_rec_any:,} = {n_rec_any / total_rows:.1%} of all rows)"
    )
    print()

    # membership-filtered (traded) rows — this is what the model actually sees
    ds_keys = dataset(cfg).select("ts", "pair")
    n_member = ds_keys.height
    member_zero = zero.join(ds_keys, on=["ts", "pair"], how="inner")
    n_member_zero = member_zero.height
    n_member_rec = int(member_zero["recoverable"].sum())
    n_member_rec_any = int(member_zero["recoverable_any"].sum())

    print("## Step 2b — membership-filtered (traded) rows")
    print(f"membership-filtered dataset rows: {n_member:,}")
    print(
        f"of which fund_known==0 in old build: {n_member_zero:,} "
        f"({n_member_zero / n_member:.1%})"
    )
    print(
        f"RECOVERABLE on traded rows: {n_member_rec:,} = "
        f"{n_member_rec / n_member:.1%} of traded rows, "
        f"{(n_member_rec / n_member_zero if n_member_zero else 0):.1%} of traded fund_known==0 rows"
    )
    print(
        f"  (without warmup: {n_member_rec_any:,} = "
        f"{n_member_rec_any / n_member:.1%} of traded rows)"
    )
    print()

    # ---- step 4 (computed here): per-pair top-20 by recoverable rows -------
    per_pair = (
        zero.group_by("pair")
        .agg(
            recoverable=pl.col("recoverable").sum(),
            zero_rows=pl.len(),
        )
        .join(
            member_zero.group_by("pair").agg(
                member_recoverable=pl.col("recoverable").sum()
            ),
            on="pair",
            how="left",
        )
        .with_columns(pl.col("member_recoverable").fill_null(0))
        .join(inv.select("pair", "has_partition", "fund_min_ts"), on="pair", how="left")
        .sort(["recoverable", "pair"], descending=[True, False])
    )
    print("## Step 4 — top-20 pairs by recoverable rows")
    print(f"{'pair':<14} {'recoverable':>11} {'traded_rec':>10} {'zero_rows':>9}  fund_min_ts")
    for r in per_pair.head(20).iter_rows(named=True):
        print(
            f"{r['pair']:<14} {r['recoverable']:>11,} {r['member_recoverable']:>10,} "
            f"{r['zero_rows']:>9,}  {r['fund_min_ts']}"
        )
    n_pairs_rec = per_pair.filter(pl.col("recoverable") > 0).height
    print(f"\npairs with >0 recoverable rows: {n_pairs_rec}")
    print()

    # ---- step 3: verify the fix on a small slice ---------------------------
    # deterministic pick: pairs whose snapshot rows were 100% fund_known==0,
    # that have a partition, ranked by recoverable rows (desc), then name.
    all_zero_pairs = (
        snap.group_by("pair")
        .agg(known_share=pl.col("fund_known").mean())
        .filter(pl.col("known_share") == 0.0)["pair"]
        .to_list()
    )
    verify_pairs = (
        per_pair.filter(
            pl.col("pair").is_in(all_zero_pairs) & pl.col("has_partition")
        )
        .sort(["recoverable", "pair"], descending=[True, False])
        .head(N_VERIFY)["pair"]
        .to_list()
    )
    print("## Step 3 — smoke test of fixed add_funding_features")
    print(f"pairs (previously fund_known==0 everywhere, partition on disk): {verify_pairs}")

    bars = (
        store.scan_raw(cfg)
        .select("ts", "pair", "close")
        .filter(
            pl.col("pair").is_in(verify_pairs)
            & (pl.col("ts").dt.strftime("%Y-%m") == VERIFY_MONTH)
        )
        .collect()
    )
    out = add_funding_features(bars, cfg)
    chk = (
        out.group_by("pair")
        .agg(
            rows=pl.len(),
            known_share=pl.col("fund_known").mean(),
            nonzero_rate_share=(pl.col("fund_rate") != 0).mean(),
            rate_min=pl.col("fund_rate").min(),
            rate_max=pl.col("fund_rate").max(),
        )
        .sort("pair")
    )
    print(f"month: {VERIFY_MONTH}, bars: {bars.height}")
    print(f"{'pair':<14} {'rows':>5} {'known':>6} {'rate!=0':>8} {'rate_min':>12} {'rate_max':>12}")
    ok = True
    for r in chk.iter_rows(named=True):
        print(
            f"{r['pair']:<14} {r['rows']:>5} {r['known_share']:>6.3f} "
            f"{r['nonzero_rate_share']:>8.3f} {r['rate_min']:>12.6f} {r['rate_max']:>12.6f}"
        )
        if r["known_share"] < 1.0 or r["nonzero_rate_share"] == 0.0:
            ok = False
    print(f"smoke test: {'PASS' if ok else 'FAIL'} "
          f"(expect known=1.000 and some non-zero rates on every pair)")

    # full inventory dump for reference (sorted, deterministic)
    print("\n## Appendix — full partition inventory (sorted by pair)")
    print(f"{'pair':<14} {'part':>5} {'n_settle':>8}  min_ts .. max_ts")
    for r in inv.sort("pair").iter_rows(named=True):
        rng = (
            f"{r['fund_min_ts']} .. {r['fund_max_ts']}"
            if r["has_partition"]
            else "-"
        )
        print(f"{r['pair']:<14} {str(r['has_partition']):>5} {r['n_settlements']:>8}  {rng}")


if __name__ == "__main__":
    main()
