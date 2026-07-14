"""Skeptic probe A: fill participation vs bar liquidity, per-symbol PnL."""
import polars as pl

ART = "artifacts"

fills = (
    pl.read_parquet(f"{ART}/backtest/fills.parquet")
    .with_columns(
        ts=pl.from_epoch("ts", time_unit="ns").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
        pair=pl.col("instrument").str.split(".").list.first(),
        notional=pl.col("qty") * pl.col("px"),
    )
)
print("fills:", fills.height, "window:", fills["ts"].min(), "->", fills["ts"].max())

pairs = fills["pair"].unique().to_list()
print("n traded pairs:", len(pairs))

# raw bars for traded pairs (ts = bar CLOSE time)
raw = (
    pl.scan_parquet(f"{ART}/data/raw/exchange=binance/**/part.parquet")
    .filter(pl.col("pair").is_in(pairs))
    .select("ts", "pair", "open", "high", "low", "close", "volume")
    .collect()
)
raw = raw.with_columns(quote_vol=pl.col("volume") * pl.col("close"))

# ---- (a) participation: fill notional vs the bar hour's quote volume ----
j = fills.join(raw, on=["ts", "pair"], how="left", suffix="_bar")
print("fills with no matching bar:", j.filter(pl.col("close").is_null()).height)
j = j.with_columns(
    participation=pl.col("notional") / pl.col("quote_vol"),
    qty_vs_quarter=pl.col("qty") / (pl.col("volume") / 4),
)
print("\nparticipation quantiles (fill notional / bar quote volume):")
for q in [0.5, 0.9, 0.99, 0.999, 1.0]:
    print(f"  q{q}: {j['participation'].quantile(q):.6f}")
print("\nfills with participation > 25% of bar quote volume:", j.filter(pl.col("participation") > 0.25).height)
print("fills with participation > 10%:", j.filter(pl.col("participation") > 0.10).height)
print("fills with participation > 1%:", j.filter(pl.col("participation") > 0.01).height)
print("fills with qty > bar_volume/4 (per-tick liquidity):", j.filter(pl.col("qty_vs_quarter") > 1).height)

print("\nWORST 25 fills by participation:")
print(
    j.sort("participation", descending=True)
    .select("ts", "pair", "side", "liquidity", "notional", "quote_vol", "participation", "qty_vs_quarter")
    .head(25)
)

# distribution of bar quote volume for bars where we filled
print("\nmedian bar quote volume at fill hours, by pair (20 thinnest):")
print(
    j.group_by("pair")
    .agg(
        med_bar_qvol=pl.col("quote_vol").median(),
        n_fills=pl.len(),
        med_particip=pl.col("participation").median(),
        max_particip=pl.col("participation").max(),
    )
    .sort("med_bar_qvol")
    .head(20)
)

# ---- (b) per-symbol PnL (flat at end -> sum signed cash flows = realized) ----
pnl = (
    fills.with_columns(
        cash_flow=pl.when(pl.col("side") == "BUY")
        .then(-pl.col("notional"))
        .otherwise(pl.col("notional"))
        - pl.col("commission_usdt"),
        signed_qty=pl.when(pl.col("side") == "BUY").then(pl.col("qty")).otherwise(-pl.col("qty")),
    )
    .group_by("pair")
    .agg(
        pnl=pl.col("cash_flow").sum(),
        fees=pl.col("commission_usdt").sum(),
        gross=pl.col("notional").sum(),
        n=pl.len(),
        end_pos=pl.col("signed_qty").sum(),
        first=pl.col("ts").min(),
        last=pl.col("ts").max(),
    )
    .sort("pnl", descending=True)
)
print("\nresidual positions (should be ~0):", pnl.filter(pl.col("end_pos").abs() > 1e-6).height)
with pl.Config(tbl_rows=100, tbl_cols=10, fmt_str_lengths=20):
    print("\nPER-SYMBOL PnL (all):")
    print(pnl.select("pair", "pnl", "fees", "gross", "n", "first", "last"))

tot = pnl["pnl"].sum()
top5 = pnl.head(5)["pnl"].sum()
top10 = pnl.head(10)["pnl"].sum()
print(f"\ntotal fills PnL: {tot:.0f} | top5 share: {top5/tot:.2%} | top10 share: {top10/tot:.2%}")

# PnL by year
pnl_y = (
    fills.with_columns(
        cash_flow=pl.when(pl.col("side") == "BUY").then(-pl.col("notional")).otherwise(pl.col("notional"))
        - pl.col("commission_usdt"),
        year=pl.col("ts").dt.year(),
    )
    .group_by("year")
    .agg(pnl=pl.col("cash_flow").sum())
    .sort("year")
)
print("\nfills cash-flow PnL by year (unmarked):")
print(pnl_y)
