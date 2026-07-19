"""Diag 5 — SIGNAL DECAY: IC by horizon and by year.

Spearman information coefficient of the OOS score vs forward log-return at
horizons 1h..48h on membership-filtered rows.  Forward closes come from the
raw store joined at ts + h (exact timestamp alignment, so membership gaps
don't fabricate returns).  Reports both the pooled Spearman and the mean
cross-sectional (per-timestamp) IC with its t-stat, plus IC by calendar
year at the 24h label horizon.

Run: uv run python analysis/diag_ic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import load_bars
from tml.config import load_config
from tml.data import store

HORIZONS = [1, 4, 8, 12, 24, 48]


def add_fwd(bars: pl.DataFrame, raw: pl.DataFrame, h: int) -> pl.DataFrame:
    fwd = raw.select(
        (pl.col("ts") - pl.duration(hours=h)).alias("ts"),
        "pair",
        pl.col("close").alias("close_fwd"),
    )
    return (
        bars.join(fwd, on=["ts", "pair"], how="inner")
        .with_columns(fwd_ret=(pl.col("close_fwd") / pl.col("close")).log())
        .drop_nulls("fwd_ret")
    )


def cross_sectional_ic(df: pl.DataFrame) -> tuple[float, float, int]:
    """mean per-timestamp Spearman IC, its t-stat, and #timestamps used."""
    ics = (
        df.group_by("ts")
        .agg(ic=pl.corr(pl.col("score").rank(), pl.col("fwd_ret").rank()), n=pl.len())
        .filter(pl.col("n") >= 10)
        .drop_nulls("ic")
        .filter(pl.col("ic").is_not_nan())
    )
    v = ics["ic"].to_numpy()
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    return float(v.mean()), float(t), len(v)


def main() -> None:
    bars = load_bars()
    cfg = load_config()
    raw = store.scan_raw(cfg).select("ts", "pair", "close").collect()

    print(f"rows: {bars.height}\n")
    print("IC by horizon (all years pooled):")
    print(f"{'horizon':>8} {'rows':>9} {'pooled rho':>11} {'xsec IC':>9} {'t-stat':>8} {'#ts':>7}")
    for h in HORIZONS:
        df = add_fwd(bars, raw, h)
        rho = stats.spearmanr(df["score"].to_numpy(), df["fwd_ret"].to_numpy()).statistic
        mic, t, nts = cross_sectional_ic(df)
        print(f"{h:>7}h {df.height:>9} {rho:>11.4f} {mic:>9.4f} {t:>8.1f} {nts:>7}")

    print("\nIC by year at the 24h label horizon:")
    df24 = add_fwd(bars, raw, 24).with_columns(year=pl.col("ts").dt.year())
    print(f"{'year':>6} {'rows':>9} {'pooled rho':>11} {'xsec IC':>9} {'t-stat':>8}")
    for year in sorted(df24["year"].unique().to_list()):
        d = df24.filter(pl.col("year") == year)
        rho = stats.spearmanr(d["score"].to_numpy(), d["fwd_ret"].to_numpy()).statistic
        mic, t, _ = cross_sectional_ic(d)
        print(f"{year:>6} {d.height:>9} {rho:>11.4f} {mic:>9.4f} {t:>8.1f}")

    print("\nIC by year x horizon (pooled rho):")
    header = f"{'year':>6}" + "".join(f"{str(h) + 'h':>9}" for h in HORIZONS)
    print(header)
    frames = {h: add_fwd(bars, raw, h).with_columns(year=pl.col("ts").dt.year()) for h in HORIZONS}
    years = sorted(df24["year"].unique().to_list())
    for year in years:
        row = f"{year:>6}"
        for h in HORIZONS:
            d = frames[h].filter(pl.col("year") == year)
            rho = stats.spearmanr(d["score"].to_numpy(), d["fwd_ret"].to_numpy()).statistic
            row += f"{rho:>9.4f}"
        print(row)


if __name__ == "__main__":
    main()
