"""Research report: every number traceable to an artifact file.

Honesty rules encoded here:
  * DSR refuses to compute without the trials ledger (selection-bias guard).
  * PSR/DSR use a NON-overlapping (stride = label horizon) subsample of the
    signal-portfolio series — overlapping H-bar labels would inflate n_obs.
  * The pre-cost "signal Sharpe" is labeled as research-grade; net-of-cost
    truth comes only from the nautilus backtest artifacts when present.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from tml.config import Config
from tml.data import store
from tml.validation import metrics
from tml.validation.trials import TrialsLedger

HOURS_PER_YEAR = 24 * 365


def portfolio_returns(oos: pl.DataFrame) -> pl.DataFrame:
    """Equal-weight sign(score) portfolio, one (vol-scaled, H-bar) return per ts."""
    return (
        oos.with_columns(r=pl.col("y_true") * pl.col("score").sign())
        .group_by("ts")
        .agg(pl.col("r").mean())
        .sort("ts")
    )


def compute_cv_metrics(cfg: Config) -> dict:
    oos = pl.read_parquet(store.oos_path(cfg))
    h = cfg.labels.horizon_bars

    per_fold = []
    for (fold_id,), fold_df in oos.group_by("fold_id", maintain_order=True):
        s = fold_df["score"].to_numpy()
        y = fold_df["y_true"].to_numpy()
        per_fold.append(
            {
                "fold_id": int(fold_id),
                "n": fold_df.height,
                "ts_start": cast(datetime, fold_df["ts"].min()).isoformat(),
                "ts_end": cast(datetime, fold_df["ts"].max()).isoformat(),
                "ic": metrics.spearman_ic(s, y),
                "hit_rate": metrics.hit_rate(s, y),
                "decile_spread": metrics.decile_spread(s, y),
            }
        )

    pooled_ic = metrics.spearman_ic(oos["score"].to_numpy(), oos["y_true"].to_numpy())
    ic_ts, ic_series = metrics.per_ts_ic(
        oos["ts"].to_numpy(), oos["score"].to_numpy(), oos["y_true"].to_numpy()
    )
    ic_nw_t = metrics.newey_west_tstat(ic_series, lag=h)

    # per-member diagnostics: which model carries the signal?
    y_all = oos["y_true"].to_numpy()
    members = {}
    for c in [c for c in oos.columns if c.startswith("score_")] + ["score"]:
        s = oos[c].to_numpy()
        _, ics = metrics.per_ts_ic(oos["ts"].to_numpy(), s, y_all)
        members[c.removeprefix("score_") if c != "score" else "ensemble(mean)"] = {
            "pooled_ic": metrics.spearman_ic(s, y_all),
            "nw_tstat": metrics.newey_west_tstat(ics, lag=h),
        }

    port = portfolio_returns(oos)
    # non-overlapping subsample for distribution-sensitive stats
    port_no = port.gather_every(h)["r"].to_numpy()
    sr_pp = float(np.mean(port_no) / np.std(port_no, ddof=1)) if len(port_no) > 2 else float("nan")
    skew, kurt = metrics.moments(port_no)
    psr = metrics.probabilistic_sharpe(sr_pp, 0.0, len(port_no), skew, kurt)

    ledger = TrialsLedger(store.trials_path(cfg))
    if not ledger.read():
        raise RuntimeError(
            "trials.jsonl is missing or empty — DSR cannot be honest without the full "
            "history of evaluated configurations. Run `tml cv` (it appends to the ledger)."
        )
    # deflate the pre-cost candidate against the pre-cost (cv) trial family
    # only — mixing in net-of-cost backtest SRs would make the expected-max
    # benchmark measure the cost gap instead of selection breadth
    dsr = metrics.deflated_sharpe(sr_pp, ledger.sharpe_history("cv"), len(port_no), skew, kurt)

    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_hash": cfg.config_hash(),
        "n_oos": oos.height,
        "n_folds": len(per_fold),
        "pooled_ic": pooled_ic,
        "per_ts_ic_mean": float(np.nanmean(ic_series)) if len(ic_series) else float("nan"),
        "ic_newey_west_tstat_lag_h": ic_nw_t,
        "members": members,
        "per_fold": per_fold,
        "signal_portfolio": {
            "note": (
                "pre-cost, vol-scaled, H-bar-overlapping research metric; "
                "PSR/DSR computed on the stride-H non-overlapping subsample"
            ),
            "sr_per_period_nonoverlap": sr_pp,
            "sr_annualized_nonoverlap": sr_pp * math.sqrt(HOURS_PER_YEAR / h)
            if not math.isnan(sr_pp)
            else float("nan"),
            "n_nonoverlap_obs": int(len(port_no)),
            "skew": skew,
            "kurtosis_pearson": kurt,
            "psr_vs_zero": psr,
            **dsr,
        },
    }
    path = store.reports_dir(cfg) / "cv_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def _plot_ic(per_fold: list[dict], path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ids = [f["fold_id"] for f in per_fold]
    ics = [f["ic"] for f in per_fold]
    ax.bar(ids, ics, color=["tab:green" if v and v > 0 else "tab:red" for v in ics])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("fold")
    ax.set_ylabel("Spearman IC")
    ax.set_title("Out-of-sample IC per walk-forward fold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_signal_curve(oos: pl.DataFrame, path) -> None:
    port = portfolio_returns(oos)
    cum = port.with_columns(cum=pl.col("r").cum_sum())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(cum["ts"].to_numpy(), cum["cum"].to_numpy(), lw=1)
    ax.set_title("Cumulative signal-portfolio return (pre-cost, vol-scaled units)")
    ax.set_ylabel("cumulative vol-scaled return")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_equity(equity: pl.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ts = equity["ts"].to_numpy()
    eq = equity["equity"].to_numpy()
    axes[0].plot(ts, eq, lw=1)
    axes[0].set_ylabel("equity, USDT")
    axes[0].set_title("Backtest equity (net of fees and slippage)")
    peak = np.maximum.accumulate(eq)
    axes[1].fill_between(ts, (eq - peak) / peak * 100, 0, color="tab:red", alpha=0.5)
    axes[1].set_ylabel("drawdown, %")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _trials_disclosure(cfg: Config) -> str:
    ledger = TrialsLedger(store.trials_path(cfg))
    cv_raw, cv_distinct = ledger.counts("cv")
    bt_raw, bt_distinct = ledger.counts("backtest")
    return (
        "- Trials ledger (`artifacts/trials.jsonl`, append-only): "
        f"cv {cv_distinct} distinct configuration(s) over {cv_raw} record(s); "
        f"backtest {bt_distinct} distinct over {bt_raw}. DSR benchmarks use distinct "
        "same-stage, same-estimator trials only; idempotent re-runs are not new trials."
    )


def build_report(cfg: Config) -> int:
    cvm = compute_cv_metrics(cfg)
    oos = pl.read_parquet(store.oos_path(cfg))
    rdir = store.reports_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    _plot_ic(cvm["per_fold"], rdir / "ic_per_fold.png")
    _plot_signal_curve(oos, rdir / "signal_curve.png")

    bt_stats = None
    stats_path = store.backtest_dir(cfg) / "stats.json"
    if stats_path.exists():
        bt_stats = json.loads(stats_path.read_text())
        equity_path = store.backtest_dir(cfg) / "equity.parquet"
        if equity_path.exists():
            _plot_equity(pl.read_parquet(equity_path), rdir / "equity.png")

    sp = cvm["signal_portfolio"]
    lines = [
        "# Research report",
        "",
        f"- generated: {cvm['generated_at']}",
        f"- config hash: `{cvm['config_hash']}`",
        f"- OOS predictions: {cvm['n_oos']} rows, {cvm['n_folds']} folds",
        f"- data: {cfg.data.exchange} {cfg.data.timeframe}, {len(cfg.data.pairs)} pairs, since {cfg.data.since:%Y-%m-%d}",
        "",
        "## Predictive quality (out-of-sample)",
        "",
        f"- pooled Spearman IC: **{cvm['pooled_ic']:+.4f}**",
        f"- mean per-timestamp cross-sectional IC: {cvm['per_ts_ic_mean']:+.4f}",
        f"- Newey-West t-stat (lag = H = {cfg.labels.horizon_bars}): **{cvm['ic_newey_west_tstat_lag_h']:.2f}**",
        "",
        "### Per-model diagnostics",
        "",
        "| model | pooled IC | NW t-stat |",
        "|---|---|---|",
        *(
            f"| {name} | {m['pooled_ic']:+.4f} | {m['nw_tstat']:.2f} |"
            for name, m in cvm["members"].items()
        ),
        "",
        "| fold | period | n | IC | hit rate | decile spread |",
        "|---|---|---|---|---|---|",
    ]
    for f in cvm["per_fold"]:
        lines.append(
            f"| {f['fold_id']} | {f['ts_start'][:10]} .. {f['ts_end'][:10]} | {f['n']} "
            f"| {f['ic']:+.4f} | {f['hit_rate']:.3f} | {f['decile_spread']:+.3f} |"
        )
    lines += [
        "",
        "![IC per fold](ic_per_fold.png)",
        "",
        "## Signal portfolio (pre-cost, research-grade)",
        "",
        f"> {sp['note']}",
        "",
        f"- SR per period (non-overlapping): {sp['sr_per_period_nonoverlap']:.4f}",
        f"- SR annualized: **{sp['sr_annualized_nonoverlap']:.2f}**",
        f"- PSR vs 0: {sp['psr_vs_zero']:.3f} (n = {sp['n_nonoverlap_obs']}, skew = {sp['skew']:.2f}, kurt = {sp['kurtosis_pearson']:.1f})",
        f"- **DSR: {sp['dsr']:.3f}** over {int(sp['n_trials'])} recorded trial(s) "
        f"(expected-max SR benchmark = {sp['sr_benchmark_expected_max']:.4f})",
        "",
        "![Signal curve](signal_curve.png)",
        "",
    ]
    if bt_stats is not None:
        # the net result is the actual claim — it gets the same PSR/DSR
        # treatment as the research signal, on the ledger-comparable
        # stride-H non-overlapping subsample of net equity returns
        eqdf = pl.read_parquet(store.backtest_dir(cfg) / "equity.parquet")
        eq_h = eqdf.gather_every(cfg.labels.horizon_bars)["equity"].to_numpy()
        r_net = np.diff(eq_h) / eq_h[:-1]
        sr_net = float(np.mean(r_net) / np.std(r_net, ddof=1)) if len(r_net) > 2 else float("nan")
        skew_n, kurt_n = metrics.moments(r_net)
        psr_net = metrics.probabilistic_sharpe(sr_net, 0.0, len(r_net), skew_n, kurt_n)
        dsr_net = metrics.deflated_sharpe(
            sr_net,
            TrialsLedger(store.trials_path(cfg)).sharpe_history("backtest"),
            len(r_net),
            skew_n,
            kurt_n,
        )
        lines += [
            "## Backtest (net of costs, nautilus_trader)",
            "",
            *(f"- {k}: {v}" for k, v in bt_stats.items()),
            f"- PSR vs 0 (net, stride-{cfg.labels.horizon_bars} non-overlapping, n={len(r_net)}): "
            f"{psr_net:.3f}",
            f"- **DSR (net): {dsr_net['dsr']:.3f}** over {int(dsr_net['n_trials'])} recorded trial(s)",
            "",
            "![Equity](equity.png)",
            "",
        ]
    else:
        lines += ["## Backtest", "", "_no backtest artifacts yet — run `tml backtest`_", ""]
    lines += [
        "## Disclosures",
        "",
        "- Survivorship/selection bias: the universe was fixed in July 2026 (criteria: "
        "Binance listing <= 2021-06 AND Bybit-linear liquidity as of 2026-07). This "
        "hindsight conditioning admits long-decay boom-bust names that a momentum model "
        "shorts profitably; treat headline results as an upper bound and compare against "
        "the original-10 sensitivity run. A point-in-time universe is future work.",
        "- Labels overlap (H bars); IC significance uses Newey-West, PSR/DSR use "
        "non-overlapping subsampling.",
        _trials_disclosure(cfg),
    ]

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = rdir / f"report_{stamp}.md"
    path.write_text("\n".join(lines))
    print(f"report: {path}")
    print(
        f"report: pooled IC {cvm['pooled_ic']:+.4f}, NW t {cvm['ic_newey_west_tstat_lag_h']:.2f}, "
        f"DSR {sp['dsr']:.3f} (n_trials={int(sp['n_trials'])})"
    )
    return 0
