"""Live-vs-backtest report.

Reads the durable executions journal (artifacts/live/journal/fills.jsonl,
captured each tick from Bybit) and compares realized trading behaviour to
the backtest's expectations. Uses Bybit's per-close `closedPnl`, which is
the account's REALIZED P&L and excludes the demo gift coins that otherwise
contaminate total equity.

Answers the questions a paper track exists to answer:
  * are execution costs what the backtest assumed (maker share, fee/trade)?
  * is the win rate / PnL-per-trade in the backtest's ballpark?
  * how many trades/week vs expected?
It does NOT try to declare the strategy proven — weeks of Sharpe-0.5-ish
returns are mostly noise; this checks the *mechanics* match, which is what
the first month can actually establish.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from tml.config import Config
from tml.live.state import journal_dir


def _backtest_expectations(cfg: Config) -> dict:
    p = cfg.artifacts / "backtest" / "stats.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def build_live_report(cfg: Config) -> int:
    fills_path = journal_dir(cfg) / "fills.jsonl"
    if not fills_path.exists():
        print("live-report: no fills journal yet — the bot hasn't traded, or ticks haven't run")
        return 0
    fills = [json.loads(line) for line in fills_path.open() if line.strip()]
    if not fills:
        print("live-report: fills journal empty")
        return 0

    ts = [f["ts_ms"] for f in fills]
    span_days = (max(ts) - min(ts)) / 86_400_000 or 1e-9
    n_fills = len(fills)
    maker = sum(1 for f in fills if f["is_maker"])
    total_fees = sum(f["fee_usdt"] for f in fills)
    gross_traded = sum(f["qty"] * f["px"] for f in fills)

    closes = [f for f in fills if f["closed_pnl"] != 0.0]  # closing executions carry realized PnL
    realized = sum(f["closed_pnl"] for f in closes)
    wins = sum(1 for f in closes if f["closed_pnl"] > 0)
    n_trades = len(closes)

    bt = _backtest_expectations(cfg)
    exp_maker = bt.get("maker_fill_share")

    lines = [
        "# Live vs backtest — paper track",
        "",
        f"- window: {datetime.fromtimestamp(min(ts)/1000, UTC):%Y-%m-%d %H:%M} .. "
        f"{datetime.fromtimestamp(max(ts)/1000, UTC):%Y-%m-%d %H:%M} UTC "
        f"({span_days:.1f} days)",
        f"- executions: {n_fills} ({7*n_fills/span_days:.1f}/week)",
        f"- closing trades (realized): {n_trades} ({7*n_trades/span_days:.1f}/week)",
        "",
        "## Realized P&L (Bybit closedPnl — excludes demo gift coins)",
        "",
        f"- realized P&L: **{realized:+.2f} USDT**",
        f"- fees paid: {total_fees:.2f} USDT",
        f"- net of fees: **{realized - total_fees:+.2f} USDT**",
        f"- win rate: {wins}/{n_trades} = {wins/n_trades:.1%}" if n_trades else "- win rate: n/a (no closed trades yet)",
        f"- avg realized / trade: {realized/n_trades:+.2f} USDT" if n_trades else "",
        "",
        "## Execution quality vs backtest",
        "",
        f"- maker fill share: **live {maker/n_fills:.1%}** vs backtest "
        f"{exp_maker:.1%}" if exp_maker is not None else f"- maker fill share: live {maker/n_fills:.1%}",
        f"- effective cost: live {total_fees/max(gross_traded,1e-9)*1e4:.1f} bps of turnover "
        f"vs assumed maker {cfg.execution.maker_fee_bps:.0f} bps",
        f"- turnover: {gross_traded:,.0f} USDT",
        "",
        "## Honest read",
        "",
        f"With {n_trades} closed trades over {span_days:.1f} days this is still mostly "
        "noise on P&L (a ~Sharpe-2 strategy needs months to distinguish from luck). "
        "What IS checkable now: whether execution costs and maker-fill share match the "
        "backtest — if they do, the +153.8%/Sharpe 2.49 backtest rests on realistic "
        "assumptions. Divergence in costs/fills is the early warning to investigate.",
    ]
    out = cfg.artifacts / "reports" / f"live_report_{datetime.now(UTC):%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(x for x in lines if x is not None))

    summary = (
        f"📊 <b>tml тиждень</b> ({span_days:.0f}д)\n"
        f"угод: {n_trades} | вінрейт: {wins/n_trades:.0%}\n" if n_trades else f"📊 <b>tml</b> ({span_days:.0f}д): ще нема закритих угод\n"
    )
    if n_trades:
        summary += (
            f"реалізовано: {realized:+.0f} USDT (комісії {total_fees:.0f})\n"
            f"мейкер-філів: {maker/n_fills:.0%} (бектест {exp_maker:.0%})"
            if exp_maker is not None
            else ""
        )
    from tml.live import notify

    notify.send(summary)
    print(f"live-report -> {out}")
    print("\n".join(x for x in lines if x is not None))
    return 0
