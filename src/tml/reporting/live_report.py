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

    # P&L from fill cash flows per symbol (self-contained; doesn't rely on
    # Bybit's closedPnl field). Net cash flow = sells - buys - fees; when a
    # symbol is flat this IS its realized P&L. Round trips counted per symbol
    # as sign flips through zero.
    by_symbol: dict[str, dict] = {}
    for f in sorted(fills, key=lambda r: r["ts_ms"]):
        s = by_symbol.setdefault(f["symbol"], {"cash": 0.0, "pos": 0.0, "trips": 0, "wins": 0, "open_cash": 0.0})
        signed = f["qty"] if f["side"] == "Buy" else -f["qty"]
        flow = (-f["qty"] * f["px"] if f["side"] == "Buy" else f["qty"] * f["px"]) - f["fee_usdt"]
        s["cash"] += flow
        prev = s["pos"]
        s["pos"] += signed
        # a round trip completes when position returns to (near) zero
        if abs(prev) > 1e-9 and abs(s["pos"]) <= 1e-9:
            s["trips"] += 1
            if s["cash"] - s["open_cash"] > 0:
                s["wins"] += 1
            s["open_cash"] = s["cash"]
        elif abs(prev) <= 1e-9 and abs(s["pos"]) > 1e-9:
            s["open_cash"] = s["cash"] + f["qty"] * f["px"] + f["fee_usdt"]  # reset basis at open
    realized = sum(v["cash"] for v in by_symbol.values() if abs(v["pos"]) <= 1e-9)
    n_trades = sum(v["trips"] for v in by_symbol.values())
    wins = sum(v["wins"] for v in by_symbol.values())
    open_syms = [k for k, v in by_symbol.items() if abs(v["pos"]) > 1e-9]

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
        "## Realized P&L (closed round trips, net of fees — excludes open positions & demo gift coins)",
        "",
        f"- realized P&L (net): **{realized:+.2f} USDT**",
        f"- fees paid (all fills): {total_fees:.2f} USDT",
        f"- closed round trips: {n_trades}"
        + (f", win rate {wins}/{n_trades} = {wins/n_trades:.1%}" if n_trades else " (none closed yet)"),
        f"- avg realized / trip: {realized/n_trades:+.2f} USDT" if n_trades else "",
        f"- still-open positions: {len(open_syms)} ({', '.join(open_syms) if open_syms else 'none'})",
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
        summary += f"реалізовано: {realized:+.0f} USDT (комісії {total_fees:.0f})\n"
    summary += (
        f"мейкер-філів: {maker/n_fills:.0%}" + (f" (бектест {exp_maker:.0%})" if exp_maker is not None else "")
    )
    from tml.live import notify

    notify.send(summary)
    print(f"live-report -> {out}")
    print("\n".join(x for x in lines if x is not None))
    return 0
