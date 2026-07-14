"""Build and push the dashboard snapshot to lifeos.

Reads the durable executions journal (fills.jsonl) + equity journal and
produces the JSON the lifeos `/api/trading/ingest` endpoint stores. All
money is integer cents (USDT * 100) to match lifeos's convention. Realized
P&L is computed from fill CASH FLOWS per symbol (self-contained), the same
way live_report does — round trips complete when a position returns to zero.

push_snapshot is best-effort: a failure NEVER breaks a trading tick.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from tml.config import Config
from tml.live.state import journal_dir


def _cents(usdt: float) -> int:
    return round(usdt * 100)


def _positions_detail(ex) -> list[dict]:
    resp = ex.session.get_positions(category="linear", settleCoin="USDT")
    out = []
    for p in resp["result"]["list"]:
        size = float(p["size"] or 0)
        if size:
            signed = size if p["side"] == "Buy" else -size
            out.append({"symbol": p["symbol"], "size": signed, "value": float(p.get("positionValue") or 0)})
    return out


def compute_snapshot(cfg: Config, ex=None) -> dict:
    """Build the dashboard payload. When `ex` (a BybitDemoExecutor) is given,
    closed trades + realized P&L come from Bybit's authoritative closed-PnL
    (captures round trips opened before the local journal existed); the
    equity curve, maker share and fees still come from the local journal."""
    fills_path = journal_dir(cfg) / "fills.jsonl"
    fills = (
        [json.loads(line) for line in fills_path.open() if line.strip()]
        if fills_path.exists()
        else []
    )
    fills.sort(key=lambda r: r["ts_ms"])

    # equity curve from the equity journal
    eq_path = journal_dir(cfg) / "equity.jsonl"
    curve = []
    if eq_path.exists():
        for line in eq_path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            curve.append({"ts": r["bar_ts"], "equityCents": _cents(r["total_equity"])})
    equity_cents = curve[-1]["equityCents"] if curve else 0

    # per-symbol round trips from cash flows
    trades: list[dict] = []
    pos: dict[str, dict] = {}
    total_fees = 0.0
    n_fills = len(fills)
    maker = sum(1 for f in fills if f["is_maker"])
    for f in fills:
        total_fees += f["fee_usdt"]
        s = pos.setdefault(f["symbol"], {"qty": 0.0, "cash": 0.0, "open_ts": None, "open_side": None, "last_px": f["px"]})
        s["last_px"] = f["px"]
        signed = f["qty"] if f["side"] == "Buy" else -f["qty"]
        flow = (-f["qty"] * f["px"] if f["side"] == "Buy" else f["qty"] * f["px"]) - f["fee_usdt"]
        prev = s["qty"]
        if abs(prev) <= 1e-9:  # opening
            s["cash"] = flow
            s["open_ts"] = f["ts_ms"]
            s["open_side"] = "long" if signed > 0 else "short"
        else:
            s["cash"] += flow
        s["qty"] = prev + signed
        if abs(prev) > 1e-9 and abs(s["qty"]) <= 1e-9:  # closed a round trip
            trades.append(
                {
                    "execKey": f["exec_id"],
                    "symbol": f["symbol"],
                    "side": s["open_side"],
                    "openTs": datetime.fromtimestamp(s["open_ts"] / 1000, UTC).isoformat(),
                    "closeTs": datetime.fromtimestamp(f["ts_ms"] / 1000, UTC).isoformat(),
                    "pnlCents": _cents(s["cash"]),
                    "feesCents": 0,
                    "isMaker": bool(f["is_maker"]),
                }
            )

    open_positions = [
        {"symbol": sym, "qty": round(v["qty"], 6), "notionalCents": _cents(abs(v["qty"]) * v["last_px"])}
        for sym, v in pos.items()
        if abs(v["qty"]) > 1e-9
    ]

    # authoritative closed trades from Bybit when we have an executor —
    # captures trips opened before the local journal started
    if ex is not None:
        try:
            span_ms = 14 * 86_400_000
            start = (fills[0]["ts_ms"] if fills else int(datetime.now(UTC).timestamp() * 1000)) - span_ms
            closed = ex.closed_pnl_since(start)
            trades = [
                {
                    "execKey": c["order_id"],
                    "symbol": c["symbol"],
                    "side": c["side"],
                    "openTs": datetime.fromtimestamp(c["open_ms"] / 1000, UTC).isoformat(),
                    "closeTs": datetime.fromtimestamp(c["close_ms"] / 1000, UTC).isoformat(),
                    "pnlCents": _cents(c["closed_pnl"]),
                    "feesCents": 0,
                    "isMaker": False,
                }
                for c in closed
            ]
            open_positions = [
                {
                    "symbol": p["symbol"],
                    "qty": round(p["size"], 6),
                    "notionalCents": _cents(p["value"]),
                }
                for p in _positions_detail(ex)
            ]
        except Exception:  # noqa: BLE001 — fall back to journal reconstruction
            pass

    realized = sum(t["pnlCents"] for t in trades)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnlCents"] > 0)
    span_days = (fills[-1]["ts_ms"] - fills[0]["ts_ms"]) / 86_400_000 if len(fills) > 1 else 0.0

    # backtest reference from stats.json
    bt: dict = {}
    stats_path = cfg.artifacts / "backtest" / "stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        bt = {
            "makerSharePct": round((st.get("maker_fill_share") or 0) * 100, 1),
            "totalReturnPct": st.get("total_return_pct"),
            "sharpe": st.get("sharpe_net_annualized"),
        }

    return {
        "windowDays": round(span_days, 2),
        "equityCents": equity_cents,
        "realizedCents": realized,
        "feesCents": _cents(total_fees),
        "winRatePct": round(100 * wins / n_trades, 1) if n_trades else 0.0,
        "makerSharePct": round(100 * maker / n_fills, 1) if n_fills else 0.0,
        "nTradesClosed": n_trades,
        "tradesPerWeek": round(7 * n_trades / span_days, 2) if span_days > 0 else 0.0,
        "openPositions": open_positions,
        "backtest": bt,
        "equityCurve": curve[-500:],  # cap payload size
        "trades": trades[-100:],
    }


def push_snapshot(cfg: Config) -> bool:
    """POST the snapshot to lifeos. Best-effort; never raises."""
    url = os.environ.get("LIFEOS_TRADING_URL", "")
    secret = os.environ.get("LIFEOS_TRADING_SECRET", "")
    if not url or not secret:
        return False
    try:
        import urllib.request

        from tml.live.executor import BybitDemoExecutor

        try:
            ex = BybitDemoExecutor(cfg)
        except Exception:  # noqa: BLE001 — offline/no creds: journal-only snapshot
            ex = None
        req = urllib.request.Request(
            url,
            data=json.dumps(compute_snapshot(cfg, ex)).encode(),
            headers={"Content-Type": "application/json", "x-trading-secret": secret},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False
