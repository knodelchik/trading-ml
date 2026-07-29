"""Bybit DEMO executor — demo-only by construction.

Verified against pybit 5.16.1 source and Bybit v5 docs:
  * HTTP(demo=True) pins the session to https://api-demo.bybit.com; demo
    keys are minted in the Demo Trading UI and do not work on mainnet
    (and vice versa) — an extra endpoint assert makes this unmistakable;
  * linear USDT perps, one-way mode, leverage 1x per symbol (error 110043
    "leverage not modified" is idempotent-OK);
  * long/flat only: entries are plain Buys, exits are Sell + reduceOnly
    (cannot accidentally open a short);
  * qty is Decimal-quantized to the instrument's qtyStep (ADA/DOGE steps
    are whole numbers), 5 USDT min notional everywhere;
  * demo retains orders/executions ~7 days server-side — the journal in
    state.py is the durable record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

from tml.config import Config
from tml.data.store import pair_key


def load_env_file(root: Path) -> None:
    """Minimal .env loader (no extra dependency); existing env vars win."""
    p = root / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class LotFilter:
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    tick_size: Decimal


def _snap(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    """Snap to a multiple of `step` via division — Decimal.quantize() snaps
    to the step's EXPONENT, which is wrong for grids like '0.10' or '10'."""
    return (value / step).to_integral_value(rounding=rounding) * step


def quantize_price(px: float, side: str, f: LotFilter) -> Decimal:
    """Snap a maker limit price to the tick grid, staying passive:
    buys round DOWN, sells round UP."""
    from decimal import ROUND_CEILING

    rounding = ROUND_DOWN if side == "Buy" else ROUND_CEILING
    return _snap(Decimal(str(px)), f.tick_size, rounding)


def quantize_qty(qty: float, f: LotFilter) -> Decimal:
    """Round DOWN to the instrument's step; below min lot -> 0 (no order).

    Decimal end-to-end: str(float) artifacts like 0.30000000000000004 get
    rejected by the exchange.
    """
    q = _snap(Decimal(str(qty)), f.qty_step, ROUND_DOWN)
    return q if q >= f.min_qty else Decimal(0)


class BybitDemoExecutor:
    def __init__(self, cfg: Config):
        load_env_file(cfg.root)
        key = os.environ.get("BYBIT_DEMO_API_KEY", "")
        secret = os.environ.get("BYBIT_DEMO_API_SECRET", "")
        if not key or not secret:
            raise RuntimeError(
                "BYBIT_DEMO_API_KEY / BYBIT_DEMO_API_SECRET not set — create DEMO keys "
                "in Bybit's Demo Trading UI and put them in ~/trading-ml/.env "
                "(see .env.example). Demo keys only; real keys will not work here."
            )
        from pybit.unified_trading import HTTP

        self.session = HTTP(demo=True, api_key=key, api_secret=secret)
        assert "api-demo" in self.session.endpoint, (
            f"refusing to run: endpoint {self.session.endpoint} is not the demo domain"
        )
        self.cfg = cfg

    # --- account & instruments -------------------------------------------------

    def lot_filters(self, pairs: list[str], require_all: bool = True) -> dict[str, LotFilter]:
        # per-symbol queries: the unfiltered instruments list is PAGINATED
        # and silently drops symbols beyond the first page
        from pybit.exceptions import InvalidRequestError

        out: dict[str, LotFilter] = {}
        for pair in pairs:
            symbol = pair_key(pair)
            try:
                items = self.session.get_instruments_info(category="linear", symbol=symbol)[
                    "result"
                ]["list"]
            except InvalidRequestError:
                # symbol invalid on Bybit (e.g. PEPE trades as 1000PEPEUSDT
                # with a 1000x multiplier — a contract mismatch we don't
                # trade); treat as not-tradable and skip
                continue
            if items and items[0].get("status") == "Trading":
                f = items[0]["lotSizeFilter"]
                out[symbol] = LotFilter(
                    qty_step=Decimal(f["qtyStep"]),
                    min_qty=Decimal(f["minOrderQty"]),
                    min_notional=Decimal(f.get("minNotionalValue", "5")),
                    tick_size=Decimal(items[0]["priceFilter"]["tickSize"]),
                )
        missing = {pair_key(p) for p in pairs} - set(out)
        if missing and require_all:
            raise RuntimeError(f"symbols not tradable on Bybit linear: {sorted(missing)}")
        return out

    def ensure_leverage_1x(self, symbols: list[str]) -> None:
        from pybit.exceptions import InvalidRequestError

        for symbol in symbols:
            try:
                self.session.set_leverage(
                    category="linear", symbol=symbol, buyLeverage="1", sellLeverage="1"
                )
            except InvalidRequestError as e:
                if getattr(e, "status_code", None) != 110043:  # 110043 = already 1x
                    raise

    def positions(self) -> dict[str, float]:
        """symbol -> signed base qty (one-way mode).

        MUST paginate: /v5/position/list defaults to 20 per page, and with a
        22-pair universe >20 open positions is an expected state — a
        truncated read makes a live position look flat and the tick would
        double-open it. Terminate on a short page (the demo API can return
        a non-empty cursor even when complete).
        """
        out: dict[str, float] = {}
        cursor = ""
        while True:
            kwargs = {"category": "linear", "settleCoin": "USDT", "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            result = self.session.get_positions(**kwargs)["result"]
            page = result["list"]
            for p in page:
                size = float(p["size"] or 0)
                if size:
                    out[p["symbol"]] = size if p["side"] == "Buy" else -size
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(page) < 200:
                break
        return out

    def open_unrealized(self) -> float:
        """Total unrealized P&L (USDT) across open perp positions. Immune to
        the demo gift-coin spot balance that dominates raw equity — this plus
        the realized journal is the bot's true trading P&L."""
        total = 0.0
        cursor = ""
        while True:
            kwargs = {"category": "linear", "settleCoin": "USDT", "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            result = self.session.get_positions(**kwargs)["result"]
            page = result["list"]
            for p in page:
                if float(p["size"] or 0):
                    total += float(p.get("unrealisedPnl") or 0)
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(page) < 200:
                break
        return total

    def equity_usdt(self) -> dict[str, float]:
        resp = self.session.get_wallet_balance(accountType="UNIFIED")
        acct = resp["result"]["list"][0]
        return {
            "total_equity": float(acct["totalEquity"]),
            "available": float(acct.get("totalAvailableBalance") or 0),
        }

    # --- trading ----------------------------------------------------------------

    def quantize(self, symbol: str, qty: float, filters: dict[str, LotFilter]) -> Decimal:
        return quantize_qty(qty, filters[symbol])

    def market_order(self, symbol: str, side: str, qty: Decimal, link_id: str, reduce_only: bool) -> str:
        resp = self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            orderLinkId=link_id[:36],
            reduceOnly=reduce_only,
        )
        return resp["result"]["orderId"]

    def limit_post_only(
        self, symbol: str, side: str, qty: Decimal, price: Decimal, link_id: str, reduce_only: bool
    ) -> str:
        """Maker order. NOTE: a PostOnly that would cross is NOT rejected at
        the API level — it returns retCode 0 and gets cancelled async with
        rejectReason=EC_PostOnlyWillTakeLiquidity; the next tick detects the
        vanished order and re-decides."""
        resp = self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty),
            price=str(price),
            timeInForce="PostOnly",
            orderLinkId=link_id[:36],
            reduceOnly=reduce_only,
        )
        return resp["result"]["orderId"]

    def open_orders(self) -> list[dict]:
        """All our resting orders (paginated; page cap is 50)."""
        out: list[dict] = []
        cursor = ""
        while True:
            kwargs = {"category": "linear", "settleCoin": "USDT", "limit": 50, "openOnly": 0}
            if cursor:
                kwargs["cursor"] = cursor
            result = self.session.get_open_orders(**kwargs)["result"]
            page = result["list"]
            out.extend(
                {
                    "symbol": o["symbol"],
                    "order_id": o["orderId"],
                    "order_link_id": o.get("orderLinkId", ""),
                    "side": o["side"],
                    "qty": float(o["qty"]),
                    "cum_exec_qty": float(o.get("cumExecQty") or 0),
                    "price": float(o["price"]),
                    "reduce_only": bool(o.get("reduceOnly")),
                    "created_ms": int(o["createdTime"]),
                }
                for o in page
            )
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(page) < 50:
                break
        return out

    def cancel(self, symbol: str, order_id: str) -> None:
        self.session.cancel_order(category="linear", symbol=symbol, orderId=order_id)

    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        """(best bid, best ask) from Bybit — the real book, so a maker limit
        priced at the touch does NOT cross (pricing one tick inside the
        Binance close crossed the Bybit spread and got PostOnly-rejected)."""
        t = self.session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
        return float(t["bid1Price"]), float(t["ask1Price"])

    def recent_closed_pnl(self, days: int = 7) -> list[dict]:
        """Closed round trips from Bybit over the last `days` (authoritative
        realized P&L; captures trips opened before our local journal existed).
        Bybit requires startTime AND endTime together, window <= 7 days."""
        import time

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - min(days, 7) * 86_400_000
        out: list[dict] = []
        cursor = ""
        while True:
            kwargs = {
                "category": "linear",
                "settleCoin": "USDT",
                "startTime": start_ms,
                "endTime": now_ms,
                "limit": 100,
            }
            if cursor:
                kwargs["cursor"] = cursor
            result = self.session.get_closed_pnl(**kwargs)["result"]
            page = result["list"]
            out.extend(
                {
                    # closing side is opposite the position; Sell closes a long
                    "symbol": c["symbol"],
                    "side": "long" if c["side"] == "Sell" else "short",
                    "qty": float(c["qty"]),
                    "closed_pnl": float(c["closedPnl"]),
                    "open_ms": int(c["createdTime"]),
                    "close_ms": int(c["updatedTime"]),
                    "order_id": c["orderId"],
                }
                for c in page
            )
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(page) < 100:
                break
        return out

    def executions_since(self, start_ms: int) -> list[dict]:
        """All linear USDT executions since start_ms (paginated). The durable
        local fills journal is built from this — Bybit demo keeps executions
        only ~7 days, so every tick captures the window since the last one."""
        out: list[dict] = []
        cursor = ""
        while True:
            kwargs = {"category": "linear", "startTime": start_ms, "limit": 100}
            if cursor:
                kwargs["cursor"] = cursor
            result = self.session.get_executions(**kwargs)["result"]
            page = result["list"]
            out.extend(
                {
                    "exec_id": e["execId"],
                    "ts_ms": int(e["execTime"]),
                    "symbol": e["symbol"],
                    "side": e["side"],
                    "qty": float(e["execQty"]),
                    "px": float(e["execPrice"]),
                    "fee_usdt": float(e["execFee"]),
                    "is_maker": bool(e.get("isMaker")),
                    "closed_pnl": float(e.get("closedPnl") or 0.0),
                    "order_link_id": e.get("orderLinkId", ""),
                }
                for e in page
            )
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(page) < 100:
                break
        return out

    def executions_for(self, symbol: str, order_id: str) -> list[dict]:
        resp = self.session.get_executions(category="linear", symbol=symbol, orderId=order_id)
        return [
            {
                "symbol": e["symbol"],
                "side": e["side"],
                "exec_px": float(e["execPrice"]),
                "exec_qty": float(e["execQty"]),
                "fee_usdt": float(e["execFee"]),
                "fee_rate": float(e.get("feeRate") or 0),
                "exec_time_ms": int(e["execTime"]),
                "order_link_id": e.get("orderLinkId", ""),
            }
            for e in resp["result"]["list"]
        ]
