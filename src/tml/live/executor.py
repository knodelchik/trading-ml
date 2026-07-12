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


def quantize_qty(qty: float, f: LotFilter) -> Decimal:
    """Round DOWN to the instrument's step; below min lot -> 0 (no order).

    Decimal end-to-end: str(float) artifacts like 0.30000000000000004 get
    rejected by the exchange.
    """
    q = Decimal(str(qty)).quantize(f.qty_step, rounding=ROUND_DOWN)
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

    def lot_filters(self) -> dict[str, LotFilter]:
        # per-symbol queries: the unfiltered instruments list is PAGINATED
        # and silently drops symbols beyond the first page
        out: dict[str, LotFilter] = {}
        for pair in self.cfg.data.pairs:
            symbol = pair_key(pair)
            items = self.session.get_instruments_info(category="linear", symbol=symbol)["result"]["list"]
            if items and items[0].get("status") == "Trading":
                f = items[0]["lotSizeFilter"]
                out[symbol] = LotFilter(
                    qty_step=Decimal(f["qtyStep"]),
                    min_qty=Decimal(f["minOrderQty"]),
                    min_notional=Decimal(f.get("minNotionalValue", "5")),
                )
        missing = {pair_key(p) for p in self.cfg.data.pairs} - set(out)
        if missing:
            raise RuntimeError(f"symbols not tradable on Bybit linear: {sorted(missing)}")
        return out

    def ensure_leverage_1x(self) -> None:
        from pybit.exceptions import InvalidRequestError

        for pair in self.cfg.data.pairs:
            try:
                self.session.set_leverage(
                    category="linear", symbol=pair_key(pair), buyLeverage="1", sellLeverage="1"
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
