"""Signal-following strategy: long/flat/short via the shared tml.policy.

Execution styles (cfg.execution.mode):
  * "taker" — market order at the decision bar's close (M1-M4 baseline).
  * "maker" — post-only GTC limit one tick inside the close (below for
    buys, above for sells); if unfilled after limit_ttl_bars, cancel and
    fall back to a market order for the remainder. While a limit works,
    new decisions for that instrument are suppressed (mirrors live flow).

Conservative maker modeling relies on the runner passing
FillModel(prob_fill_on_limit=0.0): a resting limit fills only when price
trades strictly THROUGH it, never on a touch, and always at the limit
price (no improvement). Fees are stamped per fill by liquidity side
(instrument.maker_fee / taker_fee) by the engine's MakerTakerFeeModel.
"""

from __future__ import annotations

from collections import defaultdict, deque

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import LiquiditySide, OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy

from tml.backtest.data_prep import bar_type_for
from tml.config import ExecutionConfig
from tml.policy import decide, position_state


class SignalStrategy(Strategy):
    def __init__(
        self,
        signals: dict[tuple[str, int], float],  # (instrument_id str, ts_event ns) -> score
        instruments: list,
        timeframe: str,
        enter_threshold: float,
        exit_threshold: float,
        notional_per_pair: float,
        initial_cash: float,
        execution: ExecutionConfig,
        min_rebalance_notional: float = 25.0,
        execution_delay_bars: int = 0,
        allow_short: bool = False,
        force_close: set[tuple[str, int]] | None = None,
    ):
        super().__init__()
        self.signals = signals
        self._instruments = instruments
        self._timeframe = timeframe
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.allow_short = allow_short
        self.notional = notional_per_pair
        self.min_rebalance_notional = min_rebalance_notional
        self.exec = execution
        # worst-case (taker) rate for the cash cushion on entries
        self.fee_rate = (execution.taker_fee_bps + execution.slippage_bps) / 10_000
        self.delay = execution_delay_bars
        self._pending: dict = defaultdict(deque)
        # instrument_id -> {"cid": ClientOrderId, "bars": int, "side": OrderSide}
        self._working: dict = {}
        self._cash = initial_cash
        # (instrument_id str, ts_ns) marks a pair's LAST in-universe bar:
        # flatten there with a market order (delisting/membership exit is
        # not a moment to be posting passive limits)
        self.force_close = force_close or set()
        self.fills: list[dict] = []
        self.missing_signals = 0

    def on_start(self) -> None:
        for inst in self._instruments:
            self.subscribe_bars(bar_type_for(inst, self._timeframe))

    def on_bar(self, bar: Bar) -> None:
        inst_id = bar.bar_type.instrument_id
        if (str(inst_id), int(bar.ts_event)) in self.force_close:
            self._flatten_at_market(inst_id, float(bar.close))
            return
        score = self.signals.get((str(inst_id), int(bar.ts_event)))
        if score is None:
            self.missing_signals += 1
            return
        if self.delay > 0:
            q = self._pending[inst_id]
            q.append(score)
            if len(q) <= self.delay:
                return
            score = q.popleft()

        # manage a working limit order first: while it rests, no new decisions
        if inst_id in self._working:
            self._manage_working(inst_id)
            return

        px = float(bar.close)
        inst = self.cache.instrument(inst_id)
        current_qty = float(self.portfolio.net_position(inst_id) or 0.0)
        state = position_state(current_qty, px, self.min_rebalance_notional)
        action = decide(score, state, self.enter_threshold, self.exit_threshold, self.allow_short)
        if action == "hold":
            return

        if action == "open_long":
            headroom = self._cash / (1 + 2 * self.fee_rate) - self.min_rebalance_notional
            affordable = max(0.0, min(self.notional, headroom))
            if affordable < self.min_rebalance_notional:
                return
            side, qty_f = OrderSide.BUY, affordable / px
        elif action == "open_short":
            side, qty_f = OrderSide.SELL, self.notional / px
        elif state == "long":  # close long
            side, qty_f = OrderSide.SELL, current_qty
        else:  # close short
            side, qty_f = OrderSide.BUY, abs(current_qty)

        qty = inst.make_qty(qty_f)
        if qty.as_double() <= 0:
            return

        if self.exec.mode == "maker":
            tick = inst.price_increment
            # one tick inside the close: at the close itself the book sits
            # exactly there and a post-only would be rejected as crossing
            raw_px = bar.close - tick if side == OrderSide.BUY else bar.close + tick
            order = self.order_factory.limit(
                instrument_id=inst_id,
                order_side=side,
                quantity=qty,
                price=inst.make_price(raw_px),
                time_in_force=TimeInForce.GTC,
                post_only=True,
            )
            self._working[inst_id] = {"cid": order.client_order_id, "bars": 0, "side": side}
            self.submit_order(order)
        else:
            self.submit_order(
                self.order_factory.market(
                    instrument_id=inst_id, order_side=side, quantity=qty, time_in_force=TimeInForce.GTC
                )
            )

    def _flatten_at_market(self, inst_id, px: float) -> None:
        if inst_id in self._working:
            order = self.cache.order(self._working[inst_id]["cid"])
            if order is not None and order.is_open:
                self.cancel_order(order)
            del self._working[inst_id]
        current_qty = float(self.portfolio.net_position(inst_id) or 0.0)
        if abs(current_qty) * px < self.min_rebalance_notional:
            return
        inst = self.cache.instrument(inst_id)
        qty = inst.make_qty(abs(current_qty))
        if qty.as_double() <= 0:
            return
        self.submit_order(
            self.order_factory.market(
                instrument_id=inst_id,
                order_side=OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                quantity=qty,
                time_in_force=TimeInForce.GTC,
            )
        )

    def _manage_working(self, inst_id) -> None:
        w = self._working[inst_id]
        order = self.cache.order(w["cid"])
        if order is None or order.is_closed:
            del self._working[inst_id]
            return
        w["bars"] += 1
        if w["bars"] < self.exec.limit_ttl_bars:
            return
        remaining = order.leaves_qty
        self.cancel_order(order)
        del self._working[inst_id]
        if remaining.as_double() > 0:
            self.submit_order(
                self.order_factory.market(
                    instrument_id=inst_id,
                    order_side=w["side"],
                    quantity=remaining,
                    time_in_force=TimeInForce.GTC,
                )
            )

    def on_order_rejected(self, event) -> None:
        # post-only that would cross gets rejected; drop the working slot so
        # the next bar can decide afresh
        for inst_id, w in list(self._working.items()):
            if w["cid"] == event.client_order_id:
                del self._working[inst_id]

    def on_order_filled(self, event) -> None:
        # Own journal of fills — cash ledger + equity reconstruction + the
        # engine-balance cross-check all build on it.
        is_buy = event.order_side == OrderSide.BUY
        qty = float(event.last_qty)
        px = float(event.last_px)
        fee = float(event.commission.as_double()) if event.commission else 0.0
        self._cash += (-qty * px if is_buy else qty * px) - fee
        self.fills.append(
            {
                "ts": int(event.ts_event),
                "instrument": str(event.instrument_id),
                "side": "BUY" if is_buy else "SELL",
                "qty": qty,
                "px": px,
                "commission_usdt": fee,
                "liquidity": "MAKER" if event.liquidity_side == LiquiditySide.MAKER else "TAKER",
            }
        )

    def on_stop(self) -> None:
        for inst_id in list(self._working):
            order = self.cache.order(self._working[inst_id]["cid"])
            if order is not None and order.is_open:
                self.cancel_order(order)
        self._working.clear()
        for inst in self._instruments:
            self.close_all_positions(inst.id)
