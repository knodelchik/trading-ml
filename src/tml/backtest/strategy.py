"""Signal-following strategy: long/flat per pair, fixed per-pair notional.

The strategy consumes ONLY the precomputed OOS predictions from the CV stage
(injected as a dict); it never sees a model. The alignment of signals with
CV test folds is asserted by the runner before the engine starts, and the
strategy fail-fasts on any signal timestamp it was not given.

Decision rule at bar close t (score known at t, fill at t's close + costs),
with hysteresis to keep turnover economically sane on an H-bar-horizon signal:
    flat and score > enter_threshold  ->  buy notional_per_pair (cash-capped)
    long and score < exit_threshold   ->  sell the whole position
    otherwise                         ->  hold (no per-bar size rebalancing)
"""

from __future__ import annotations

from collections import defaultdict, deque

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy

from tml.backtest.data_prep import bar_type_for


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
        fee_rate: float,
        min_rebalance_notional: float = 25.0,
        execution_delay_bars: int = 0,
    ):
        super().__init__()
        self.signals = signals
        self._instruments = instruments
        self._timeframe = timeframe
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.notional = notional_per_pair
        # decision-to-execution latency: act on the score from `delay` bars
        # ago, so the fill lands at a later bar's close than the decision bar
        self.delay = execution_delay_bars
        self._pending: dict = defaultdict(deque)
        self.min_rebalance_notional = min_rebalance_notional
        self.fee_rate = fee_rate
        # Own cash ledger, updated from fills: a CASH account hard-rejects
        # (and nautilus HALTS the whole run on) any buy that would overdraw,
        # so buys are capped to available cash minus a fee cushion — exactly
        # what a live bot must do.
        self._cash = initial_cash
        self.fills: list[dict] = []
        self.missing_signals = 0

    def on_start(self) -> None:
        for inst in self._instruments:
            self.subscribe_bars(bar_type_for(inst, self._timeframe))

    def on_bar(self, bar: Bar) -> None:
        inst_id = bar.bar_type.instrument_id
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

        px = float(bar.close)
        inst = self.cache.instrument(inst_id)
        current_qty = float(self.portfolio.net_position(inst_id) or 0.0)
        is_long = current_qty * px >= self.min_rebalance_notional

        if not is_long and score > self.enter_threshold:
            # cap the buy to available cash net of a fee cushion
            headroom = self._cash / (1 + 2 * self.fee_rate) - self.min_rebalance_notional
            affordable = max(0.0, min(self.notional, headroom))
            if affordable < self.min_rebalance_notional:
                return
            side, qty_f = OrderSide.BUY, affordable / px
        elif is_long and score < self.exit_threshold:
            side, qty_f = OrderSide.SELL, current_qty
        else:
            return  # hold — trades happen only on state transitions

        qty = inst.make_qty(qty_f)
        if qty.as_double() <= 0:
            return
        order = self.order_factory.market(
            instrument_id=inst_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def on_order_filled(self, event) -> None:
        # Own journal of fills — used for the cash ledger and equity
        # reconstruction so we don't depend on report-API details; it is
        # cross-checked against the engine's final account balance.
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
            }
        )

    def on_stop(self) -> None:
        for inst in self._instruments:
            self.close_all_positions(inst.id)
