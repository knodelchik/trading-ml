"""Backtest mechanics verified against hand-computed numbers.

3 bars, close prices 100 -> 110 -> 120. Signal: long at bar 1, exit at bar 2.
Expectations (notional 1000 USDT, costs 13 bps baked into taker fee):
    BUY  10 LTC @ 100, fee = 1000 * 0.0013 = 1.30
    SELL 10 LTC @ 110, fee = 1100 * 0.0013 = 1.43
    final cash = 100000 + 10*(110-100) - 1.30 - 1.43 = 100097.27
This also pins down the execution model: market orders fill at the CLOSE of
the bar that triggered them (bar_execution), which is our documented
decision-at-close semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from tests.conftest import make_config
from tml.config import ExecutionConfig
from tml.data import store
from tml.validation.splits import Fold

T0 = datetime(2021, 1, 1, 1, 0, tzinfo=UTC)
CLOSES = [100.0, 110.0, 120.0]


def _write_bars(cfg) -> None:
    df = pl.DataFrame(
        {
            "ts": [T0 + timedelta(hours=i) for i in range(3)],
            "open": CLOSES,
            "high": CLOSES,
            "low": CLOSES,
            "close": CLOSES,
            # generous volume: the engine splits a bar into 4 ticks carrying
            # volume/4 liquidity each; tiny volume would partial-fill orders
            "volume": [10_000.0, 10_000.0, 10_000.0],
            "pair": "LTCUSDT",
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )
    store.write_raw_year(cfg, "LTC/USDT", 2021, df)


def test_two_fills_hand_computed_pnl(tmp_path):
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money

    from tml.backtest.data_prep import VENUE_NAME, bars_for_pair, make_instrument
    from tml.backtest.strategy import SignalStrategy

    cfg = make_config(tmp_path, pairs=("LTC/USDT",))
    _write_bars(cfg)

    # bypass_logging: the Rust logging subsystem may only be initialized once
    # per process; a second engine in the same pytest run would abort
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SMOKE-001"), logging=LoggingConfig(bypass_logging=True)
        )
    )
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        default_leverage=Decimal(1),
        starting_balances=[Money(100_000, USDT)],
        fill_model=FillModel(prob_slippage=0.0, random_seed=1),
        fee_model=MakerTakerFeeModel(),
    )
    inst = make_instrument(cfg, "LTC/USDT")
    engine.add_instrument(inst)
    bars = bars_for_pair(cfg, "LTC/USDT", inst, T0, T0 + timedelta(hours=4))
    assert len(bars) == 3
    engine.add_data(bars)

    ns = lambda dt: int(dt.timestamp() * 1_000_000_000)  # noqa: E731
    signals = {
        (f"LTCUSDT.{VENUE_NAME}", ns(T0)): +1.0,
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=1))): -1.0,
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=2))): -1.0,
    }
    strategy = SignalStrategy(
        signals=signals,
        instruments=[inst],
        timeframe="1h",
        enter_threshold=0.0,
        exit_threshold=0.0,
        notional_per_pair=1000.0,
        initial_cash=100_000.0,
        execution=ExecutionConfig(taker_fee_bps=10.0, slippage_bps=3.0),
    )
    engine.add_strategy(strategy)
    engine.run()

    account = engine.portfolio.account(Venue(VENUE_NAME))
    final_usdt = float(account.balance_total(USDT).as_double())
    engine.dispose()

    assert strategy.missing_signals == 0
    assert len(strategy.fills) == 2, strategy.fills

    buy, sell = strategy.fills
    assert buy["side"] == "BUY" and sell["side"] == "SELL"
    # fills happen at the close of the TRIGGERING bar — the decision price
    assert buy["px"] == pytest.approx(100.0)
    assert sell["px"] == pytest.approx(110.0)
    assert buy["qty"] == pytest.approx(10.0)
    assert sell["qty"] == pytest.approx(10.0)
    assert buy["commission_usdt"] == pytest.approx(1.30, abs=0.01)
    assert sell["commission_usdt"] == pytest.approx(1.43, abs=0.01)
    assert final_usdt == pytest.approx(100_097.27, abs=0.05)


def test_execution_delay_shifts_fills_one_bar(tmp_path):
    """With delay=1 the same signals act one bar later:
    BUY  1000/110 = 9.090909 LTC @ 110 (fee 1000*0.0013 = 1.30)
    SELL 9.090909 LTC @ 120        (fee 1090.91*0.0013 = 1.42)
    final = 100000 + 9.090909*(120-110) - 1.30 - 1.42 = 100088.19
    """
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money

    from tml.backtest.data_prep import VENUE_NAME, bars_for_pair, make_instrument
    from tml.backtest.strategy import SignalStrategy

    cfg = make_config(tmp_path, pairs=("LTC/USDT",))
    _write_bars(cfg)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SMOKE-002"), logging=LoggingConfig(bypass_logging=True)
        )
    )
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        default_leverage=Decimal(1),
        starting_balances=[Money(100_000, USDT)],
        fill_model=FillModel(prob_slippage=0.0, random_seed=1),
        fee_model=MakerTakerFeeModel(),
    )
    inst = make_instrument(cfg, "LTC/USDT")
    engine.add_instrument(inst)
    engine.add_data(bars_for_pair(cfg, "LTC/USDT", inst, T0, T0 + timedelta(hours=4)))

    ns = lambda dt: int(dt.timestamp() * 1_000_000_000)  # noqa: E731
    signals = {
        (f"LTCUSDT.{VENUE_NAME}", ns(T0)): +1.0,
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=1))): -1.0,
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=2))): -1.0,
    }
    strategy = SignalStrategy(
        signals=signals,
        instruments=[inst],
        timeframe="1h",
        enter_threshold=0.0,
        exit_threshold=0.0,
        notional_per_pair=1000.0,
        initial_cash=100_000.0,
        execution=ExecutionConfig(taker_fee_bps=10.0, slippage_bps=3.0),
        execution_delay_bars=1,
    )
    engine.add_strategy(strategy)
    engine.run()
    account = engine.portfolio.account(Venue(VENUE_NAME))
    final_usdt = float(account.balance_total(USDT).as_double())
    engine.dispose()

    assert len(strategy.fills) == 2, strategy.fills
    buy, sell = strategy.fills
    assert buy["px"] == pytest.approx(110.0)  # decided at 100-close, filled at NEXT close
    assert sell["px"] == pytest.approx(120.0)
    assert final_usdt == pytest.approx(100_088.19, abs=0.05)


def test_on_stop_close_is_journaled(tmp_path):
    """A position still open at end-of-data is closed by on_stop, but that
    fill executes after the strategy stopped and never reaches
    on_order_filled — the runner must replay it into the journal:
    BUY 10 @ 100 (fee 1.30), on_stop SELL 10 @ 120 (fee 1.56),
    final = 100000 + 200 - 1.30 - 1.56 = 100197.14."""
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money

    from tml.backtest.data_prep import VENUE_NAME, bars_for_pair, make_instrument
    from tml.backtest.runner import _replay_missed_fills
    from tml.backtest.strategy import SignalStrategy

    cfg = make_config(tmp_path, pairs=("LTC/USDT",))
    _write_bars(cfg)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SMOKE-003"), logging=LoggingConfig(bypass_logging=True)
        )
    )
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        default_leverage=Decimal(1),
        starting_balances=[Money(100_000, USDT)],
        fill_model=FillModel(prob_slippage=0.0, random_seed=1),
        fee_model=MakerTakerFeeModel(),
    )
    inst = make_instrument(cfg, "LTC/USDT")
    engine.add_instrument(inst)
    engine.add_data(bars_for_pair(cfg, "LTC/USDT", inst, T0, T0 + timedelta(hours=4)))

    ns = lambda dt: int(dt.timestamp() * 1_000_000_000)  # noqa: E731
    signals = {
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=i))): +1.0 for i in range(3)
    }
    strategy = SignalStrategy(
        signals=signals,
        instruments=[inst],
        timeframe="1h",
        enter_threshold=0.0,
        exit_threshold=0.0,
        notional_per_pair=1000.0,
        initial_cash=100_000.0,
        execution=ExecutionConfig(taker_fee_bps=10.0, slippage_bps=3.0),
    )
    engine.add_strategy(strategy)
    engine.run()

    n_before_replay = len(strategy.fills)
    _replay_missed_fills(engine, strategy)

    account = engine.portfolio.account(Venue(VENUE_NAME))
    final_usdt = float(account.balance_total(USDT).as_double())
    engine.dispose()

    assert n_before_replay == 1  # only the BUY reached on_order_filled
    assert len(strategy.fills) == 2  # replay recovered the on_stop SELL
    sell = strategy.fills[-1]
    assert sell["side"] == "SELL"
    assert sell["px"] == pytest.approx(120.0)
    assert final_usdt == pytest.approx(100_197.14, abs=0.05)
    # journal-reconstructed cash now matches the engine account
    cash = 100_000.0
    for f in strategy.fills:
        flow = -f["qty"] * f["px"] if f["side"] == "BUY" else f["qty"] * f["px"]
        cash += flow - f["commission_usdt"]
    assert cash == pytest.approx(final_usdt, abs=0.05)


def test_short_round_trip_hand_computed(tmp_path):
    """Short on a MARGIN perp: SELL 10 @ 100 (fee 1.30), BUY back @ 110
    (fee 1.43). Realized = -100; final = 100000 - 100 - 2.73 = 99897.27."""
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money

    from tml.backtest.data_prep import VENUE_NAME, bars_for_pair, make_instrument
    from tml.backtest.strategy import SignalStrategy

    cfg = make_config(tmp_path, pairs=("LTC/USDT",))
    _write_bars(cfg)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SMOKE-004"), logging=LoggingConfig(bypass_logging=True)
        )
    )
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        default_leverage=Decimal(1),
        starting_balances=[Money(100_000, USDT)],
        fill_model=FillModel(prob_slippage=0.0, random_seed=1),
        fee_model=MakerTakerFeeModel(),
    )
    inst = make_instrument(cfg, "LTC/USDT")
    engine.add_instrument(inst)
    engine.add_data(bars_for_pair(cfg, "LTC/USDT", inst, T0, T0 + timedelta(hours=4)))

    ns = lambda dt: int(dt.timestamp() * 1_000_000_000)  # noqa: E731
    signals = {
        (f"LTCUSDT.{VENUE_NAME}", ns(T0)): -1.0,
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=1))): +1.0,
        (f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=2))): +0.0,
    }
    strategy = SignalStrategy(
        signals=signals,
        instruments=[inst],
        timeframe="1h",
        enter_threshold=0.0,
        exit_threshold=0.0,
        notional_per_pair=1000.0,
        initial_cash=100_000.0,
        execution=ExecutionConfig(taker_fee_bps=10.0, slippage_bps=3.0),
        allow_short=True,
    )
    engine.add_strategy(strategy)
    engine.run()
    account = engine.portfolio.account(Venue(VENUE_NAME))
    final_usdt = float(account.balance_total(USDT).as_double())
    engine.dispose()

    assert len(strategy.fills) == 2, strategy.fills
    sell, buy = strategy.fills
    assert sell["side"] == "SELL" and sell["px"] == pytest.approx(100.0)
    assert buy["side"] == "BUY" and buy["px"] == pytest.approx(110.0)
    assert sell["qty"] == pytest.approx(10.0) and buy["qty"] == pytest.approx(10.0)
    assert final_usdt == pytest.approx(99_897.27, abs=0.05)


def _maker_engine(cfg, closes_ohlc):
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money

    from tml.backtest.data_prep import VENUE_NAME, bars_for_pair, make_instrument

    df = pl.DataFrame(
        {
            "ts": [T0 + timedelta(hours=i) for i in range(len(closes_ohlc))],
            "open": [r[0] for r in closes_ohlc],
            "high": [r[1] for r in closes_ohlc],
            "low": [r[2] for r in closes_ohlc],
            "close": [r[3] for r in closes_ohlc],
            "volume": [10_000.0] * len(closes_ohlc),
            "pair": "LTCUSDT",
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )
    store.write_raw_year(cfg, "LTC/USDT", 2021, df)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SMOKE-MKR"), logging=LoggingConfig(bypass_logging=True)
        )
    )
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        default_leverage=Decimal(1),
        starting_balances=[Money(100_000, USDT)],
        fill_model=FillModel(prob_slippage=0.0, prob_fill_on_limit=0.0, random_seed=1),
        fee_model=MakerTakerFeeModel(),
    )
    inst = make_instrument(cfg, "LTC/USDT")
    engine.add_instrument(inst)
    engine.add_data(bars_for_pair(cfg, "LTC/USDT", inst, T0, T0 + timedelta(hours=10)))
    return engine, inst


def test_maker_fill_when_price_trades_through(tmp_path):
    """Post-only BUY at (100 - tick); next bar's low=95 trades THROUGH it:
    fill at the limit price as MAKER, fee = 1000 * 0.0002 = 0.20."""

    from tml.backtest.data_prep import VENUE_NAME
    from tml.backtest.strategy import SignalStrategy

    cfg = make_config(tmp_path, pairs=("LTC/USDT",))
    bars = [(100, 100, 100, 100), (100, 100, 95, 98), (98, 98, 98, 98)]
    engine, inst = _maker_engine(cfg, bars)

    ns = lambda dt: int(dt.timestamp() * 1_000_000_000)  # noqa: E731
    signals = {(f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=i))): s for i, s in enumerate([+1.0, +0.5, +0.5])}
    strategy = SignalStrategy(
        signals=signals,
        instruments=[inst],
        timeframe="1h",
        enter_threshold=0.0,
        exit_threshold=0.0,
        notional_per_pair=1000.0,
        initial_cash=100_000.0,
        execution=ExecutionConfig(mode="maker", maker_fee_bps=2.0, taker_fee_bps=10.0, slippage_bps=3.0, limit_ttl_bars=2),
    )
    engine.add_strategy(strategy)
    engine.run()
    from tml.backtest.runner import _replay_missed_fills

    _replay_missed_fills(engine, strategy)  # on_stop fills execute after the strategy stops
    engine.dispose()

    buys = [f for f in strategy.fills if f["side"] == "BUY"]
    assert len(buys) == 1, strategy.fills
    assert buys[0]["liquidity"] == "MAKER"
    assert buys[0]["px"] == pytest.approx(100.0, abs=1e-3)  # fills AT the limit, not at the low
    assert buys[0]["qty"] == pytest.approx(10.0)
    assert buys[0]["commission_usdt"] == pytest.approx(0.20, abs=0.01)
    # on_stop closes the long with a market order = TAKER at 13 bps
    sells = [f for f in strategy.fills if f["side"] == "SELL"]
    assert len(sells) == 1 and sells[0]["liquidity"] == "TAKER"
    assert sells[0]["commission_usdt"] == pytest.approx(10 * 98 * 0.0013, abs=0.01)


def test_maker_ttl_falls_back_to_market(tmp_path):
    """Flat rising bars never touch the BUY limit below close: after
    limit_ttl_bars=2 the strategy cancels and市 buys at market as TAKER."""
    from tml.backtest.data_prep import VENUE_NAME
    from tml.backtest.strategy import SignalStrategy

    cfg = make_config(tmp_path, pairs=("LTC/USDT",))
    bars = [(100, 100, 100, 100), (110, 110, 110, 110), (120, 120, 120, 120), (120, 120, 120, 120)]
    engine, inst = _maker_engine(cfg, bars)

    ns = lambda dt: int(dt.timestamp() * 1_000_000_000)  # noqa: E731
    signals = {(f"LTCUSDT.{VENUE_NAME}", ns(T0 + timedelta(hours=i))): +1.0 for i in range(4)}
    strategy = SignalStrategy(
        signals=signals,
        instruments=[inst],
        timeframe="1h",
        enter_threshold=0.0,
        exit_threshold=0.0,
        notional_per_pair=1000.0,
        initial_cash=100_000.0,
        execution=ExecutionConfig(mode="maker", maker_fee_bps=2.0, taker_fee_bps=10.0, slippage_bps=3.0, limit_ttl_bars=2),
    )
    engine.add_strategy(strategy)
    engine.run()
    engine.dispose()

    buys = [f for f in strategy.fills if f["side"] == "BUY"]
    assert len(buys) == 1, strategy.fills
    assert buys[0]["liquidity"] == "TAKER"  # TTL expired -> market fallback
    assert buys[0]["px"] == pytest.approx(120.0, abs=1e-3)
    assert buys[0]["commission_usdt"] == pytest.approx(10 * 120 * 0.0013, rel=0.02)


def test_signal_fold_alignment_guard():
    from tml.backtest.runner import assert_signals_in_folds

    fold = Fold(
        fold_id=0,
        test_start=T0,
        test_end=T0 + timedelta(hours=2),
        purge_embargo_cutoff=T0 - timedelta(hours=6),
    )
    inside = pl.DataFrame(
        {"ts": [T0, T0 + timedelta(hours=1)]}, schema_overrides={"ts": pl.Datetime("us", "UTC")}
    )
    assert_signals_in_folds(inside, [fold])  # no raise

    outside = pl.DataFrame(
        {"ts": [T0 + timedelta(hours=5)]}, schema_overrides={"ts": pl.Datetime("us", "UTC")}
    )
    with pytest.raises(RuntimeError, match="outside every CV test fold"):
        assert_signals_in_folds(outside, [fold])
