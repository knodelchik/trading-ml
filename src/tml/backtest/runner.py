"""Backtest runner: one portfolio engine over the whole OOS window.

Leakage guard (trap 6): before the engine starts, every OOS prediction
timestamp is asserted to lie inside a CV test fold, and bars are only fed
for the OOS window — the backtest cannot touch data the models trained on.

Verification built in: the equity curve is reconstructed independently from
the strategy's own fill journal + close prices (pure polars math) and the
final cash is cross-checked against the engine's account balance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from tml.config import Config
from tml.data import store
from tml.validation.metrics import sharpe
from tml.validation.splits import read_folds
from tml.validation.trials import TrialsLedger

HOURS_PER_YEAR = 24 * 365


def assert_signals_in_folds(oos: pl.DataFrame, folds) -> None:
    """Trap 6 guard: every OOS prediction must lie inside a CV test fold."""
    in_any_fold = pl.any_horizontal(
        [(pl.col("ts") >= f.test_start) & (pl.col("ts") < f.test_end) for f in folds]
    )
    n_outside = oos.filter(~in_any_fold).height
    if n_outside:
        raise RuntimeError(
            f"{n_outside} OOS predictions fall outside every CV test fold — "
            "retraining-schedule mismatch, refusing to backtest"
        )


def run_backtest(cfg: Config) -> int:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money

    from tml.backtest.data_prep import VENUE_NAME, bars_for_pair, closes_frame, make_instrument
    from tml.backtest.strategy import SignalStrategy

    oos = pl.read_parquet(store.oos_path(cfg))
    folds = read_folds(store.folds_path(cfg))

    # --- signal-fold alignment guard (runs BEFORE the engine) ---
    assert_signals_in_folds(oos, folds)
    window_start = min(f.test_start for f in folds)
    window_end = max(f.test_end for f in folds)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
        )
    )
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None,  # multi-currency CASH account
        starting_balances=[Money(cfg.backtest.initial_cash_usdt, USDT)],
        fill_model=FillModel(prob_slippage=0.0, random_seed=cfg.model.lgbm.seed),
        fee_model=MakerTakerFeeModel(),  # reads effective taker (incl. slippage bps) off instruments
    )

    instruments = []
    for pair in cfg.data.pairs:
        inst = make_instrument(cfg, pair)
        engine.add_instrument(inst)
        bars = bars_for_pair(cfg, pair, inst, window_start, window_end)
        engine.add_data(bars)
        instruments.append(inst)
    print(f"backtest: engine loaded, window [{window_start:%Y-%m-%d} .. {window_end:%Y-%m-%d})")

    signals = {
        (f"{row['pair']}.{VENUE_NAME}", int(row["ts_ns"])): float(row["score"])
        for row in oos.with_columns(ts_ns=pl.col("ts").dt.epoch(time_unit="ns")).iter_rows(named=True)
    }
    strategy = SignalStrategy(
        signals=signals,
        instruments=instruments,
        timeframe=cfg.data.timeframe,
        enter_threshold=cfg.backtest.enter_threshold,
        exit_threshold=cfg.backtest.exit_threshold,
        notional_per_pair=cfg.backtest.initial_cash_usdt / len(cfg.data.pairs),
        initial_cash=cfg.backtest.initial_cash_usdt,
        fee_rate=(cfg.backtest.taker_fee_bps + cfg.backtest.slippage_bps) / 10_000,
        execution_delay_bars=cfg.backtest.execution_delay_bars,
    )
    engine.add_strategy(strategy)
    engine.run()

    # --- results ---
    fills = pl.DataFrame(
        strategy.fills,
        schema={"ts": pl.Int64, "instrument": pl.String, "side": pl.String,
                "qty": pl.Float64, "px": pl.Float64, "commission_usdt": pl.Float64},
    )
    account = engine.portfolio.account(Venue(VENUE_NAME))
    engine_final_usdt = float(account.balance_total(USDT).as_double())
    engine.dispose()

    if fills.is_empty():
        raise RuntimeError("backtest produced zero fills — signals or bar wiring is broken")

    equity = _equity_curve(cfg, fills, closes_frame(cfg, window_start, window_end))
    stats = _stats(cfg, fills, equity, engine_final_usdt, strategy.missing_signals, window_start, window_end)

    out = store.backtest_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    fills.write_parquet(out / "fills.parquet")
    equity.write_parquet(out / "equity.parquet")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    # every backtest variant is a trial: threshold choices are multiple
    # testing and must deflate the Sharpe like any other configuration
    r = np.diff(equity["equity"].to_numpy()) / equity["equity"].to_numpy()[:-1]
    hourly_sr = float(np.nanmean(r) / np.nanstd(r, ddof=1)) if len(r) > 2 else None
    TrialsLedger(store.trials_path(cfg)).append(
        {
            "stage": "backtest",
            "config_hash": cfg.config_hash(),
            "enter_threshold": cfg.backtest.enter_threshold,
            "exit_threshold": cfg.backtest.exit_threshold,
            "execution_delay_bars": cfg.backtest.execution_delay_bars,
            # per-24h-period units, comparable with CV trial records
            "sr_per_period": hourly_sr * (24**0.5) if hourly_sr is not None else None,
            "net_total_return_pct": stats["total_return_pct"],
        }
    )
    print(json.dumps(stats, indent=2))
    return 0


def _equity_curve(cfg: Config, fills: pl.DataFrame, closes: pl.DataFrame) -> pl.DataFrame:
    """Mark-to-market equity per hour from the fill journal (pure polars/numpy)."""
    f = (
        fills.with_columns(
            ts=pl.from_epoch("ts", time_unit="ns").dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
            pair=pl.col("instrument").str.split(".").list.first(),
            signed_qty=pl.when(pl.col("side") == "BUY").then(pl.col("qty")).otherwise(-pl.col("qty")),
            cash_flow=pl.when(pl.col("side") == "BUY")
            .then(-pl.col("qty") * pl.col("px"))
            .otherwise(pl.col("qty") * pl.col("px"))
            - pl.col("commission_usdt"),
        )
        .group_by("ts", "pair")
        .agg(pl.col("signed_qty").sum(), pl.col("cash_flow").sum())
    )
    grid = closes.join(f.select("ts", "pair", "signed_qty"), on=["ts", "pair"], how="left")
    grid = grid.with_columns(
        pos=pl.col("signed_qty").fill_null(0.0).cum_sum().over("pair"),
    ).with_columns(mtm=pl.col("pos") * pl.col("close"))
    mtm_by_ts = grid.group_by("ts").agg(pl.col("mtm").sum()).sort("ts")

    cash_by_ts = (
        f.group_by("ts").agg(pl.col("cash_flow").sum()).sort("ts")
    )
    equity = (
        mtm_by_ts.join(cash_by_ts, on="ts", how="left")
        .sort("ts")
        .with_columns(cash=cfg.backtest.initial_cash_usdt + pl.col("cash_flow").fill_null(0.0).cum_sum())
        .with_columns(equity=pl.col("cash") + pl.col("mtm"))
        .select("ts", "cash", "mtm", "equity")
    )
    return equity


def _stats(
    cfg: Config,
    fills: pl.DataFrame,
    equity: pl.DataFrame,
    engine_final_usdt: float,
    missing_signals: int,
    window_start: datetime,
    window_end: datetime,
) -> dict:
    eq = equity["equity"].to_numpy()
    ts = equity["ts"]
    r = np.diff(eq) / eq[:-1]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak

    gross_traded = float((fills["qty"] * fills["px"]).sum())
    total_fees = float(fills["commission_usdt"].sum())
    fee_rate = (cfg.backtest.taker_fee_bps + cfg.backtest.slippage_bps) / 10_000
    fee_check = abs(total_fees - gross_traded * fee_rate) / max(total_fees, 1e-9)
    if fee_check > 0.01:
        raise RuntimeError(
            f"fee sanity check failed: journal fees {total_fees:.2f} vs "
            f"turnover*rate {gross_traded * fee_rate:.2f}"
        )
    # engine's final cash must match our reconstruction once flat at end;
    # a large mismatch means the engine halted mid-run (e.g. negative
    # balance) or fills were lost — the run is not trustworthy
    our_final_cash = float(equity["cash"][-1])
    cash_mismatch = abs(engine_final_usdt - our_final_cash) / cfg.backtest.initial_cash_usdt
    if cash_mismatch > 1e-3:
        raise RuntimeError(
            f"engine final USDT {engine_final_usdt:.2f} vs journal {our_final_cash:.2f} — "
            "mismatch indicates a halted or corrupted run; check engine ERROR logs"
        )

    years = (window_end - window_start).days / 365.25
    final_eq = float(eq[-1])
    return {
        "window": f"{window_start.isoformat()} .. {window_end.isoformat()}",
        "execution_delay_bars": cfg.backtest.execution_delay_bars,
        "initial_cash_usdt": cfg.backtest.initial_cash_usdt,
        "final_equity_usdt": round(final_eq, 2),
        "total_return_pct": round((final_eq / cfg.backtest.initial_cash_usdt - 1) * 100, 2),
        "cagr_pct": round(((final_eq / cfg.backtest.initial_cash_usdt) ** (1 / years) - 1) * 100, 2),
        "sharpe_net_annualized": round(sharpe(r, HOURS_PER_YEAR), 3),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "n_fills": fills.height,
        "gross_traded_usdt": round(gross_traded, 0),
        "turnover_x_initial": round(gross_traded / cfg.backtest.initial_cash_usdt, 1),
        "total_costs_usdt": round(total_fees, 2),
        "cost_model": f"taker {cfg.backtest.taker_fee_bps}bps + slippage {cfg.backtest.slippage_bps}bps baked into fee",
        "missing_signal_bars": missing_signals,
        "engine_vs_journal_cash_mismatch_frac": round(cash_mismatch, 6),
        "hours": int(ts.len()),
        "generated_at": datetime.now(UTC).isoformat(),
    }
