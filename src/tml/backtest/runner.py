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
from decimal import Decimal

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
        account_type=AccountType.MARGIN,  # perps; shorts allowed
        base_currency=USDT,  # single-currency USDT margin account
        starting_balances=[Money(cfg.backtest.initial_cash_usdt, USDT)],
        default_leverage=Decimal(1),  # MUST be explicit: MARGIN default is 10x
        # prob_fill_on_limit=0: resting limits fill only when price trades
        # strictly THROUGH them (never on a touch) — conservative maker model
        fill_model=FillModel(prob_slippage=0.0, prob_fill_on_limit=0.0, random_seed=cfg.model.lgbm.seed),
        fee_model=MakerTakerFeeModel(),  # stamps maker/taker fee per fill's liquidity side
    )

    from tml.universe import pairs_for

    trade_pairs = pairs_for(cfg)
    instruments = []
    skipped = 0
    for pair in trade_pairs:
        inst = make_instrument(cfg, pair)
        bars = bars_for_pair(cfg, pair, inst, window_start, window_end)
        if not bars:
            skipped += 1  # member only before the OOS window (or fully delisted pre-window)
            continue
        engine.add_instrument(inst)
        engine.add_data(bars)
        instruments.append(inst)
    if skipped:
        print(f"backtest: {skipped} symbols have no bars inside the OOS window — skipped")
    print(f"backtest: engine loaded, window [{window_start:%Y-%m-%d} .. {window_end:%Y-%m-%d})")

    signals = {
        (f"{row['pair']}.{VENUE_NAME}", int(row["ts_ns"])): float(row["score"])
        for row in oos.with_columns(ts_ns=pl.col("ts").dt.epoch(time_unit="ns")).iter_rows(named=True)
    }

    size_mult = None
    if cfg.meta.enabled:
        from tml.pipeline.meta import size_multipliers

        raw = size_multipliers(cfg) or {}
        size_mult = {(f"{pk}.{VENUE_NAME}", ts): m for (pk, ts), m in raw.items()}
        kept = sum(1 for m in size_mult.values() if m > 0)
        print(f"backtest: meta sizing on {len(size_mult)} signals, {kept} pass the p-filter")

    # point-in-time universe: a pair's membership can end mid-backtest —
    # flatten at its last in-universe bar instead of freezing the position
    force_close: set[tuple[str, int]] = set()
    if cfg.universe.enabled:
        bar_dur = pl.duration(milliseconds=cfg.data.bar_ms)
        exits = (
            oos.sort("pair", "ts")
            .with_columns(next_ts=pl.col("ts").shift(-1).over("pair"))
            .filter(pl.col("next_ts").is_null() | (pl.col("next_ts") > pl.col("ts") + bar_dur))
            .with_columns(ts_ns=pl.col("ts").dt.epoch(time_unit="ns"))
        )
        force_close = {
            (f"{r['pair']}.{VENUE_NAME}", int(r["ts_ns"])) for r in exits.iter_rows(named=True)
        }
        print(f"backtest: {len(force_close)} membership-exit points enforce flattening")
    strategy = SignalStrategy(
        signals=signals,
        instruments=instruments,
        timeframe=cfg.data.timeframe,
        enter_threshold=cfg.backtest.enter_threshold,
        exit_threshold=cfg.backtest.exit_threshold,
        # concurrent positions are bounded by universe size (or the static
        # pair count), not by every symbol that EVER was a member
        notional_per_pair=cfg.backtest.initial_cash_usdt
        / (cfg.universe.size if cfg.universe.enabled else len(cfg.data.pairs)),
        initial_cash=cfg.backtest.initial_cash_usdt,
        execution=cfg.execution,
        execution_delay_bars=cfg.backtest.execution_delay_bars,
        allow_short=cfg.backtest.allow_short,
        force_close=force_close,
        size_mult=size_mult,
    )
    engine.add_strategy(strategy)
    engine.run()

    # on_stop close_all_positions fills execute at the engine AFTER the
    # strategy is stopped, so on_order_filled never sees them — replay any
    # missing fills from the engine cache into the journal, or a run ending
    # long would fail the cash cross-check with a phantom mismatch
    _replay_missed_fills(engine, strategy)

    # --- results ---
    fills = pl.DataFrame(
        strategy.fills,
        schema={"ts": pl.Int64, "instrument": pl.String, "side": pl.String,
                "qty": pl.Float64, "px": pl.Float64, "commission_usdt": pl.Float64,
                "liquidity": pl.String},
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
    # testing and must deflate the Sharpe like any other configuration.
    # Same stride-H non-overlapping estimator as reporting's net candidate.
    eq_h = equity["equity"].gather_every(cfg.labels.horizon_bars).to_numpy()
    r_h = np.diff(eq_h) / eq_h[:-1]
    sr_h = float(np.nanmean(r_h) / np.nanstd(r_h, ddof=1)) if len(r_h) > 2 else None
    TrialsLedger(store.trials_path(cfg)).append(
        {
            "stage": "backtest",
            "config_hash": cfg.config_hash(),
            "enter_threshold": cfg.backtest.enter_threshold,
            "exit_threshold": cfg.backtest.exit_threshold,
            "execution_delay_bars": cfg.backtest.execution_delay_bars,
            "sr_per_period": sr_h,
            "net_total_return_pct": stats["total_return_pct"],
        }
    )
    print(json.dumps(stats, indent=2))
    return 0


def _replay_missed_fills(engine, strategy) -> None:
    from nautilus_trader.model.enums import LiquiditySide, OrderSide
    from nautilus_trader.model.events import OrderFilled

    seen = {
        (f["ts"], f["instrument"], f["side"], round(float(f["qty"]), 9), round(float(f["px"]), 9))
        for f in strategy.fills
    }
    for order in engine.cache.orders():
        for ev in order.events:
            if not isinstance(ev, OrderFilled):
                continue
            side = "BUY" if ev.order_side == OrderSide.BUY else "SELL"
            rec = {
                "ts": int(ev.ts_event),
                "instrument": str(ev.instrument_id),
                "side": side,
                "qty": float(ev.last_qty),
                "px": float(ev.last_px),
                "commission_usdt": float(ev.commission.as_double()) if ev.commission else 0.0,
                "liquidity": "MAKER" if ev.liquidity_side == LiquiditySide.MAKER else "TAKER",
            }
            qty_f, px_f = float(ev.last_qty), float(ev.last_px)
            key = (rec["ts"], rec["instrument"], side, round(qty_f, 9), round(px_f, 9))
            if key not in seen:
                strategy.fills.append(rec)
                seen.add(key)


def _funding_flows(cfg: Config, grid: pl.DataFrame) -> pl.DataFrame:
    """Funding cash flows for held perp positions, applied at the next bar.

    Long positions PAY positive funding (flow = -pos * close * rate); shorts
    receive it. The nautilus engine does not simulate funding, so this
    post-processing uses the same settlement history the features use.
    Returns (ts, funding_flow) aggregated per hourly mark.
    """
    from tml.data.funding import scan_funding

    frames = [lf.collect() for p in cfg.data.pairs if (lf := scan_funding(cfg, p)) is not None]
    if not frames:
        return pl.DataFrame(
            schema={"ts": pl.Datetime("us", "UTC"), "funding_flow": pl.Float64()}
        )
    g = grid.select("ts", "pair", "pos", "close").sort("pair", "ts")
    fund = (
        pl.concat(frames)
        .filter((pl.col("ts") >= g["ts"].min()) & (pl.col("ts") <= g["ts"].max()))
        .sort("pair", "ts")
        .join_asof(g, on="ts", by="pair", strategy="backward")
        .drop_nulls("pos")
        .with_columns(flow=-(pl.col("pos") * pl.col("close") * pl.col("rate")))
        # settlement (with ms jitter) hits the account by the NEXT hourly mark
        .with_columns(apply_ts=pl.col("ts").dt.truncate("1h") + pl.duration(hours=1))
    )
    return fund.group_by("apply_ts").agg(pl.col("flow").sum().alias("funding_flow")).rename(
        {"apply_ts": "ts"}
    )


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
    funding = _funding_flows(cfg, grid)
    equity = (
        mtm_by_ts.join(cash_by_ts, on="ts", how="left")
        .join(funding, on="ts", how="left")
        .sort("ts")
        .with_columns(
            cash=cfg.backtest.initial_cash_usdt + pl.col("cash_flow").fill_null(0.0).cum_sum(),
            funding_cum=pl.col("funding_flow").fill_null(0.0).cum_sum(),
        )
        .with_columns(equity=pl.col("cash") + pl.col("mtm") + pl.col("funding_cum"))
        .select("ts", "cash", "mtm", "funding_cum", "equity")
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
    # the MARGIN engine performs NO balance checks and never liquidates —
    # a silent mid-run blow-up must fail loudly here instead
    if float(eq.min()) <= 0:
        raise RuntimeError(
            f"reconstructed equity went non-positive (min {eq.min():.2f}) — "
            "the margin engine would not have halted; treat this run as a blow-up"
        )
    r = np.diff(eq) / eq[:-1]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak

    gross_traded = float((fills["qty"] * fills["px"]).sum())
    total_fees = float(fills["commission_usdt"].sum())
    taker_rate = (cfg.execution.taker_fee_bps + cfg.execution.slippage_bps) / 10_000
    maker_rate = cfg.execution.maker_fee_bps / 10_000
    expected_fees = float(
        fills.with_columns(
            expected=pl.col("qty")
            * pl.col("px")
            * pl.when(pl.col("liquidity") == "MAKER").then(maker_rate).otherwise(taker_rate)
        )["expected"].sum()
    )
    fee_check = abs(total_fees - expected_fees) / max(total_fees, 1e-9)
    if fee_check > 0.01:
        raise RuntimeError(
            f"fee sanity check failed: journal fees {total_fees:.2f} vs "
            f"per-liquidity expectation {expected_fees:.2f}"
        )
    maker_share = float(
        fills.select((pl.col("liquidity") == "MAKER").cast(pl.Float64).mean()).item() or 0.0
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
        "funding_net_usdt": round(float(equity["funding_cum"][-1]), 2),
        "allow_short": cfg.backtest.allow_short,
        "execution_mode": cfg.execution.mode,
        "maker_fill_share": round(maker_share, 4),
        "cost_model": (
            f"maker {cfg.execution.maker_fee_bps}bps / taker {cfg.execution.taker_fee_bps}bps "
            f"+ slippage {cfg.execution.slippage_bps}bps on taker fills"
        ),
        "missing_signal_bars": missing_signals,
        "engine_vs_journal_cash_mismatch_frac": round(cash_mismatch, 6),
        "hours": int(ts.len()),
        "generated_at": datetime.now(UTC).isoformat(),
    }
