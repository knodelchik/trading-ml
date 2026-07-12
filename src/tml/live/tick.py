"""One live paper-trading tick: data -> signals -> hysteresis -> Bybit demo.

Designed for an hourly scheduler AND for missed runs: every tick recomputes
targets from the freshest completed bar and reconciles against the
exchange's actual positions, so skipped hours self-heal. The orderLinkId
embeds the bar hour, giving natural idempotency if a tick double-fires.

Decision rule is the SAME hysteresis as the backtest (enter_threshold /
exit_threshold from [backtest]) applied to the persisted ensemble's score.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from tml.config import Config
from tml.live import state
from tml.policy import decide, position_state


def run_tick(cfg: Config) -> int:
    from tml.data.funding import run_funding_ingest
    from tml.data.ingest import run_ingest
    from tml.live import notify
    from tml.live.executor import BybitDemoExecutor, load_env_file
    from tml.live.signals import latest_scores

    load_env_file(cfg.root)  # telegram creds may be needed before the executor exists

    if state.kill_switch_active(cfg):
        print(f"tick: kill switch present ({state.kill_switch_path(cfg)}) — doing nothing")
        return 0

    # 1) freshest data (incremental; funding included via run_ingest)
    rc = run_ingest(cfg)
    if rc != 0:
        state.log_error(cfg, {"stage": "ingest", "rc": rc})
        return rc
    if not cfg.data.funding:
        run_funding_ingest(cfg)

    # 2) signals from the latest completed bar (journal failures: a bricked
    # model load must leave a trail, not just a dead systemd unit)
    try:
        scores, latest_ts = latest_scores(cfg)
    except Exception as e:
        state.log_error(cfg, {"stage": "signals", "error": repr(e)})
        notify.send(f"🚨 <b>tml tick впав</b> на етапі сигналів:\n<code>{e!r}</code>")
        raise
    now = datetime.now(UTC)
    bar = timedelta(milliseconds=cfg.data.bar_ms)
    if now - latest_ts > 2 * bar:
        state.log_error(cfg, {"stage": "staleness", "latest_ts": latest_ts.isoformat()})
        raise RuntimeError(f"latest bar {latest_ts} is more than 2 bars old — refusing to trade")

    # 3) exchange state
    ex = BybitDemoExecutor(cfg)
    filters = ex.lot_filters()
    ex.ensure_leverage_1x()
    positions = ex.positions()
    if not cfg.backtest.allow_short:
        short_positions = {s: q for s, q in positions.items() if q < 0}
        if short_positions:
            raise RuntimeError(f"unexpected SHORT positions on the account: {short_positions}")

    # 4) decide + submit
    hour_tag = latest_ts.strftime("%Y%m%d%H")
    exposure = sum(
        abs(q) * float(scores.filter(scores["pair"] == s)["close"][0])
        for s, q in positions.items()
        if s in set(scores["pair"].to_list())
    )
    submitted: list[tuple[str, str]] = []  # (symbol, order_id)
    tg_lines: list[str] = []
    for row in scores.iter_rows(named=True):
        symbol, px, score = row["pair"], float(row["close"]), float(row["score"])
        current_qty = positions.get(symbol, 0.0)
        pos_state = position_state(current_qty, px, cfg.live.min_order_notional_usdt)
        action = decide(
            score,
            pos_state,
            cfg.backtest.enter_threshold,
            cfg.backtest.exit_threshold,
            cfg.backtest.allow_short,
        )
        state.log_decision(
            cfg,
            {
                "bar_ts": latest_ts.isoformat(),
                "symbol": symbol,
                "score": score,
                "close_binance": px,
                "position_qty": current_qty,
                "state": pos_state,
                "action": action,
            },
        )
        if action == "hold":
            continue
        if action in ("open_long", "open_short"):
            budget = min(
                cfg.live.notional_per_pair_usdt,
                max(0.0, cfg.live.max_total_exposure_usdt - exposure),
            )
            qty = ex.quantize(symbol, budget / px, filters)
            if float(qty) * px < cfg.live.min_order_notional_usdt or qty < filters[symbol].min_qty:
                state.log_error(cfg, {"stage": "sizing", "symbol": symbol, "budget": budget})
                continue
            side = "Buy" if action == "open_long" else "Sell"
            reduce_only = False
            exposure += float(qty) * px
        else:  # close (long or short)
            qty = ex.quantize(symbol, abs(current_qty), filters)
            if qty <= 0:
                continue
            side = "Sell" if pos_state == "long" else "Buy"
            reduce_only = True
            exposure -= float(qty) * px

        link_id = f"tml-{symbol}-{hour_tag}-{'o' if side == 'Buy' else 'c'}"
        try:
            order_id = ex.market_order(symbol, side, qty, link_id, reduce_only)
            submitted.append((symbol, order_id))
            human = {
                "open_long": "🟢 відкрив ЛОНГ",
                "open_short": "🔴 відкрив ШОРТ",
                "close": "⚪ закрив " + ("лонг" if pos_state == "long" else "шорт"),
            }[action]
            tg_lines.append(
                f"{human} <b>{symbol}</b>: {qty} @ ~{px:g} (~{float(qty) * px:,.0f} USDT, скор {score:+.2f})"
            )
            state.log_order(
                cfg,
                {
                    "bar_ts": latest_ts.isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": str(qty),
                    "reduce_only": reduce_only,
                    "order_link_id": link_id,
                    "order_id": order_id,
                },
            )
        except Exception as e:  # noqa: BLE001 — journal, continue with other pairs
            state.log_error(cfg, {"stage": "order", "symbol": symbol, "error": repr(e)})
            print(f"tick: order failed for {symbol}: {e}")

    # 5) journal fills (market orders settle near-instantly)
    if submitted:
        time.sleep(2)
        for symbol, order_id in submitted:
            try:
                for fill in ex.executions_for(symbol, order_id):
                    state.log_order(cfg, {"fill": fill, "order_id": order_id})
            except Exception as e:  # noqa: BLE001
                state.log_error(cfg, {"stage": "fills", "symbol": symbol, "error": repr(e)})

    # 6) equity snapshot
    eq = ex.equity_usdt()
    state.log_equity(cfg, {"bar_ts": latest_ts.isoformat(), **eq, "n_orders": len(submitted)})

    # 7) telegram: every trade immediately; daily digest at 06:00 UTC (09:00 Київ)
    if tg_lines:
        notify.send(
            "🤖 <b>tml</b> @ " + latest_ts.strftime("%H:%M UTC") + "\n"
            + "\n".join(tg_lines)
            + f"\n💰 еквіті: {eq['total_equity']:,.0f} USDT"
        )
    else:
        from zoneinfo import ZoneInfo

        # daily digest on the 10:00 Kyiv bar (DST-proof)
        if latest_ts.astimezone(ZoneInfo("Europe/Kyiv")).hour == 10:
            n_open = len(ex.positions())
            notify.send(
                f"🌅 <b>tml щоденний звіт</b>\n"
                f"💰 еквіті: {eq['total_equity']:,.0f} USDT\n"
                f"📊 відкритих позицій: {n_open} з {len(cfg.data.pairs)} пар"
            )

    print(
        f"tick @ {latest_ts:%Y-%m-%d %H:%M} UTC: {len(submitted)} order(s), "
        f"equity {eq['total_equity']:.2f} USDT"
    )
    return 0
