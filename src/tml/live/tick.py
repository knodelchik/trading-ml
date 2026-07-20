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
import uuid
from datetime import UTC, datetime, timedelta

from tml.config import Config
from tml.live import state
from tml.policy import Action, State, decide, position_state

# passive offset beyond the touch for maker opens: demo market data lags the
# demo matching engine by seconds, and a quote AT the touch crosses whenever
# the market moved within that lag (PostOnly insta-cancel). 2bps absorbs the
# typical lag move and still saves ~9bps vs a taker entry when filled.
MAKER_OFFSET_BPS = 2.0


def _recent_open_attempts(cfg: Config, symbol: str, latest_ts, ttl_bars: int) -> int:
    """Count journaled OPEN attempts for `symbol` in the last `ttl_bars` bar
    hours (excluding the current bar). Mirrors the backtest's limit-TTL: an
    insta-cancelled PostOnly never rests on the book, so without this counter
    the limit->TTL->market fallback would never trigger live."""
    cutoff = (latest_ts - timedelta(hours=ttl_bars)).isoformat()
    n = 0
    for r in state.read_journal(cfg, "orders"):
        if (
            r.get("symbol") == symbol
            and not r.get("reduce_only", True)
            and cutoff <= r.get("bar_ts", "") < latest_ts.isoformat()
        ):
            n += 1
    return n


def gate_action(
    symbol: str,
    score: float,
    pos_state: State,
    tradable: set[str],
    has_filter: bool,
    enter: float,
    exit_: float,
    allow_short: bool,
) -> Action:
    """Universe gating on top of the shared hysteresis: current members
    (that trade on Bybit) decide normally; a holding in a symbol that left
    the universe is flattened; a non-member with no position is left alone.
    Never OPENS a non-member."""
    if symbol not in tradable:
        if pos_state == "flat" or not has_filter:
            return "hold"
        return "close"
    return decide(score, pos_state, enter, exit_, allow_short)


def run_tick(cfg: Config) -> int:
    from tml.data.funding import run_funding_ingest
    from tml.data.ingest import run_ingest
    from tml.data.store import pair_key
    from tml.live import notify
    from tml.live.executor import BybitDemoExecutor, load_env_file
    from tml.live.signals import latest_scores
    from tml.universe import current_members

    load_env_file(cfg.root)  # telegram creds may be needed before the executor exists

    if state.kill_switch_active(cfg):
        print(f"tick: kill switch present ({state.kill_switch_path(cfg)}) — doing nothing")
        return 0

    # 1) freshest data (incremental). run_ingest fetches only OHLCV bars, so
    # funding/taker must be refreshed explicitly when enabled — otherwise
    # their features silently go stale in live (this was an inverted-guard bug).
    rc = run_ingest(cfg)
    if rc != 0:
        state.log_error(cfg, {"stage": "ingest", "rc": rc})
        return rc
    if cfg.data.funding:
        run_funding_ingest(cfg)
    if cfg.data.taker:
        from tml.data.taker import run_taker_ingest

        run_taker_ingest(cfg)
    if cfg.data.basis:
        from tml.live.refresh import run_basis_refresh

        run_basis_refresh(cfg)

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

    # 3) exchange state. In universe mode only the CURRENT month's members
    # (that also trade on Bybit) are actively tradable; positions in symbols
    # that dropped out of the universe are flattened (mirrors the backtest's
    # membership-exit force-close).
    ex = BybitDemoExecutor(cfg)
    positions = ex.positions()
    if cfg.universe.enabled:
        members = current_members(cfg)
        member_syms = {pair_key(p) for p in members}
        # filters for members (to enter) plus any held symbol (to exit) —
        # a dropped-out holding is no longer a member but still needs a lot
        held_pairs = [f"{s[:-4]}/USDT" for s in positions if s not in member_syms]
        filters = ex.lot_filters(members + held_pairs, require_all=False)
        tradable = set(member_syms) & set(filters)  # members that trade on Bybit
    else:
        filters = ex.lot_filters(list(cfg.data.pairs), require_all=True)
        tradable = set(filters)
    ex.ensure_leverage_1x(sorted(set(filters)))

    # 3a) manage maker limits resting from previous ticks: cancel after TTL
    # and fall back to market for the unfilled remainder; symbols with a
    # resting order are excluded from this tick's decisions (backtest parity)
    suppressed: set[str] = set()
    bar_ms = cfg.data.bar_ms
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    for o in ex.open_orders():
        if not o["order_link_id"].startswith("tml-"):
            continue  # not ours — leave alone
        suppressed.add(o["symbol"])
        age_bars = (now_ms - o["created_ms"]) / bar_ms
        if age_bars < cfg.execution.limit_ttl_bars:
            continue
        try:
            ex.cancel(o["symbol"], o["order_id"])
            remaining = ex.quantize(o["symbol"], o["qty"] - o["cum_exec_qty"], filters)
            state.log_order(
                cfg,
                {"ttl_cancel": o["order_link_id"], "remaining": str(remaining), "age_bars": round(age_bars, 2)},
            )
            if float(remaining) > 0:
                fb_link = f"{o['order_link_id'][:33]}-f"
                order_id = ex.market_order(
                    o["symbol"], o["side"], remaining, fb_link, o["reduce_only"]
                )
                state.log_order(
                    cfg,
                    {"fallback_market": fb_link, "symbol": o["symbol"], "side": o["side"],
                     "qty": str(remaining), "order_id": order_id},
                )
        except Exception as e:  # noqa: BLE001
            state.log_error(cfg, {"stage": "ttl_manage", "symbol": o["symbol"], "error": repr(e)})
    if not cfg.backtest.allow_short:
        short_positions = {s: q for s, q in positions.items() if q < 0}
        if short_positions:
            raise RuntimeError(f"unexpected SHORT positions on the account: {short_positions}")

    # 4) decide + submit. Size each position at the backtest's notional
    # (initial_cash / universe_size) so live matches the validated sizing.
    n_slots = cfg.universe.size if cfg.universe.enabled else len(cfg.data.pairs)
    notional_per_pair = min(
        cfg.live.notional_per_pair_usdt, cfg.backtest.initial_cash_usdt / n_slots
    )
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
        if symbol in suppressed:
            state.log_decision(
                cfg,
                {"bar_ts": latest_ts.isoformat(), "symbol": symbol, "score": score,
                 "action": "suppressed_working_order"},
            )
            continue
        current_qty = positions.get(symbol, 0.0)
        pos_state = position_state(current_qty, px, cfg.live.min_order_notional_usdt)
        action = gate_action(
            symbol,
            score,
            pos_state,
            tradable,
            symbol in filters,
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
                notional_per_pair,
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

        # unique per ATTEMPT: Bybit rejects a duplicate orderLinkId even after
        # the prior order was cancelled, so a bar-hour id blocked retries once
        # a PostOnly got cancelled. Idempotency comes from position + resting-
        # order reconciliation, not the id.
        link_id = f"tml-{symbol}-{hour_tag}-{'o' if side == 'Buy' else 'c'}-{uuid.uuid4().hex[:5]}"
        try:
            # CLOSES always go to market: an exit we've decided on must
            # complete reliably (a passive close-limit that never fills leaves
            # us stuck in the position). Only OPENS use maker (patient, saves
            # fees). Demo market data lags the demo matching engine by seconds,
            # so a quote AT the touch kept crossing in moving markets
            # (EC_PostOnlyWillTakeLiquidity 6/6 after deploy) — quote a 2bps
            # passive offset beyond the touch to absorb the lag. And mirror the
            # backtest's TTL semantics: an insta-cancelled PostOnly never rests,
            # so after limit_ttl_bars consecutive failed open attempts the next
            # one goes to market (the backtest's limit->TTL->market fallback).
            failed_attempts = _recent_open_attempts(cfg, symbol, latest_ts, cfg.execution.limit_ttl_bars)
            use_maker = cfg.execution.mode == "maker" and not reduce_only
            if use_maker and failed_attempts >= cfg.execution.limit_ttl_bars:
                use_maker = False  # TTL exhausted across ticks -> market entry
            if use_maker:
                from tml.live.executor import quantize_price

                f = filters[symbol]
                bid, ask = ex.best_bid_ask(symbol)
                off = 1 + MAKER_OFFSET_BPS / 10_000 * (-1 if side == "Buy" else 1)
                raw_px = (bid if side == "Buy" else ask) * off
                limit_px = quantize_price(raw_px, side, f)
                order_id = ex.limit_post_only(symbol, side, qty, limit_px, link_id, reduce_only)
                verb = "🟢 виставив лімітку ЛОНГ" if action == "open_long" else "🔴 виставив лімітку ШОРТ"
            else:
                order_id = ex.market_order(symbol, side, qty, link_id, reduce_only)
                verb = {
                    "open_long": "🟢 відкрив ЛОНГ",
                    "open_short": "🔴 відкрив ШОРТ",
                    "close": "⚪ закрив " + ("лонг" if pos_state == "long" else "шорт"),
                }[action]
            submitted.append((symbol, order_id))
            tg_lines.append(
                f"{verb} <b>{symbol}</b>: {qty} @ ~{px:g} (~{float(qty) * px:,.0f} USDT, скор {score:+.2f})"
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

    # 5) durably journal ALL executions since last tick (maker limits fill
    # LATER than submission, so checking only just-submitted orders misses
    # them; and Bybit demo keeps executions only ~7 days). Dedupe by execId.
    if submitted:
        time.sleep(2)  # let just-submitted market orders settle before we query
    try:
        last = state.last_fill_ms(cfg)
        # first run (or gap): look back 3 days, within Bybit's 7-day window
        start_ms = (last - 3600_000) if last else (int(now.timestamp() * 1000) - 3 * 86_400_000)
        n_new = state.journal_executions(cfg, ex.executions_since(start_ms))
        if n_new:
            print(f"tick: journaled {n_new} new fills")
    except Exception as e:  # noqa: BLE001
        state.log_error(cfg, {"stage": "fills", "error": repr(e)})

    # 6) equity snapshot
    eq = ex.equity_usdt()
    state.log_equity(cfg, {"bar_ts": latest_ts.isoformat(), **eq, "n_orders": len(submitted)})

    # 6a) push dashboard snapshot to lifeos (best-effort, never breaks a tick)
    try:
        from tml.live.snapshot import push_snapshot

        push_snapshot(cfg)
    except Exception as e:  # noqa: BLE001
        state.log_error(cfg, {"stage": "push", "error": repr(e)})

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
