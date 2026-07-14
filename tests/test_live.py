"""Pure-logic tests for the live loop (no exchange access needed)."""

from __future__ import annotations

from decimal import Decimal

from tml.live.executor import LotFilter, load_env_file, quantize_qty
from tml.policy import decide, position_state


def test_decide_hysteresis_long_only():
    enter, exit_ = 0.5, 0.0
    assert decide(0.6, "flat", enter, exit_, False) == "open_long"
    assert decide(0.4, "flat", enter, exit_, False) == "hold"  # inside the band: no entry
    assert decide(0.4, "long", enter, exit_, False) == "hold"  # inside the band: keep holding
    assert decide(-0.1, "long", enter, exit_, False) == "close"
    assert decide(-0.9, "flat", enter, exit_, False) == "hold"  # shorts disabled
    assert decide(0.5, "flat", enter, exit_, False) == "hold"  # strict inequality at the edge
    assert decide(0.0, "long", enter, exit_, False) == "hold"


def test_decide_hysteresis_with_shorts():
    enter, exit_ = 0.5, 0.0
    assert decide(-0.6, "flat", enter, exit_, True) == "open_short"
    assert decide(-0.4, "flat", enter, exit_, True) == "hold"
    assert decide(-0.2, "short", enter, exit_, True) == "hold"  # negative score: stay short
    assert decide(+0.1, "short", enter, exit_, True) == "close"  # crossed -exit (0.0)
    assert decide(-0.5, "flat", enter, exit_, True) == "hold"  # strict inequality
    # long side is unchanged by allow_short
    assert decide(0.6, "flat", enter, exit_, True) == "open_long"
    assert decide(-0.1, "long", enter, exit_, True) == "close"


def test_gate_action_universe():
    from tml.live.tick import gate_action

    trad = {"BTCUSDT", "ETHUSDT"}
    e, x = 0.5, 0.0
    # current member, flat, strong score -> normal entry
    assert gate_action("BTCUSDT", 0.9, "flat", trad, True, e, x, True) == "open_long"
    assert gate_action("BTCUSDT", -0.9, "flat", trad, True, e, x, True) == "open_short"
    # non-member, flat -> never open
    assert gate_action("XRPUSDT", 0.9, "flat", trad, True, e, x, True) == "hold"
    assert gate_action("XRPUSDT", -0.9, "flat", trad, True, e, x, True) == "hold"
    # dropped-out holding (non-member, has position + filter) -> flatten
    assert gate_action("XRPUSDT", 0.9, "long", trad, True, e, x, True) == "close"
    assert gate_action("XRPUSDT", -0.9, "short", trad, True, e, x, True) == "close"
    # dropped-out holding but no lot filter (not on Bybit) -> can't act, hold
    assert gate_action("XRPUSDT", 0.9, "long", trad, False, e, x, True) == "hold"
    # current member holding exits normally on score reversal
    assert gate_action("BTCUSDT", -0.1, "long", trad, True, e, x, True) == "close"


def test_compute_snapshot(tmp_path):
    import json

    from tests.conftest import make_config
    from tml.live.snapshot import compute_snapshot
    from tml.live.state import journal_dir

    cfg = make_config(tmp_path)
    d = journal_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    # one closed LONG round trip: buy 10 @ 100, sell 10 @ 110 -> +100 gross,
    # minus 1.3+1.43 fees = +97.27 USDT = 9727 cents
    fills = [
        {"exec_id": "e1", "ts_ms": 1_000_000_000_000, "symbol": "BTCUSDT", "side": "Buy",
         "qty": 10.0, "px": 100.0, "fee_usdt": 1.30, "is_maker": True, "closed_pnl": 0.0, "order_link_id": ""},
        {"exec_id": "e2", "ts_ms": 1_000_000_000_000 + 86_400_000, "symbol": "BTCUSDT", "side": "Sell",
         "qty": 10.0, "px": 110.0, "fee_usdt": 1.43, "is_maker": False, "closed_pnl": 0.0, "order_link_id": ""},
    ]
    with (d / "fills.jsonl").open("w") as f:
        for x in fills:
            f.write(json.dumps(x) + "\n")
    with (d / "equity.jsonl").open("w") as f:
        f.write(json.dumps({"bar_ts": "2026-07-01T00:00:00+00:00", "total_equity": 100000.0}) + "\n")

    snap = compute_snapshot(cfg)
    assert snap["nTradesClosed"] == 1
    assert snap["realizedCents"] == 9727  # +97.27 USDT
    assert snap["winRatePct"] == 100.0
    assert snap["makerSharePct"] == 50.0  # 1 of 2 fills maker
    assert snap["openPositions"] == []
    assert snap["equityCents"] == 10_000_000
    assert snap["trades"][0]["symbol"] == "BTCUSDT" and snap["trades"][0]["side"] == "long"


def test_position_state():
    assert position_state(0.0, 100.0, 25.0) == "flat"
    assert position_state(1.0, 100.0, 25.0) == "long"
    assert position_state(-1.0, 100.0, 25.0) == "short"
    assert position_state(0.0001, 100.0, 25.0) == "flat"  # dust
    assert position_state(-0.0001, 100.0, 25.0) == "flat"  # dust short


def test_quantize_rounds_down_to_step():
    btc = LotFilter(
        qty_step=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal(5),
        tick_size=Decimal("0.10"),
    )
    assert quantize_qty(0.0834567, btc) == Decimal("0.083")
    # integer steps (ADA/DOGE): whole coins only
    doge = LotFilter(
        qty_step=Decimal("1"), min_qty=Decimal("1"), min_notional=Decimal(5),
        tick_size=Decimal("0.00001"),
    )
    assert quantize_qty(15789.73, doge) == Decimal("15789")
    # below min lot -> zero (no order), never rounded up
    assert quantize_qty(0.0004, btc) == Decimal(0)
    # float artifact safety
    assert str(
        quantize_qty(0.1 + 0.2, LotFilter(Decimal("0.1"), Decimal("0.1"), Decimal(5), Decimal("0.01")))
    ) == "0.3"


def test_quantize_price_stays_passive():
    from tml.live.executor import quantize_price

    btc = LotFilter(
        qty_step=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal(5),
        tick_size=Decimal("0.10"),
    )
    # buys round DOWN to the grid, sells round UP — never crossing closer
    assert quantize_price(64049.67, "Buy", btc) == Decimal("64049.6")
    assert quantize_price(64049.61, "Sell", btc) == Decimal("64049.7")
    assert quantize_price(64049.60, "Buy", btc) == Decimal("64049.6")  # already on grid


def test_order_link_id_fits_36_chars():
    for symbol in ("BTCUSDT", "DOGEUSDT", "LINKUSDT"):
        link = f"tml-{symbol}-2026071123-o"
        assert len(link) <= 36


def test_load_env_file_existing_env_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "BYBIT_DEMO_API_KEY=from_file\n# comment\n\nBYBIT_DEMO_API_SECRET=s3cret\n"
    )
    monkeypatch.setenv("BYBIT_DEMO_API_KEY", "from_env")
    monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
    load_env_file(tmp_path)
    import os

    assert os.environ["BYBIT_DEMO_API_KEY"] == "from_env"  # env var wins
    assert os.environ["BYBIT_DEMO_API_SECRET"] == "s3cret"  # file fills the gap
