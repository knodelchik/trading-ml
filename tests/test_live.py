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


def test_position_state():
    assert position_state(0.0, 100.0, 25.0) == "flat"
    assert position_state(1.0, 100.0, 25.0) == "long"
    assert position_state(-1.0, 100.0, 25.0) == "short"
    assert position_state(0.0001, 100.0, 25.0) == "flat"  # dust
    assert position_state(-0.0001, 100.0, 25.0) == "flat"  # dust short


def test_quantize_rounds_down_to_step():
    btc = LotFilter(qty_step=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal(5))
    assert quantize_qty(0.0834567, btc) == Decimal("0.083")
    # integer steps (ADA/DOGE): whole coins only
    doge = LotFilter(qty_step=Decimal("1"), min_qty=Decimal("1"), min_notional=Decimal(5))
    assert quantize_qty(15789.73, doge) == Decimal("15789")
    # below min lot -> zero (no order), never rounded up
    assert quantize_qty(0.0004, btc) == Decimal(0)
    # float artifact safety
    assert str(quantize_qty(0.1 + 0.2, LotFilter(Decimal("0.1"), Decimal("0.1"), Decimal(5)))) == "0.3"


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
