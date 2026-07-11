"""Trading policy: the ONE place where scores become position decisions.

Both the nautilus backtest strategy and the live Bybit tick import these
functions, so the traded rule cannot silently diverge between research and
production.

Symmetric hysteresis on the vol-scaled score:
    flat  & score > +enter          -> open_long
    flat  & score < -enter (shorts) -> open_short
    long  & score < exit            -> close
    short & score > -exit           -> close
No direct long<->short flips: a flip takes two decisions (close now, and
possibly open the other way on a later bar if the signal persists) — this
keeps every order's economics identical to the backtest's assumptions.
"""

from __future__ import annotations

from typing import Literal

State = Literal["long", "short", "flat"]
Action = Literal["open_long", "open_short", "close", "hold"]


def position_state(qty: float, px: float, min_notional: float) -> State:
    """Signed base quantity -> discrete state (dust counts as flat)."""
    notional = qty * px
    if notional >= min_notional:
        return "long"
    if notional <= -min_notional:
        return "short"
    return "flat"


def decide(
    score: float,
    state: State,
    enter_threshold: float,
    exit_threshold: float,
    allow_short: bool,
) -> Action:
    if state == "flat":
        if score > enter_threshold:
            return "open_long"
        if allow_short and score < -enter_threshold:
            return "open_short"
        return "hold"
    if state == "long":
        return "close" if score < exit_threshold else "hold"
    # short
    return "close" if score > -exit_threshold else "hold"
