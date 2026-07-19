"""Tick-time basis refresh — loud on failure, never silently stale.

Funding's in-tick refresh (run_funding_ingest) lets a transient per-pair
error crash the whole tick. This one does one better on both failure modes:
  * a pair whose REST pull fails is journaled via log_error and left
    stale — the staleness cutoff in features/basis.py then serves flagged
    zeros (basis_known=0) for it, so the model sees "no basis regime"
    instead of a frozen last value, and every other pair still trades;
  * after the pull, an explicit freshness audit journals any pair whose
    stored series still lags, so a stale feature can never pass unnoticed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from tml.config import Config
from tml.data.basis import PerpUnlisted, refresh_basis_pair, scan_basis
from tml.features.basis import MAX_STALE_BARS
from tml.live import state


def run_basis_refresh(cfg: Config) -> int:
    from tml.universe import pairs_for

    pairs = pairs_for(cfg)
    total = 0
    unlisted: set[str] = set()
    failed: list[str] = []
    for pair in pairs:
        try:
            total += refresh_basis_pair(cfg, pair)
        except PerpUnlisted:
            unlisted.add(pair)  # permanent (no perp), not stale — features already 0-fill
        except Exception as e:  # noqa: BLE001 — one pair must not kill the tick
            failed.append(pair)
            state.log_error(cfg, {"stage": "basis_refresh", "pair": pair, "error": repr(e)})
            print(f"basis refresh FAILED for {pair}: {e!r} — falls back to basis_known=0")

    # freshness audit: whatever the reason, a series older than the feature
    # layer's staleness cutoff is journaled loudly; those pairs score with
    # basis_known=0 this tick rather than reusing a frozen basis value
    now = datetime.now(UTC)
    max_age = timedelta(milliseconds=MAX_STALE_BARS * cfg.data.bar_ms)
    stale = []
    for pair in pairs:
        if pair in unlisted:
            continue
        lf = scan_basis(cfg, pair)
        if lf is None:
            continue
        last = lf.select(pl.col("ts").max()).collect().item()
        if last is None or now - last > max_age:
            stale.append(pair)
    if stale:
        state.log_error(cfg, {"stage": "basis_stale", "pairs": stale})
        print(f"basis refresh: {len(stale)} pair(s) STALE -> basis_known=0 this tick: {stale}")
    print(
        f"basis refresh: +{total} premium bars, {len(unlisted)} unlisted, "
        f"{len(failed)} failed, {len(stale)} stale"
    )
    return 0
