"""Purged walk-forward cross-validation with embargo.

Folds walk forward over the unique-timestamp index: expanding train window,
fixed-width test windows. A training sample survives only if its label ends
strictly before `test_start - embargo`; that single condition implements
both purging (labels overlapping the test window) and the embargo margin.

This module is the methodological core of the project. Its invariants are
enforced by tests/test_splits.py and consumed at runtime by the backtest's
signal-fold alignment guard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from tml.config import CVConfig


@dataclass(frozen=True)
class Fold:
    fold_id: int
    test_start: datetime  # inclusive
    test_end: datetime  # exclusive
    purge_embargo_cutoff: datetime  # train samples need label_end_ts < this

    def train_expr(self, end_col: str = "label_end_ts") -> pl.Expr:
        """Purge + embargo in one condition; `end_col` selects which
        horizon's label end drives the purge in multi-horizon training."""
        return pl.col(end_col) < self.purge_embargo_cutoff

    def test_expr(self) -> pl.Expr:
        return (pl.col("ts") >= self.test_start) & (pl.col("ts") < self.test_end)

    def to_json(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "purge_embargo_cutoff": self.purge_embargo_cutoff.isoformat(),
        }


class PurgedWalkForwardCV:
    def __init__(self, cv: CVConfig, bar_ms: int):
        self.cv = cv
        self.bar = timedelta(milliseconds=bar_ms)

    def split(self, ts_index: pl.Series) -> list[Fold]:
        """ts_index: sorted unique bar-close timestamps across all pairs."""
        ts = ts_index.sort()
        n = len(ts)
        folds: list[Fold] = []
        i = self.cv.train_min_bars
        fold_id = 0
        while i + self.cv.test_span_bars <= n:
            test_start = ts[i]
            test_end = ts[i + self.cv.test_span_bars - 1] + self.bar
            cutoff = test_start - self.cv.embargo_bars * self.bar
            folds.append(
                Fold(
                    fold_id=fold_id,
                    test_start=test_start,
                    test_end=test_end,
                    purge_embargo_cutoff=cutoff,
                )
            )
            fold_id += 1
            i += self.cv.step_bars
        leftover = n - (self.cv.train_min_bars + (fold_id - 1) * self.cv.step_bars + self.cv.test_span_bars) if folds else n
        if leftover > 0:
            print(f"cv: {leftover} trailing bars do not fill a full test window and are unused")
        return folds


def write_folds(folds: list[Fold], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([f.to_json() for f in folds], indent=2))


def read_folds(path: Path) -> list[Fold]:
    raw = json.loads(path.read_text())
    return [
        Fold(
            fold_id=r["fold_id"],
            test_start=datetime.fromisoformat(r["test_start"]),
            test_end=datetime.fromisoformat(r["test_end"]),
            purge_embargo_cutoff=datetime.fromisoformat(r["purge_embargo_cutoff"]),
        )
        for r in raw
    ]
