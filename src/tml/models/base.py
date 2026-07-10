"""Model adapter protocol.

Fold discipline: an adapter is constructed fresh per fold, and EVERYTHING
that learns from data — scalers, encoders, feature selection, the model
itself — happens inside fit(), which only ever receives the fold's training
slice. No module-level fitted state is permitted.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ModelAdapter(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...
