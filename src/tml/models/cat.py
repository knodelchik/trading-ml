"""CatBoost regression adapter (same fold protocol as LGBMAdapter)."""

from __future__ import annotations

from typing import Any

import numpy as np

from tml.config import CatBoostConfig


class CatBoostAdapter:
    def __init__(self, cfg: CatBoostConfig):
        self.cfg = cfg
        self.model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor(
            loss_function="RMSE",
            depth=self.cfg.depth,
            iterations=self.cfg.iterations,
            learning_rate=self.cfg.learning_rate,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            random_seed=self.cfg.seed,
            thread_count=self.cfg.thread_count,
            allow_writing_files=False,
            verbose=False,
        )
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() must be called first"
        return self.model.predict(x)
