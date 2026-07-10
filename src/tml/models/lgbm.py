"""LightGBM regression adapter (deterministic configuration).

deterministic + force_row_wise + fixed num_threads + fixed seed make two
runs on identical data bit-identical, which test_determinism.py asserts.
"""

from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMRegressor

from tml.config import LGBMConfig


class LGBMAdapter:
    def __init__(self, cfg: LGBMConfig):
        self.cfg = cfg
        self.model: LGBMRegressor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self.model = LGBMRegressor(
            objective="regression",
            num_leaves=self.cfg.num_leaves,
            learning_rate=self.cfg.learning_rate,
            n_estimators=self.cfg.n_estimators,
            min_child_samples=self.cfg.min_data_in_leaf,
            colsample_bytree=self.cfg.feature_fraction,
            random_state=self.cfg.seed,
            n_jobs=self.cfg.num_threads,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        # plain numpy in fit AND predict (passing feature_name here would make
        # sklearn warn on every numpy predict)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() must be called first"
        with warnings.catch_warnings():
            # lightgbm 4.6 registers auto-generated column names at fit even
            # for numpy input, so sklearn's validate_data warns on every
            # numpy predict; upstream noise, not a real name mismatch
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names", category=UserWarning
            )
            return self.model.predict(x)
