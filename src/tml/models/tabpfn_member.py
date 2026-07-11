"""TabPFN ensemble member (in-context tabular foundation model).

Facts (verified against installed tabpfn 8.0.8):
  * hard cap 10k training samples — the adapter feeds the most RECENT rows,
    a deliberate recency-weighted view that complements the boosters'
    full-history view;
  * inference_precision must be pinned to float32: the MPS autocast path is
    only ~1e-6 reproducible, which would break bit-identical determinism;
  * fit() is cheap (preprocessing); the transformer runs at predict() —
    chunked to bound the n_test x n_train attention memory;
  * first use downloads ~73 MB of weights to ~/.cache/tabpfn/.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tml.config import TabPFNConfig


class TabPFNAdapter:
    def __init__(self, cfg: TabPFNConfig):
        self.cfg = cfg
        self.model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        import torch
        from tabpfn import TabPFNRegressor

        n = min(len(x), self.cfg.max_train_rows)
        self.model = TabPFNRegressor(
            n_estimators=8,
            device=self.cfg.device,
            random_state=self.cfg.seed,
            inference_precision=torch.float32,
        )
        # most recent rows: input is sorted by (ts, pair) upstream
        self.model.fit(x[-n:], y[-n:])

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() must be called first"
        import torch

        chunk = self.cfg.predict_chunk
        parts = [self.model.predict(x[i : i + chunk]) for i in range(0, len(x), chunk)]
        out = np.concatenate(parts)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return out
