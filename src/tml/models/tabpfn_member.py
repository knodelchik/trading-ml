"""TabPFN ensemble member (in-context tabular foundation model).

Runs in a SUBPROCESS (see tabpfn_worker.py): torch's OpenMP runtime
deadlocks against LightGBM/CatBoost's copy when loaded into the same
process on macOS. The adapter keeps the ModelAdapter protocol; fit() just
stashes training data and predict() round-trips arrays through a temp dir.

Model facts (verified against installed tabpfn 8.0.8):
  * 10k-sample in-context cap — the worker feeds the most RECENT rows,
    a recency-weighted view complementing the boosters' full history;
  * fp32 pinned (MPS autocast is force-disabled upstream anyway);
  * weights cached at ~/Library/Caches/tabpfn/ after one-time licensed
    download; NON-COMMERCIAL license — drop this member before any
    real-money use.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tml.config import TabPFNConfig

WORKER_TIMEOUT_S = 3600


class TabPFNAdapter:
    def __init__(self, cfg: TabPFNConfig):
        self.cfg = cfg
        self._x: Any = None
        self._y: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        # in-context learner: "fitting" happens inside predict's forward pass
        self._x = x
        self._y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self._x is not None, "fit() must be called first"
        with tempfile.TemporaryDirectory(prefix="tabpfn_") as td:
            in_path = Path(td) / "in.npz"
            out_path = Path(td) / "out.npy"
            np.savez_compressed(
                in_path,
                x_tr=self._x,
                y_tr=self._y,
                x_te=x,
                seed=self.cfg.seed,
                n_estimators=self.cfg.n_estimators,
                max_train_rows=self.cfg.max_train_rows,
                predict_chunk=self.cfg.predict_chunk,
                device=self.cfg.device,
            )
            proc = subprocess.run(
                [sys.executable, "-m", "tml.models.tabpfn_worker", str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_S,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"tabpfn worker failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
                )
            return np.load(out_path)
