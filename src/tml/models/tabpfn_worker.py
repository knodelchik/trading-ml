"""Subprocess worker for the TabPFN member.

WHY A SEPARATE PROCESS: LightGBM/CatBoost and PyTorch each bundle their own
OpenMP runtime; loading both libomp copies into one process deadlocks on
macOS (observed: torch fill_kernel stuck forever in __kmp_join_barrier).
Isolating torch in its own process gives each library a private OpenMP.
IPC cost is a few MB of float32 arrays per fold — noise next to inference.

Usage: python -m tml.models.tabpfn_worker <in.npz> <out.npy>
  in.npz: x_tr, y_tr, x_te float32 arrays + scalar config fields
"""

from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    x_tr, y_tr, x_te = data["x_tr"], data["y_tr"], data["x_te"]
    seed = int(data["seed"])
    n_estimators = int(data["n_estimators"])
    max_train_rows = int(data["max_train_rows"])
    chunk = int(data["predict_chunk"])
    device = str(data["device"])

    import torch
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        device=device,
        random_state=seed,
        inference_precision=torch.float32,
    )
    n = min(len(x_tr), max_train_rows)
    model.fit(x_tr[-n:], y_tr[-n:])  # most recent rows: sorted by (ts, pair) upstream
    parts = [model.predict(x_te[i : i + chunk]) for i in range(0, len(x_te), chunk)]
    np.save(out_path, np.concatenate(parts).astype(np.float64))
    return 0


if __name__ == "__main__":
    sys.exit(main())
