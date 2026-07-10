"""Append-only ledger of every configuration ever evaluated.

The Deflated Sharpe Ratio is only meaningful if the trial count is honest:
every CV run appends a record here, and the report REFUSES to compute DSR
when the ledger is missing. Deleting this file to "reset" the count is
self-deception — the selection bias it measures does not reset.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class TrialsLedger:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"recorded_at": datetime.now(UTC).isoformat(), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def sharpe_history(self) -> list[float]:
        """Per-period (non-annualized) SRs of all recorded trials."""
        return [
            r["sr_per_period"]
            for r in self.read()
            if r.get("sr_per_period") is not None
        ]
