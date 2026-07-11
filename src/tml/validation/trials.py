"""Append-only ledger of every configuration ever evaluated.

The Deflated Sharpe Ratio is only meaningful if the trial count is honest:
every CV run appends a record here, and the report REFUSES to compute DSR
when the ledger is missing. Deleting this file to "reset" the count is
self-deception — the selection bias it measures does not reset.

Two subtleties encoded here (found by adversarial review):
  * Idempotent RE-RUNS of the same configuration are not new trials —
    `sharpe_history` dedupes by configuration identity while the file
    itself stays append-only for audit.
  * All SR values entering one deflation must come from THE SAME estimator
    on THE SAME cost basis; records carry an `estimator` tag and histories
    are filtered by it and by stage (pre-cost cv vs net backtest), so the
    expected-max benchmark measures selection breadth, not estimator gaps.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

# current SR estimator convention for ledger records: simple SR of the
# stride-H non-overlapping per-period return series
ESTIMATOR = "stride_h_nonoverlap_v2"


class TrialsLedger:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"recorded_at": datetime.now(UTC).isoformat(), "estimator": ESTIMATOR, **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def sharpe_history(self, stage: str) -> list[float]:
        """Distinct-configuration SRs for one stage, current estimator only."""
        seen: dict[tuple, float] = {}
        for r in self.read():
            if r.get("sr_per_period") is None:
                continue
            if r.get("stage") != stage or r.get("estimator") != ESTIMATOR:
                continue
            key = (
                r.get("config_hash"),
                r.get("enter_threshold"),
                r.get("exit_threshold"),
                r.get("execution_delay_bars"),
            )
            seen[key] = r["sr_per_period"]
        return list(seen.values())

    def counts(self, stage: str) -> tuple[int, int]:
        """(raw records, distinct current-estimator trials) for disclosure."""
        raw = [r for r in self.read() if r.get("stage") == stage]
        return len(raw), len(self.sharpe_history(stage))
