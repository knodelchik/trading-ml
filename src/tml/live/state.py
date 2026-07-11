"""Local journal for live paper trading.

Bybit demo retains orders/records for only ~7 days server-side, so the
LOCAL journal is the source of truth for the paper track record. Everything
is append-only JSONL under artifacts/live/journal/.

Kill switch: creating the file artifacts/live/STOP makes the next tick
close nothing, open nothing, and exit — a hand brake that needs no code.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tml.config import Config


def journal_dir(cfg: Config) -> Path:
    return cfg.artifacts / "live" / "journal"


def kill_switch_path(cfg: Config) -> Path:
    return cfg.artifacts / "live" / "STOP"


def kill_switch_active(cfg: Config) -> bool:
    return kill_switch_path(cfg).exists()


def _append(cfg: Config, name: str, record: dict) -> None:
    d = journal_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    record = {"recorded_at": datetime.now(UTC).isoformat(), **record}
    with open(d / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def log_decision(cfg: Config, record: dict) -> None:
    _append(cfg, "decisions", record)


def log_order(cfg: Config, record: dict) -> None:
    _append(cfg, "orders", record)


def log_equity(cfg: Config, record: dict) -> None:
    _append(cfg, "equity", record)


def log_error(cfg: Config, record: dict) -> None:
    _append(cfg, "errors", record)


def read_journal(cfg: Config, name: str) -> list[dict]:
    p = journal_dir(cfg) / f"{name}.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]
