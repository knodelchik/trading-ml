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


def journal_executions(cfg: Config, execs: list[dict]) -> int:
    """Append exchange executions not already recorded (dedupe by exec_id).
    Returns the number of new fills captured."""
    d = journal_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "fills.jsonl"
    seen = {json.loads(line)["exec_id"] for line in path.open()} if path.exists() else set()
    new = [e for e in execs if e["exec_id"] not in seen]
    if new:
        with open(path, "a") as f:
            for e in sorted(new, key=lambda r: r["ts_ms"]):
                f.write(json.dumps(e, sort_keys=True) + "\n")
    return len(new)


def journal_closed_pnl(cfg: Config, trades: list[dict]) -> int:
    """Append closed round trips not already recorded (dedupe by order_id).
    Bybit demo only serves ~7 days of closed-PnL, so this local accumulation
    is the durable source for the bot's LIFETIME realized P&L (immune to the
    server-side window as long as we tick at least weekly). Returns new count.
    """
    d = journal_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "closed_pnl.jsonl"
    seen = {json.loads(line)["order_id"] for line in path.open()} if path.exists() else set()
    new = [t for t in trades if t["order_id"] not in seen]
    if new:
        with open(path, "a") as f:
            for t in sorted(new, key=lambda r: r["close_ms"]):
                f.write(json.dumps(t, sort_keys=True) + "\n")
    return len(new)


def realized_total(cfg: Config) -> float:
    """Lifetime realized P&L (USDT) from the durable closed-PnL journal."""
    return sum(t["closed_pnl"] for t in read_journal(cfg, "closed_pnl"))


def equity_baseline(cfg: Config) -> float | None:
    """The first logged account equity — the honest baseline for the bot's
    all-in result (price P&L + funding + fees), since the account is all USDT.
    None if nothing logged yet."""
    path = journal_dir(cfg) / "equity.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            if line.strip():
                return float(json.loads(line)["total_equity"])
    return None


def last_fill_ms(cfg: Config) -> int | None:
    path = journal_dir(cfg) / "fills.jsonl"
    if not path.exists():
        return None
    last = None
    with path.open() as f:
        for line in f:
            if line.strip():
                last = json.loads(line)["ts_ms"]
    return last


def log_error(cfg: Config, record: dict) -> None:
    _append(cfg, "errors", record)


def read_journal(cfg: Config, name: str) -> list[dict]:
    p = journal_dir(cfg) / f"{name}.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]
