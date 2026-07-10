"""Command-line entry point: one subcommand per pipeline stage.

Stages communicate exclusively through parquet artifacts on disk, so any
stage can be rerun in isolation and RAM stays bounded.
"""

from __future__ import annotations

import argparse
import sys

from tml.config import load_config

STAGES = ["ingest", "quality", "features", "labels", "cv", "backtest", "report"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tml", description="ML trading research pipeline")
    parser.add_argument("--config", default="config/research.toml", help="path to research.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [*STAGES, "all"]:
        sub.add_parser(name, help=f"run the {name} stage" if name != "all" else "run every stage in order")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    commands = STAGES if args.command == "all" else [args.command]
    for command in commands:
        rc = _run_stage(command, cfg)
        if rc != 0:
            return rc
    return 0


def _run_stage(command: str, cfg) -> int:
    # Imports are lazy so `tml ingest` doesn't pay for matplotlib/nautilus imports.
    if command == "ingest":
        from tml.data.ingest import run_ingest

        return run_ingest(cfg)
    if command == "quality":
        from tml.data.quality import run_quality

        return run_quality(cfg)
    if command == "features":
        from tml.features.engine import run_features

        return run_features(cfg)
    if command == "labels":
        from tml.labeling.labels import run_labels

        return run_labels(cfg)
    if command == "cv":
        from tml.pipeline.research import run_cv

        return run_cv(cfg)
    if command == "backtest":
        from tml.backtest.runner import run_backtest

        return run_backtest(cfg)
    if command == "report":
        from tml.reporting.report import build_report

        return build_report(cfg)
    raise ValueError(f"unknown command {command}")


if __name__ == "__main__":
    sys.exit(main())
