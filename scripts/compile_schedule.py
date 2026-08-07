#!/usr/bin/env python3
"""Compile curriculum.yaml into schedule.json (P3-T03)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config.loader import load_configs  # noqa: E402
from schedule import compile_schedule, write_schedule_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Session 6 mixture schedule.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ASSIGNMENT / "configs" / "demo.yaml",
        help="Demo config YAML (default: configs/demo.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output schedule.json path (default: submission_artifacts/schedule.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    demo, curriculum = load_configs(_ASSIGNMENT, demo_path=args.config.resolve())
    schedule = compile_schedule(
        curriculum,
        total_steps=demo.training.total_steps,
    )
    output_path = args.output or (demo.paths.submission_artifacts / "schedule.json")
    write_schedule_json(output_path.resolve(), schedule)
    print(f"wrote schedule for {schedule.total_steps} steps -> {output_path.resolve()}")
    if schedule.warnings:
        print(f"warnings: {len(schedule.warnings)}")
        for warning in schedule.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
