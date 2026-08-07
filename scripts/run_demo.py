#!/usr/bin/env python3
"""Run the full Session 6 demonstration end to end (P11-T02).

    uv run python scripts/run_demo.py

Regenerates `submission_artifacts/` from a clean directory and exits non-zero if any
phase fails, so it is usable as a CI gate as well as a demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from demo import PENDING_PHASES, DemoResult, run_demo  # noqa: E402

_RULE = "=" * 78


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Session 6 demo end to end.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Artifacts root (default: paths.submission_artifacts from configs/demo.yaml)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not empty the artifacts directory first (default: clean regeneration)",
    )
    return parser.parse_args()


def _print_report(result: DemoResult) -> None:
    print()
    print(_RULE)
    print("SESSION 6 DEMO")
    print(_RULE)
    for phase in result.phases:
        status = "PASS" if phase.passed else "FAIL"
        print(f"[{status}] Phase {phase.number}: {phase.name}")
        print(f"        {phase.detail}")
    print(_RULE)
    print(f"artifacts: {result.artifacts_dir}")
    print(f"evidence:  {result.artifacts_dir / 'evidence.md'}")

    if PENDING_PHASES:
        print()
        print("Not implemented yet (this run is not a complete submission):")
        for pending in PENDING_PHASES:
            print(f"  - {pending}")

    print()
    passed = sum(1 for phase in result.phases if phase.passed)
    verdict = "DEMO PASSED" if result.passed else "DEMO FAILED"
    print(f"{verdict}: {passed}/{len(result.phases)} phases passed")


def main() -> int:
    args = _parse_args()
    result = run_demo(
        _ASSIGNMENT,
        artifacts_dir=args.output.resolve() if args.output else None,
        clean=not args.keep_existing,
    )
    _print_report(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
