#!/usr/bin/env python3
"""CI-style invariant runner over submission_artifacts/ (PX-T03)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preflight.verify import verify_artifacts  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Session 6 submission artifacts against SCOPE invariants.",
    )
    parser.add_argument(
        "artifacts_dir",
        type=Path,
        nargs="?",
        default=_ASSIGNMENT / "submission_artifacts",
        help="Path to submission_artifacts/ (default: assignment submission_artifacts/)",
    )
    parser.add_argument(
        "--assignment-root",
        type=Path,
        default=_ASSIGNMENT,
        help="Assignment root for resolving committed tokenizer/corpus paths",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    if not artifacts_dir.is_dir():
        print(f"error: artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        return 2

    result = verify_artifacts(artifacts_dir, args.assignment_root.resolve())
    if result.passed:
        print("ok: all 14 requirement checks passed")
        return 0

    print("artifact verification failed:", file=sys.stderr)
    if result.failed_requirements:
        print(f"  failed requirements: {', '.join(result.failed_requirements)}", file=sys.stderr)
    if result.missing_evidence_paths:
        print(f"  missing paths: {', '.join(result.missing_evidence_paths)}", file=sys.stderr)
    for detail in result.details[:20]:
        print(f"  - {detail}", file=sys.stderr)
    if len(result.details) > 20:
        print(f"  ... and {len(result.details) - 20} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
