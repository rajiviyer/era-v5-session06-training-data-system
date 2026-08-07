"""Build `evidence.json` from generated artifacts (P11-T04, P11-T05).

The collector loads `submission_artifacts/`, runs every check in `checks.py`, and writes
the bundle. It contains no verdicts of its own: `passed` on each requirement is the
conjunction of checks that read files, and a requirement whose evidence paths do not
exist on disk fails the bundle rather than being reported anyway.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import json

from shards.io import write_bytes_atomic

from .checks import CHECKS, load_artifacts
from .types import (
    EVIDENCE_JSON_FILENAME,
    REQUIREMENT_KEYS,
    EvidenceBundle,
    EvidenceError,
    RequirementResult,
)

DEFAULT_DEMO_COMMAND = "python session06/assignment/scripts/run_demo.py"


def collect_evidence(
    artifacts_dir: Path,
    assignment_root: Path,
    *,
    demo_command: str = DEFAULT_DEMO_COMMAND,
) -> EvidenceBundle:
    """Run every requirement check against the generated artifacts."""
    artifacts = load_artifacts(artifacts_dir, assignment_root)

    results: dict[str, RequirementResult] = {}
    for check in CHECKS:
        result = check(artifacts)
        if result.key in results:
            raise EvidenceError(f"duplicate requirement key produced: {result.key}")
        results[result.key] = result

    unexpected = sorted(set(results) - set(REQUIREMENT_KEYS))
    missing = sorted(set(REQUIREMENT_KEYS) - set(results))
    if unexpected or missing:
        raise EvidenceError(
            f"checks do not cover SCOPE.md §9.2 exactly: missing {missing}, "
            f"unexpected {unexpected}"
        )

    # An evidence_path a grader cannot open is not evidence. The primary path must be a
    # readable file; the supporting paths may also be directories (a manifests/ tree is a
    # legitimate pointer, but it is not what `evidence_path` should hand a reader).
    missing_paths = tuple(
        sorted(
            {
                path
                for result in results.values()
                for path in result.evidence_paths
                if not (artifacts.root / path).exists()
            }
            | {
                result.evidence_path
                for result in results.values()
                if not (artifacts.root / result.evidence_path).is_file()
            }
        )
    )

    return EvidenceBundle(
        requirements=results,
        generated_at=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        demo_command=demo_command,
        git_commit=_git_commit(assignment_root),
        artifacts_dir=artifacts.root.name,
        missing_evidence_paths=missing_paths,
    )


def write_evidence_json(artifacts_dir: Path, bundle: EvidenceBundle) -> Path:
    """Write `evidence.json` next to the artifacts it describes.

    Keys are left unsorted on purpose: the requirements appear in SCOPE.md §9.2 order,
    so a grader reading the file top to bottom reads it in the order of the contract.
    """
    target = Path(artifacts_dir).resolve() / EVIDENCE_JSON_FILENAME
    text = json.dumps(bundle.to_dict(), indent=2) + "\n"
    write_bytes_atomic(target, text.encode("utf-8"))
    return target


def _git_commit(assignment_root: Path) -> str | None:
    """The commit the artifacts were produced at, or None outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(assignment_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None
