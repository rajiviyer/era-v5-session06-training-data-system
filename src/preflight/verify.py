"""CI-style artifact verification over submission_artifacts/ (PX-T03)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evidence.checks import CHECKS, load_artifacts


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of running every evidence check against an artifacts tree."""

    passed: bool
    failed_requirements: tuple[str, ...]
    missing_evidence_paths: tuple[str, ...]
    details: tuple[str, ...]


def verify_artifacts(artifacts_dir: Path, assignment_root: Path) -> VerificationResult:
    """Run SCOPE.md §9.2 requirement checks; return a structured verdict."""
    artifacts = load_artifacts(artifacts_dir, assignment_root)

    failed: list[str] = []
    missing_paths: set[str] = set()
    details: list[str] = []

    for check in CHECKS:
        result = check(artifacts)
        if not result.passed:
            failed.append(result.key)
        for item in result.checks:
            if not item.passed:
                details.append(f"{result.key}: {item.name}: {item.detail}")
        for path in result.evidence_paths:
            if not (artifacts.root / path).exists():
                missing_paths.add(path)
        if not (artifacts.root / result.evidence_path).is_file():
            missing_paths.add(result.evidence_path)

    return VerificationResult(
        passed=not failed and not missing_paths,
        failed_requirements=tuple(sorted(failed)),
        missing_evidence_paths=tuple(sorted(missing_paths)),
        details=tuple(details),
    )
