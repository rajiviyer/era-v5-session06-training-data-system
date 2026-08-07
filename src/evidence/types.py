"""Evidence bundle types and the requirement vocabulary (P11-T04, P11-T05).

The fourteen keys in `REQUIREMENT_KEYS` are SCOPE.md §9.2 verbatim. They are the
grading contract, so they live in one place and both `evidence.json` and `evidence.md`
are generated from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVIDENCE_JSON_FILENAME = "evidence.json"
EVIDENCE_MD_FILENAME = "evidence.md"

REQUIREMENT_KEYS: tuple[str, ...] = (
    "immutable_shards_with_manifests",
    "frozen_tokenizer_hashes",
    "packing_policies",
    "correct_masks",
    "curriculum_and_floors",
    "eval_firewall",
    "opus_audit_trail",
    "consumption_ledger",
    "learning_ledger",
    "checkpoint_ledger_binding",
    "crash_resume_no_skip_repeat",
    "replay_hash_match",
    "fork_new_branch",
    "packing_and_throughput",
)

REQUIREMENT_TITLES: dict[str, str] = {
    "immutable_shards_with_manifests": "Immutable tokenized shards with manifests",
    "frozen_tokenizer_hashes": "Frozen tokenizer and content hashes",
    "packing_policies": "Packing policies for different data types",
    "correct_masks": "Correct loss, attention, and position IDs",
    "curriculum_and_floors": "Curriculum stages, lane weights, protected floors",
    "eval_firewall": "Evaluation and validation firewall",
    "opus_audit_trail": "OPUS accept / reject / defer / override audit",
    "consumption_ledger": "Training consumption ledger",
    "learning_ledger": "Learning ledger linked to consumption",
    "checkpoint_ledger_binding": "Checkpoints tied to ledger offsets",
    "crash_resume_no_skip_repeat": "Crash recovery with no skipped or repeated batch",
    "replay_hash_match": "Replay of a historical range matches recorded hashes",
    "fork_new_branch": "Fork from an earlier checkpoint onto a new branch",
    "packing_and_throughput": "Packing utilization and useful tokens/sec",
}


class EvidenceError(RuntimeError):
    """Raised when the bundle cannot be built from the artifacts on disk."""


@dataclass(frozen=True)
class Check:
    """One assertion inside a requirement, with the number it was decided on.

    `detail` carries the measured value rather than a restatement of the assertion, so a
    reader can see *what* the artifacts said and not only that something passed.
    """

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class RequirementResult:
    """Verdict for one SCOPE.md §9.2 requirement."""

    key: str
    checks: tuple[Check, ...]
    evidence_paths: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def evidence_path(self) -> str:
        """The primary artifact, for the `evidence_path` field SCOPE.md §9.2 shows."""
        return self.evidence_paths[0] if self.evidence_paths else ""

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "evidence_path": self.evidence_path,
            "evidence_paths": list(self.evidence_paths),
            "title": REQUIREMENT_TITLES.get(self.key, self.key),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything `evidence.json` reports."""

    requirements: dict[str, RequirementResult]
    generated_at: str
    demo_command: str
    git_commit: str | None
    artifacts_dir: str
    missing_evidence_paths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return (
            not self.missing_evidence_paths
            and set(self.requirements) == set(REQUIREMENT_KEYS)
            and all(result.passed for result in self.requirements.values())
        )

    @property
    def failed_keys(self) -> tuple[str, ...]:
        return tuple(
            key for key in REQUIREMENT_KEYS if not self.requirements[key].passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": {
                key: self.requirements[key].to_dict() for key in REQUIREMENT_KEYS
            },
            "passed": self.passed,
            "failed_requirements": list(self.failed_keys),
            "missing_evidence_paths": list(self.missing_evidence_paths),
            "generated_at": self.generated_at,
            "demo_command": self.demo_command,
            "git_commit": self.git_commit,
            "artifacts_dir": self.artifacts_dir,
        }
