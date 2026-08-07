"""OPUS selector and audit types (P5-T01–T05)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from firewall.types import BatchCandidate, FirewallResult

OpusDecision = Literal["accepted", "rejected", "deferred", "protected_override"]
CandidatePath = Literal["always_on", "opus"]

AUDIT_FILENAME = "opus_audit.jsonl"
DEFER_BAND = 0.15


@dataclass(frozen=True)
class OpusCandidateContext:
    """Deterministic inputs for OPUS scoring."""

    run_id: str
    branch_id: str
    seed: int
    global_step: int
    candidate_id: str
    sample_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    capability_lane: str
    curriculum_stage: str
    path: CandidatePath
    curriculum_band: str | None = None
    effective_token_estimate: int = 0


@dataclass(frozen=True)
class OpusSelectorConfig:
    """Thresholds for accept / defer / reject decisions."""

    accept_threshold: float
    protected_floor_lanes: tuple[str, ...]
    defer_band: float = DEFER_BAND

    @property
    def defer_threshold(self) -> float:
        return max(0.0, self.accept_threshold - self.defer_band)


@dataclass(frozen=True)
class OpusAuditRecord:
    """One append-only OPUS audit row."""

    opus_decision_id: str
    candidate_id: str
    global_step: int
    run_id: str
    branch_id: str
    sample_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    capability_lane: str
    curriculum_stage: str
    path: CandidatePath
    opus_score: float | None
    decision: OpusDecision | Literal["accepted"]
    rejection_reason: str | None
    protected_floor_override: bool
    opus_bypassed: bool
    effective_token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "opus_decision_id": self.opus_decision_id,
            "candidate_id": self.candidate_id,
            "global_step": self.global_step,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "sample_ids": list(self.sample_ids),
            "shard_ids": list(self.shard_ids),
            "capability_lane": self.capability_lane,
            "curriculum_stage": self.curriculum_stage,
            "path": self.path,
            "decision": self.decision,
            "rejection_reason": self.rejection_reason,
            "protected_floor_override": self.protected_floor_override,
            "opus_bypassed": self.opus_bypassed,
            "effective_token_estimate": self.effective_token_estimate,
        }
        if self.opus_score is not None:
            payload["opus_score"] = round(self.opus_score, 6)
        return payload


@dataclass(frozen=True)
class OpusResult:
    """Outcome of OPUS evaluation for one candidate."""

    candidate_id: str
    global_step: int
    decision: OpusDecision | Literal["accepted"]
    opus_score: float | None
    protected_floor_override: bool
    opus_bypassed: bool
    committed: bool
    audit: OpusAuditRecord


@dataclass(frozen=True)
class BatchPipelineResult:
    """Firewall + OPUS gate outcome before ledger commit (P5-T06)."""

    candidate_id: str
    global_step: int
    firewall: FirewallResult
    opus: OpusResult | None
    committed: bool
    candidate: BatchCandidate


def make_opus_decision_id(
    *,
    run_id: str,
    branch_id: str,
    global_step: int,
    candidate_id: str,
) -> str:
    """Stable OPUS decision identifier."""
    payload = f"{run_id}|{branch_id}|{global_step}|{candidate_id}|opus"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"opus-{digest}"
