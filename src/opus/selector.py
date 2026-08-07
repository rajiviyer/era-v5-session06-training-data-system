"""OPUS accept / reject / defer logic with protected-floor override (P5-T02–T03)."""

from __future__ import annotations

from .scorer import DeterministicOpusScorer, OpusScorer
from .types import (
    OpusAuditRecord,
    OpusCandidateContext,
    OpusDecision,
    OpusResult,
    OpusSelectorConfig,
    make_opus_decision_id,
)


def evaluate_opus(
    context: OpusCandidateContext,
    config: OpusSelectorConfig,
    *,
    scorer: OpusScorer | None = None,
) -> OpusResult:
    """Score one candidate and return an auditable OPUS decision."""
    active_scorer = scorer or DeterministicOpusScorer()
    decision_id = make_opus_decision_id(
        run_id=context.run_id,
        branch_id=context.branch_id,
        global_step=context.global_step,
        candidate_id=context.candidate_id,
    )

    if context.path == "always_on":
        audit = OpusAuditRecord(
            opus_decision_id=decision_id,
            candidate_id=context.candidate_id,
            global_step=context.global_step,
            run_id=context.run_id,
            branch_id=context.branch_id,
            sample_ids=context.sample_ids,
            shard_ids=context.shard_ids,
            capability_lane=context.capability_lane,
            curriculum_stage=context.curriculum_stage,
            path=context.path,
            opus_score=None,
            decision="accepted",
            rejection_reason=None,
            protected_floor_override=False,
            opus_bypassed=True,
            effective_token_estimate=context.effective_token_estimate,
        )
        return OpusResult(
            candidate_id=context.candidate_id,
            global_step=context.global_step,
            decision="accepted",
            opus_score=None,
            protected_floor_override=False,
            opus_bypassed=True,
            committed=True,
            audit=audit,
        )

    score = active_scorer.score(context)
    decision, rejection_reason, protected_override = _decide(context, score, config)
    committed = decision in {"accepted", "protected_override"}

    audit = OpusAuditRecord(
        opus_decision_id=decision_id,
        candidate_id=context.candidate_id,
        global_step=context.global_step,
        run_id=context.run_id,
        branch_id=context.branch_id,
        sample_ids=context.sample_ids,
        shard_ids=context.shard_ids,
        capability_lane=context.capability_lane,
        curriculum_stage=context.curriculum_stage,
        path=context.path,
        opus_score=score,
        decision=decision,
        rejection_reason=rejection_reason,
        protected_floor_override=protected_override,
        opus_bypassed=False,
        effective_token_estimate=context.effective_token_estimate,
    )
    return OpusResult(
        candidate_id=context.candidate_id,
        global_step=context.global_step,
        decision=decision,
        opus_score=score,
        protected_floor_override=protected_override,
        opus_bypassed=False,
        committed=committed,
        audit=audit,
    )


def _decide(
    context: OpusCandidateContext,
    score: float,
    config: OpusSelectorConfig,
) -> tuple[OpusDecision, str | None, bool]:
    if score >= config.accept_threshold:
        return "accepted", None, False

    if context.capability_lane in config.protected_floor_lanes:
        return (
            "protected_override",
            f"score {score:.4f} below threshold {config.accept_threshold:.4f}; "
            f"protected floor applied for lane {context.capability_lane}",
            True,
        )

    if score >= config.defer_threshold:
        return (
            "deferred",
            f"score {score:.4f} below accept threshold {config.accept_threshold:.4f}; "
            "candidate deferred for later stage review",
            False,
        )

    return (
        "rejected",
        f"score {score:.4f} below defer threshold {config.defer_threshold:.4f}",
        False,
    )
