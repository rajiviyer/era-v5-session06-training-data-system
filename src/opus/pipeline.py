"""Wire firewall and OPUS into the batch gate (P5-T06)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.schemas import OpusConfig
from firewall.gate import evaluate_firewall
from firewall.types import BatchCandidate, EvalRegistry

from .audit import append_opus_audit
from .selector import evaluate_opus
from .types import BatchPipelineResult, CandidatePath, OpusCandidateContext, OpusSelectorConfig


def build_opus_context(
    candidate: BatchCandidate,
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    curriculum_stage: str,
    capability_lane: str,
    path: CandidatePath,
    documents_by_id: dict[str, dict[str, Any]] | None = None,
) -> OpusCandidateContext:
    """Build deterministic OPUS context from a firewall candidate."""
    curriculum_band: str | None = None
    token_estimate = 0
    if documents_by_id is not None:
        for sample_id in candidate.sample_ids:
            document = documents_by_id.get(sample_id)
            if document is None:
                continue
            if curriculum_band is None:
                curriculum_band = str(document.get("curriculum_band", "")) or None
            token_estimate += int(document.get("char_count", 0))

    return OpusCandidateContext(
        run_id=run_id,
        branch_id=branch_id,
        seed=seed,
        global_step=candidate.global_step,
        candidate_id=candidate.candidate_id,
        sample_ids=candidate.sample_ids,
        shard_ids=candidate.shard_ids,
        content_hashes=candidate.content_hashes,
        capability_lane=capability_lane,
        curriculum_stage=curriculum_stage,
        path=path,
        curriculum_band=curriculum_band,
        effective_token_estimate=token_estimate,
    )


def selector_config_from_demo(
    opus: OpusConfig,
    protected_floor_lanes: tuple[str, ...],
) -> OpusSelectorConfig:
    """Build OPUS selector config from demo and curriculum settings."""
    return OpusSelectorConfig(
        accept_threshold=opus.accept_threshold,
        protected_floor_lanes=protected_floor_lanes,
    )


def run_batch_gate(
    candidate: BatchCandidate,
    *,
    registry: EvalRegistry,
    run_id: str,
    branch_id: str,
    seed: int,
    curriculum_stage: str,
    capability_lane: str,
    path: CandidatePath,
    opus_config: OpusConfig,
    protected_floor_lanes: tuple[str, ...],
    audit_path: Path | None = None,
    documents_by_id: dict[str, dict[str, Any]] | None = None,
) -> BatchPipelineResult:
    """Evaluate firewall then OPUS; append audit for every OPUS decision."""
    firewall_result = evaluate_firewall(
        candidate,
        registry,
        documents_by_id=documents_by_id,
    )
    if firewall_result.decision == "blocked":
        return BatchPipelineResult(
            candidate_id=candidate.candidate_id,
            global_step=candidate.global_step,
            firewall=firewall_result,
            opus=None,
            committed=False,
            candidate=candidate,
        )

    opus_context = build_opus_context(
        candidate,
        run_id=run_id,
        branch_id=branch_id,
        seed=seed,
        curriculum_stage=curriculum_stage,
        capability_lane=capability_lane,
        path=path,
        documents_by_id=documents_by_id,
    )
    selector_config = selector_config_from_demo(opus_config, protected_floor_lanes)
    opus_result = evaluate_opus(opus_context, selector_config)

    if audit_path is not None:
        append_opus_audit(audit_path, opus_result.audit)

    return BatchPipelineResult(
        candidate_id=candidate.candidate_id,
        global_step=candidate.global_step,
        firewall=firewall_result,
        opus=opus_result,
        committed=opus_result.committed,
        candidate=candidate,
    )
