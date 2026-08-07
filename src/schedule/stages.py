"""Parse curriculum config into stage records (P3-T01)."""

from __future__ import annotations

from config.schemas import CurriculumConfig, CurriculumPhase

from .types import StageRecord


def parse_stage_records(curriculum: CurriculumConfig) -> tuple[StageRecord, ...]:
    """Convert validated curriculum phases into compiler stage records."""
    return tuple(_stage_from_phase(phase) for phase in curriculum.phases)


def _stage_from_phase(phase: CurriculumPhase) -> StageRecord:
    return StageRecord(
        name=phase.name,
        step_start=phase.step_start,
        step_end=phase.step_end,
        opus_mixture=dict(phase.opus_mixture),
        lr_multiplier=phase.lr_multiplier,
        anneal_eligible_only=phase.anneal_eligible_only,
        tier_d_indic_fraction=phase.tier_d_indic_fraction,
    )
