"""Compiled mixture schedule types (P3-T01–T03)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LANE_KEYS: frozenset[str] = frozenset(
    {"web", "code", "indic", "stem", "reasoning", "long_context", "agentic"}
)


@dataclass(frozen=True)
class StageRecord:
    """One curriculum stage used by the mixture timeline compiler."""

    name: str
    step_start: int
    step_end: int
    opus_mixture: dict[str, float]
    lr_multiplier: float | None = None
    anneal_eligible_only: bool = False
    tier_d_indic_fraction: float | None = None


@dataclass(frozen=True)
class StepSchedule:
    """Per-step lane quotas and stage metadata."""

    step: int
    phase: str
    in_transition: bool
    transition_from: str | None
    transition_to: str | None
    always_on_fraction: float
    opus_fraction: float
    always_on_sub_shares: dict[str, float]
    opus_lane_quotas: dict[str, float]
    lr_multiplier: float | None
    anneal_eligible_only: bool


@dataclass(frozen=True)
class PlannedSample:
    """One planned training sample for a global step."""

    sample_id: str
    shard_id: str
    capability_lane: str
    path: str
    sub_share: str | None = None


@dataclass(frozen=True)
class StepPlan:
    """Deterministic sample plan for one global training step."""

    step: int
    phase: str
    run_id: str
    branch_id: str
    samples: tuple[PlannedSample, ...]

    @property
    def always_on_count(self) -> int:
        return sum(1 for sample in self.samples if sample.path == "always_on")


@dataclass(frozen=True)
class RunPlan:
    """Deterministic sample plan for an entire run."""

    run_id: str
    branch_id: str
    seed: int
    global_batch_size: int
    steps: tuple[StepPlan, ...]


@dataclass(frozen=True)
class CompiledSchedule:
    """Full compiled mixture timeline for a demo run."""

    schema_version: str
    total_steps: int
    always_on_fraction: float
    opus_fraction: float
    warmup_steps: int
    transition_blend: tuple[float, float]
    protected_floor_lanes: tuple[str, ...]
    phase_boundaries: tuple[StageRecord, ...]
    steps: tuple[StepSchedule, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "total_steps": self.total_steps,
            "always_on_fraction": self.always_on_fraction,
            "opus_fraction": self.opus_fraction,
            "warmup_steps": self.warmup_steps,
            "transition_blend": list(self.transition_blend),
            "protected_floor_lanes": list(self.protected_floor_lanes),
            "phase_boundaries": [
                {
                    "name": phase.name,
                    "step_start": phase.step_start,
                    "step_end": phase.step_end,
                    "opus_mixture": dict(phase.opus_mixture),
                    "lr_multiplier": phase.lr_multiplier,
                    "anneal_eligible_only": phase.anneal_eligible_only,
                    "tier_d_indic_fraction": phase.tier_d_indic_fraction,
                }
                for phase in self.phase_boundaries
            ],
            "warnings": list(self.warnings),
            "steps": [
                {
                    "step": entry.step,
                    "phase": entry.phase,
                    "in_transition": entry.in_transition,
                    "transition_from": entry.transition_from,
                    "transition_to": entry.transition_to,
                    "always_on_fraction": entry.always_on_fraction,
                    "opus_fraction": entry.opus_fraction,
                    "always_on_sub_shares": dict(entry.always_on_sub_shares),
                    "opus_lane_quotas": dict(entry.opus_lane_quotas),
                    "lr_multiplier": entry.lr_multiplier,
                    "anneal_eligible_only": entry.anneal_eligible_only,
                }
                for entry in self.steps
            ],
        }
