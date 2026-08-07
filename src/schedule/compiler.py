"""Mixture timeline compiler (P3-T02)."""

from __future__ import annotations

from config.schemas import CurriculumConfig

from .errors import ScheduleError
from .stages import parse_stage_records
from .types import CompiledSchedule, LANE_KEYS, StageRecord, StepSchedule

SCHEMA_VERSION = "1.0"
_WEIGHT_TOLERANCE = 1e-6


def compile_schedule(
    curriculum: CurriculumConfig,
    *,
    total_steps: int,
    lane_supply: dict[str, int] | None = None,
) -> CompiledSchedule:
    """Compile curriculum stages into per-step lane quotas."""
    if total_steps <= 0:
        raise ScheduleError("total_steps must be positive")

    stages = parse_stage_records(curriculum)
    _validate_stage_coverage(stages, total_steps=total_steps)

    steps: list[StepSchedule] = []
    for step in range(total_steps):
        steps.append(_compile_step(step, curriculum, stages))

    warnings = _collect_supply_warnings(steps, lane_supply)

    return CompiledSchedule(
        schema_version=SCHEMA_VERSION,
        total_steps=total_steps,
        always_on_fraction=curriculum.batch.always_on_fraction,
        opus_fraction=curriculum.batch.opus_fraction,
        warmup_steps=curriculum.transitions.warmup_steps,
        transition_blend=curriculum.transitions.blend,
        protected_floor_lanes=curriculum.protected_floors.lanes,
        phase_boundaries=stages,
        steps=tuple(steps),
        warnings=tuple(warnings),
    )


def _validate_stage_coverage(stages: tuple[StageRecord, ...], *, total_steps: int) -> None:
    if not stages:
        raise ScheduleError("curriculum must define at least one stage")
    if stages[0].step_start != 0:
        raise ScheduleError("first stage must start at step 0")
    if stages[-1].step_end != total_steps:
        raise ScheduleError(
            f"final stage must end at total_steps ({total_steps}, got {stages[-1].step_end})"
        )
    for previous, current in zip(stages, stages[1:]):
        if current.step_start != previous.step_end:
            raise ScheduleError(
                "stages must be contiguous: "
                f"'{previous.name}' ends at {previous.step_end}, "
                f"'{current.name}' starts at {current.step_start}"
            )


def _compile_step(
    step: int,
    curriculum: CurriculumConfig,
    stages: tuple[StageRecord, ...],
) -> StepSchedule:
    current = _stage_for_step(stages, step)
    previous = _previous_stage(stages, current)
    offset = step - current.step_start
    in_transition = (
        previous is not None
        and 0 <= offset < curriculum.transitions.warmup_steps
    )

    if in_transition and previous is not None:
        outgoing_weight, incoming_weight = curriculum.transitions.blend
        opus_mixture = _blend_mixtures(
            previous.opus_mixture,
            current.opus_mixture,
            outgoing_weight=outgoing_weight,
            incoming_weight=incoming_weight,
        )
        transition_from = previous.name
        transition_to = current.name
    else:
        opus_mixture = dict(current.opus_mixture)
        transition_from = None
        transition_to = None

    opus_lane_quotas = {
        lane: weight * curriculum.batch.opus_fraction
        for lane, weight in opus_mixture.items()
    }

    return StepSchedule(
        step=step,
        phase=current.name,
        in_transition=in_transition,
        transition_from=transition_from,
        transition_to=transition_to,
        always_on_fraction=curriculum.batch.always_on_fraction,
        opus_fraction=curriculum.batch.opus_fraction,
        always_on_sub_shares=dict(curriculum.always_on.sub_shares),
        opus_lane_quotas=opus_lane_quotas,
        lr_multiplier=current.lr_multiplier,
        anneal_eligible_only=current.anneal_eligible_only,
    )


def _stage_for_step(stages: tuple[StageRecord, ...], step: int) -> StageRecord:
    for stage in stages:
        if stage.step_start <= step < stage.step_end:
            return stage
    raise ScheduleError(f"no stage covers training step {step}")


def _previous_stage(
    stages: tuple[StageRecord, ...],
    current: StageRecord,
) -> StageRecord | None:
    for index, stage in enumerate(stages):
        if stage.name == current.name:
            if index == 0:
                return None
            return stages[index - 1]
    return None


def _blend_mixtures(
    outgoing: dict[str, float],
    incoming: dict[str, float],
    *,
    outgoing_weight: float,
    incoming_weight: float,
) -> dict[str, float]:
    lanes = LANE_KEYS
    blended = {
        lane: outgoing.get(lane, 0.0) * outgoing_weight + incoming.get(lane, 0.0) * incoming_weight
        for lane in lanes
    }
    total = sum(blended.values())
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        raise ScheduleError(f"blended mixture must sum to 1.0 (got {total:.6f})")
    return blended


def _collect_supply_warnings(
    steps: list[StepSchedule],
    lane_supply: dict[str, int] | None,
) -> list[str]:
    if lane_supply is None:
        return []

    warnings: list[str] = []
    peak_demand: dict[str, float] = {lane: 0.0 for lane in LANE_KEYS}
    for entry in steps:
        for lane, quota in entry.opus_lane_quotas.items():
            if quota > peak_demand.get(lane, 0.0):
                peak_demand[lane] = quota

    for lane, demand in sorted(peak_demand.items()):
        if demand <= _WEIGHT_TOLERANCE:
            continue
        available = lane_supply.get(lane, 0)
        if available <= 0:
            warnings.append(
                f"supply shortfall: lane '{lane}' quota {demand:.4f} but zero admitted shards"
            )
    return warnings
