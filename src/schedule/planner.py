"""Deterministic sample planner (P3-T06)."""

from __future__ import annotations

from .always_on import is_always_on_slot, sample_always_on_candidate
from .deterministic import deterministic_choice, deterministic_unit_float
from .errors import ScheduleError
from .filters import filter_always_on_candidates, filter_opus_candidates
from .pool import SampleCandidate
from .types import PlannedSample, RunPlan, StepPlan, StepSchedule


def plan_step(
    step: StepSchedule,
    candidates: tuple[SampleCandidate, ...],
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    global_batch_size: int,
) -> StepPlan:
    """Plan one global step from compiled quotas and admitted sample pool."""
    if global_batch_size <= 0:
        raise ScheduleError("global_batch_size must be positive")

    always_on_pool = filter_always_on_candidates(candidates)
    opus_pool = filter_opus_candidates(candidates, step)

    planned: list[PlannedSample] = []
    planned_ids: set[str] = set()
    for slot_index in range(global_batch_size):
        if is_always_on_slot(
            run_id=run_id,
            branch_id=branch_id,
            seed=seed,
            step=step.step,
            slot_index=slot_index,
            always_on_fraction=step.always_on_fraction,
        ):
            sample = _plan_always_on_slot(
                _exclude_planned(always_on_pool, planned_ids),
                step,
                run_id=run_id,
                branch_id=branch_id,
                seed=seed,
                slot_index=slot_index,
            )
        else:
            sample = _plan_opus_slot(
                _exclude_planned(opus_pool, planned_ids),
                step,
                run_id=run_id,
                branch_id=branch_id,
                seed=seed,
                slot_index=slot_index,
            )
        planned.append(sample)
        planned_ids.add(sample.sample_id)

    return StepPlan(
        step=step.step,
        phase=step.phase,
        run_id=run_id,
        branch_id=branch_id,
        samples=tuple(planned),
    )


def plan_run(
    schedule_steps: tuple[StepSchedule, ...],
    candidates: tuple[SampleCandidate, ...],
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    global_batch_size: int,
) -> RunPlan:
    """Plan every step in a compiled schedule."""
    steps = tuple(
        plan_step(
            step,
            candidates,
            run_id=run_id,
            branch_id=branch_id,
            seed=seed,
            global_batch_size=global_batch_size,
        )
        for step in schedule_steps
    )
    return RunPlan(
        run_id=run_id,
        branch_id=branch_id,
        seed=seed,
        global_batch_size=global_batch_size,
        steps=steps,
    )


def _exclude_planned(
    candidates: tuple[SampleCandidate, ...],
    planned_ids: set[str],
) -> tuple[SampleCandidate, ...]:
    """Draw without replacement within a global step.

    A document repeated inside one step would be packed twice into the same context
    window, inflating its effective epoch count and teaching the model on duplicated
    text. If a pool is too small to fill the batch, repeating beats leaving slots
    unfilled, so exhaustion falls back to the full pool.
    """
    if not planned_ids:
        return candidates
    remaining = tuple(
        candidate for candidate in candidates if candidate.sample_id not in planned_ids
    )
    return remaining or candidates


def _plan_always_on_slot(
    candidates: tuple[SampleCandidate, ...],
    step: StepSchedule,
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    slot_index: int,
) -> PlannedSample:
    sampled = sample_always_on_candidate(
        candidates,
        step.always_on_sub_shares,
        run_id=run_id,
        branch_id=branch_id,
        seed=seed,
        step=step.step,
        slot_index=slot_index,
    )
    if sampled is None:
        raise ScheduleError(
            f"step {step.step} slot {slot_index}: no Always-ON candidates available"
        )
    candidate, sub_share = sampled
    return PlannedSample(
        sample_id=candidate.sample_id,
        shard_id=candidate.shard_id,
        capability_lane=candidate.capability_lane,
        path="always_on",
        sub_share=sub_share,
    )


def _plan_opus_slot(
    candidates: tuple[SampleCandidate, ...],
    step: StepSchedule,
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    slot_index: int,
) -> PlannedSample:
    lane = _pick_opus_lane(
        step.opus_lane_quotas,
        run_id=run_id,
        branch_id=branch_id,
        seed=seed,
        step=step.step,
        slot_index=slot_index,
    )
    lane_candidates = tuple(
        candidate for candidate in candidates if candidate.capability_lane == lane
    )
    if not lane_candidates:
        lane_candidates = candidates
    if not lane_candidates:
        raise ScheduleError(
            f"step {step.step} slot {slot_index}: no OPUS candidates available"
        )

    candidate = deterministic_choice(
        lane_candidates,
        run_id,
        branch_id,
        seed,
        step.step,
        slot_index,
        "opus",
        lane,
    )
    return PlannedSample(
        sample_id=candidate.sample_id,
        shard_id=candidate.shard_id,
        capability_lane=candidate.capability_lane,
        path="opus",
    )


def _pick_opus_lane(
    quotas: dict[str, float],
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    step: int,
    slot_index: int,
) -> str:
    ordered = sorted(quotas.items())
    positive = [(lane, weight) for lane, weight in ordered if weight > 0]
    if not positive:
        raise ScheduleError(f"step {step} slot {slot_index}: empty OPUS lane quotas")

    total = sum(weight for _, weight in positive)
    threshold = (
        deterministic_unit_float(run_id, branch_id, seed, step, slot_index, "opus_lane") * total
    )
    cumulative = 0.0
    for lane, weight in positive:
        cumulative += weight
        if threshold <= cumulative:
            return lane
    return positive[-1][0]
