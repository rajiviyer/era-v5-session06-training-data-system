"""Always-ON floor sampler (P3-T04)."""

from __future__ import annotations

from collections.abc import Callable

from .deterministic import deterministic_choice, deterministic_unit_float
from .pool import SampleCandidate

SubShareRule = Callable[[SampleCandidate], bool]

SUB_SHARE_RULES: dict[str, SubShareRule] = {
    "indic_tier_a": lambda candidate: (
        candidate.capability_lane == "indic" and candidate.indic_tier == "A"
    ),
    "en_in_code_mixed": lambda candidate: (
        candidate.capability_lane == "code" and candidate.language in {"en", "en-IN"}
    ),
    "india_curated": lambda candidate: (
        candidate.capability_lane == "indic" and candidate.language not in {"en", "en-IN"}
    ),
    "benchmark_train_format": lambda candidate: (
        candidate.capability_lane == "web" and candidate.curriculum_band in {"B3", "B4", "B5"}
    ),
    "agentic_format": lambda candidate: candidate.capability_lane == "agentic",
    "reasoning_short": lambda candidate: (
        candidate.capability_lane == "reasoning" and candidate.reasoning_trace_band == "short"
    ),
}


def is_always_on_slot(
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    step: int,
    slot_index: int,
    always_on_fraction: float,
) -> bool:
    """Decide whether one batch slot uses the Always-ON path."""
    value = deterministic_unit_float(run_id, branch_id, seed, step, slot_index, "path")
    return value < always_on_fraction


def pick_always_on_sub_share(
    sub_shares: dict[str, float],
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    step: int,
    slot_index: int,
) -> str:
    """Pick one Always-ON sub-share using curriculum weights."""
    ordered = sorted(sub_shares.items())
    total = sum(weight for _, weight in ordered)
    threshold = deterministic_unit_float(run_id, branch_id, seed, step, slot_index, "sub_share") * total
    cumulative = 0.0
    for name, weight in ordered:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return ordered[-1][0]


def candidates_for_sub_share(
    candidates: tuple[SampleCandidate, ...],
    sub_share: str,
) -> tuple[SampleCandidate, ...]:
    """Filter Always-ON candidates for one sub-share bucket."""
    rule = SUB_SHARE_RULES.get(sub_share)
    if rule is None:
        return ()
    return tuple(candidate for candidate in candidates if rule(candidate))


def sample_always_on_candidate(
    candidates: tuple[SampleCandidate, ...],
    sub_shares: dict[str, float],
    *,
    run_id: str,
    branch_id: str,
    seed: int,
    step: int,
    slot_index: int,
) -> tuple[SampleCandidate, str] | None:
    """Sample one Always-ON document, honoring sub-share quotas."""
    if not candidates:
        return None

    ordered_shares = sorted(sub_shares)
    for offset in range(len(ordered_shares)):
        sub_share = pick_always_on_sub_share(
            sub_shares,
            run_id=run_id,
            branch_id=branch_id,
            seed=seed,
            step=step,
            slot_index=slot_index + offset,
        )
        bucket = candidates_for_sub_share(candidates, sub_share)
        if bucket:
            candidate = deterministic_choice(
                bucket,
                run_id,
                branch_id,
                seed,
                step,
                slot_index,
                "always_on",
                sub_share,
            )
            return candidate, sub_share

    fallback = deterministic_choice(
        candidates,
        run_id,
        branch_id,
        seed,
        step,
        slot_index,
        "always_on_fallback",
    )
    return fallback, "fallback"
