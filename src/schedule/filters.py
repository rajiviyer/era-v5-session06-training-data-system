"""Candidate filters for OPUS and Always-ON paths (P3-T05)."""

from __future__ import annotations

from .pool import SampleCandidate
from .types import StepSchedule


def filter_always_on_candidates(
    candidates: tuple[SampleCandidate, ...],
) -> tuple[SampleCandidate, ...]:
    """Return documents eligible for the Always-ON floor path."""
    return tuple(
        candidate
        for candidate in candidates
        if candidate.always_on_eligible
    )


def filter_opus_candidates(
    candidates: tuple[SampleCandidate, ...],
    step: StepSchedule,
) -> tuple[SampleCandidate, ...]:
    """Apply anneal reserve and OPUS eligibility rules for one step."""
    filtered: list[SampleCandidate] = []
    for candidate in candidates:
        if not candidate.opus_eligible:
            continue
        if step.anneal_eligible_only:
            if not candidate.anneal_eligible:
                continue
        elif candidate.anneal_eligible:
            continue
        filtered.append(candidate)
    return tuple(filtered)
