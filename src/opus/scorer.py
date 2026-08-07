"""Deterministic OPUS scoring interface (P5-T01)."""

from __future__ import annotations

from typing import Protocol

from schedule.deterministic import deterministic_unit_float

from .types import OpusCandidateContext

_BAND_QUALITY: dict[str, float] = {
    "B1": 0.08,
    "B2": 0.05,
    "B3": 0.02,
    "B4": 0.0,
    "B5": -0.05,
}


class OpusScorer(Protocol):
    """Deterministic, reproducible OPUS scorer."""

    def score(self, context: OpusCandidateContext) -> float:
        """Return a score in [0, 1]. Higher is more acceptable."""


class DeterministicOpusScorer:
    """Hash- and metadata-based scorer with no hidden random state."""

    def score(self, context: OpusCandidateContext) -> float:
        base = deterministic_unit_float(
            context.run_id,
            context.branch_id,
            context.seed,
            context.global_step,
            context.candidate_id,
            *context.shard_ids,
            *context.content_hashes,
            context.capability_lane,
            context.curriculum_stage,
            context.path,
        )
        band_bonus = 0.0
        if context.curriculum_band is not None:
            band_bonus = _BAND_QUALITY.get(context.curriculum_band, 0.0)
        return min(1.0, max(0.0, base + band_bonus))
