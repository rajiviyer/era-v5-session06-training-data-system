"""Per-shard learning aggregates and consumption links (P8-T03, P8-T04).

Everything here is recomputed from `learning.jsonl` and `consumption.jsonl`. Nothing is
carried over from the training run in memory, which is exactly how the P11 evidence
collector will use it: read the artifacts, redo the arithmetic, compare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .errors import LedgerError
from .learning import LearningLedgerEvent, perplexity_from_loss
from .types import ConsumptionLedgerEvent

# A toy run of 50 steps is noisy, so only a loss move larger than this counts as a
# trend. The label is a coarse signal for V6 corpus planning (SCOPE.md §6.8), not a
# verdict on the data.
USEFULNESS_MARGIN = 0.05

Usefulness = str


@dataclass(frozen=True)
class ShardExposure:
    """One time a shard's documents were trained on."""

    global_step: int
    ledger_offset: int
    loss_bearing_tokens: int
    mean_loss: float


@dataclass(frozen=True)
class ShardLearningAggregate:
    """Loss and perplexity for one shard across the whole run."""

    shard_id: str
    capability_lane: str
    sample_ids: tuple[str, ...]
    exposures: tuple[ShardExposure, ...]
    loss_bearing_tokens: int
    mean_loss: float
    perplexity: float
    first_step: int
    last_step: int
    loss_delta: float
    usefulness: Usefulness

    @property
    def exposure_count(self) -> int:
        return len(self.exposures)


def effective_events(
    events: Sequence[LearningLedgerEvent],
) -> tuple[LearningLedgerEvent, ...]:
    """Drop rows a later attempt superseded.

    A crash rolls the model back to a checkpoint, so the loss rows the crashed attempt
    wrote after that checkpoint describe weight updates that no longer exist. Counting
    them would inflate both exposure counts and the loss trend with learning the model
    does not carry. For each (step, microbatch, sample) the newest attempt wins; rows
    with no newer version are kept untouched.
    """
    newest: dict[tuple[int, str, str], LearningLedgerEvent] = {}
    for event in events:
        key = (event.global_step, event.microbatch_id, event.sample_id)
        current = newest.get(key)
        if current is None or event.attempt > current.attempt:
            newest[key] = event
    return tuple(newest.values())


def aggregate_by_shard(
    events: Sequence[LearningLedgerEvent],
) -> tuple[ShardLearningAggregate, ...]:
    """Token-weighted loss per shard, with one exposure record per (shard, step).

    Weighting by loss-bearing tokens rather than averaging row means keeps a document
    that contributed four tokens from counting as much as one that contributed four
    hundred. Superseded attempts are excluded; see `effective_events`.
    """
    if not events:
        return ()

    grouped: dict[str, list[LearningLedgerEvent]] = {}
    for event in effective_events(events):
        grouped.setdefault(event.shard_id, []).append(event)

    aggregates: list[ShardLearningAggregate] = []
    for shard_id in sorted(grouped):
        shard_events = grouped[shard_id]
        exposures = _exposures_for_shard(shard_events)
        total_tokens = sum(event.loss_bearing_tokens for event in shard_events)
        if total_tokens <= 0:
            raise LedgerError(f"shard {shard_id} has no loss-bearing tokens")
        weighted = sum(
            event.mean_loss * event.loss_bearing_tokens for event in shard_events
        )
        mean_loss = weighted / total_tokens
        loss_delta = exposures[-1].mean_loss - exposures[0].mean_loss

        aggregates.append(
            ShardLearningAggregate(
                shard_id=shard_id,
                capability_lane=shard_events[0].capability_lane,
                sample_ids=tuple(sorted({event.sample_id for event in shard_events})),
                exposures=exposures,
                loss_bearing_tokens=total_tokens,
                mean_loss=mean_loss,
                perplexity=perplexity_from_loss(mean_loss),
                first_step=exposures[0].global_step,
                last_step=exposures[-1].global_step,
                loss_delta=loss_delta,
                usefulness=classify_usefulness(len(exposures), loss_delta),
            )
        )
    return tuple(aggregates)


def classify_usefulness(exposure_count: int, loss_delta: float) -> Usefulness:
    """Label a shard by how its loss moved across exposures (SCOPE.md §6.8).

    One exposure cannot show a trend, so it is `review` rather than a guess.
    """
    if exposure_count < 2:
        return "review"
    if loss_delta <= -USEFULNESS_MARGIN:
        return "useful"
    if loss_delta >= USEFULNESS_MARGIN:
        return "harmful"
    return "neutral"


def _exposures_for_shard(
    shard_events: list[LearningLedgerEvent],
) -> tuple[ShardExposure, ...]:
    """Collapse a shard's rows into one token-weighted record per global step."""
    by_step: dict[int, list[LearningLedgerEvent]] = {}
    for event in shard_events:
        by_step.setdefault(event.global_step, []).append(event)

    exposures: list[ShardExposure] = []
    for step in sorted(by_step):
        step_events = by_step[step]
        tokens = sum(event.loss_bearing_tokens for event in step_events)
        weighted = sum(event.mean_loss * event.loss_bearing_tokens for event in step_events)
        exposures.append(
            ShardExposure(
                global_step=step,
                ledger_offset=min(event.ledger_offset for event in step_events),
                loss_bearing_tokens=tokens,
                mean_loss=weighted / tokens,
            )
        )
    return tuple(exposures)


@dataclass(frozen=True)
class LearningLinkReport:
    """Whether every learning row lines up with a committed consumption row (P8-T04)."""

    learning_rows: int
    committed_batches: int
    linked_offsets: int
    orphan_offsets: tuple[tuple[int, int], ...]
    """`(attempt, ledger_offset)` pairs claiming a batch that was never committed."""
    unreported_offsets: tuple[tuple[int, int], ...]
    """`(attempt, ledger_offset)` pairs the model trained on but never reported loss for."""
    mismatches: tuple[str, ...]

    @property
    def linked(self) -> bool:
        return not self.orphan_offsets and not self.unreported_offsets and not self.mismatches


def verify_learning_links(
    learning_events: Sequence[LearningLedgerEvent],
    consumption_events: Sequence[ConsumptionLedgerEvent],
) -> LearningLinkReport:
    """Join the two ledgers on `(attempt, ledger_offset)` and report every disagreement.

    Three failures are distinguished because they mean different things: an orphan is a
    loss recorded against a batch that was never committed, an unreported offset is a
    batch the model trained on but never reported loss for, and a mismatch is a row whose
    step, shard, or batch hash contradicts the consumption record.

    The join key includes `attempt` so a resumed run cannot appear linked by matching a
    superseded row from the crashed attempt.
    """
    consumption_by_key = {
        (event.attempt, event.ledger_offset): event for event in consumption_events
    }
    orphans: list[tuple[int, int]] = []
    mismatches: list[str] = []
    reported: set[tuple[int, int]] = set()

    for event in learning_events:
        key = (event.attempt, event.ledger_offset)
        committed = consumption_by_key.get(key)
        label = f"attempt {event.attempt} offset {event.ledger_offset}"
        if committed is None:
            orphans.append(key)
            continue
        reported.add(key)
        if event.global_step != committed.global_step:
            mismatches.append(
                f"{label}: global_step {event.global_step} != {committed.global_step}"
            )
        if event.shard_id not in committed.shard_ids:
            mismatches.append(
                f"{label}: shard {event.shard_id} not in committed "
                f"shard_ids {list(committed.shard_ids)}"
            )
        if event.batch_content_hash != committed.batch_content_hash:
            mismatches.append(
                f"{label}: batch_content_hash does not match the consumption row"
            )

    unreported = tuple(sorted(key for key in consumption_by_key if key not in reported))
    return LearningLinkReport(
        learning_rows=len(learning_events),
        committed_batches=len(consumption_by_key),
        linked_offsets=len(reported),
        orphan_offsets=tuple(sorted(set(orphans))),
        unreported_offsets=unreported,
        mismatches=tuple(dict.fromkeys(mismatches)),
    )
