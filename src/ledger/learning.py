"""Learning ledger events: what the model actually learned from (P8-T01, P8-T02).

The consumption ledger answers *what was fed to the model*. The learning ledger answers
*what came back*: one append-only row per (committed microbatch, document) with the loss
that document earned, the OPUS score that admitted it, and the ledger offset that ties
it to the exact consumption row.

**Granularity (decision D5): sample-level.** One row per document, not per packed
sequence. Concat-and-chop puts several documents in one sequence, so a per-sequence row
would blur two shards together and make per-shard aggregates fiction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .errors import LedgerError
from .types import ConsumptionLedgerEvent

if TYPE_CHECKING:
    from trainer.loss import DocumentLoss

LEARNING_LEDGER_FILENAME = "learning.jsonl"
LEARNING_SCHEMA_VERSION = "1.0"

ANNEAL_STAGE = "anneal"
MODEL_PHASES: frozenset[str] = frozenset({"early", "mid", "late", "anneal"})

# Losses are rounded before they are recorded so the in-memory event and the JSONL row
# hold the same number: an aggregate recomputed from the file must match one computed
# from the returned events exactly, not approximately.
LOSS_DECIMALS = 6


@dataclass(frozen=True)
class LearningLedgerEvent:
    """One append-only learning ledger row (SCOPE.md §6.8)."""

    run_id: str
    branch_id: str
    global_step: int
    ledger_offset: int
    microbatch_id: str
    sample_id: str
    shard_id: str
    capability_lane: str
    curriculum_stage: str
    model_phase: str
    loss_bearing_tokens: int
    mean_loss: float
    perplexity: float
    opus_score: float | None
    opus_decision_id: str
    batch_content_hash: str
    attempt: int = 0
    """Crash-resume generation, inherited from the consumption row this loss came from."""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": LEARNING_SCHEMA_VERSION,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "attempt": self.attempt,
            "global_step": self.global_step,
            "ledger_offset": self.ledger_offset,
            "microbatch_id": self.microbatch_id,
            "sample_id": self.sample_id,
            "shard_id": self.shard_id,
            "capability_lane": self.capability_lane,
            "curriculum_stage": self.curriculum_stage,
            "model_phase": self.model_phase,
            "loss_bearing_tokens": self.loss_bearing_tokens,
            "mean_loss": self.mean_loss,
            "perplexity": self.perplexity,
            "opus_decision_id": self.opus_decision_id,
            "batch_content_hash": self.batch_content_hash,
        }
        if self.opus_score is not None:
            payload["opus_score"] = round(self.opus_score, 6)
        return payload


def model_phase_for_step(global_step: int, total_steps: int, curriculum_stage: str) -> str:
    """Coarse training phase for a step (SCOPE.md §6.8).

    Anneal is a curriculum fact, not a position on the timeline, so the stage name wins
    over the step fraction when the run reaches it.
    """
    if total_steps <= 0:
        raise LedgerError("total_steps must be positive")
    if curriculum_stage == ANNEAL_STAGE:
        return ANNEAL_STAGE
    fraction = global_step / total_steps
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "mid"
    return "late"


def build_learning_events(
    document_losses: Sequence[DocumentLoss],
    consumption_event: ConsumptionLedgerEvent,
    *,
    lanes_by_sample: Mapping[str, str],
    shards_by_sample: Mapping[str, str],
    opus_score: float | None,
    total_steps: int,
) -> tuple[LearningLedgerEvent, ...]:
    """Build the learning rows for one committed microbatch.

    Every identity field is copied from the consumption event rather than recomputed, so
    a learning row can never claim a step, offset, or batch hash the run did not commit
    (P8-T04).
    """
    if not document_losses:
        raise LedgerError(
            f"microbatch {consumption_event.microbatch_id} trained but attributed "
            "no loss to any document"
        )

    phase = model_phase_for_step(
        consumption_event.global_step,
        total_steps,
        consumption_event.curriculum_stage,
    )

    events: list[LearningLedgerEvent] = []
    for document_loss in document_losses:
        sample_id = document_loss.document_id
        shard_id = shards_by_sample.get(sample_id)
        if shard_id is None:
            raise LedgerError(
                f"document {sample_id} carried loss but is not in the committed "
                f"microbatch {consumption_event.microbatch_id}"
            )
        mean_loss = round(document_loss.mean_loss, LOSS_DECIMALS)
        events.append(
            LearningLedgerEvent(
                run_id=consumption_event.run_id,
                branch_id=consumption_event.branch_id,
                global_step=consumption_event.global_step,
                ledger_offset=consumption_event.ledger_offset,
                microbatch_id=consumption_event.microbatch_id,
                sample_id=sample_id,
                shard_id=shard_id,
                capability_lane=lanes_by_sample.get(
                    sample_id, consumption_event.mixture_lane
                ),
                curriculum_stage=consumption_event.curriculum_stage,
                model_phase=phase,
                loss_bearing_tokens=document_loss.loss_bearing_tokens,
                mean_loss=mean_loss,
                perplexity=perplexity_from_loss(mean_loss),
                opus_score=opus_score,
                opus_decision_id=consumption_event.opus_decision_id,
                batch_content_hash=consumption_event.batch_content_hash,
                attempt=consumption_event.attempt,
            )
        )
    return tuple(validate_learning_event(event) for event in events)


def perplexity_from_loss(mean_loss: float) -> float:
    """Perplexity of a mean cross-entropy loss, rounded like the loss it derives from."""
    if not math.isfinite(mean_loss):
        raise LedgerError(f"mean_loss must be finite, got {mean_loss}")
    return round(math.exp(mean_loss), LOSS_DECIMALS)


def validate_learning_event(event: LearningLedgerEvent) -> LearningLedgerEvent:
    """Validate one learning ledger event."""
    for field in ("run_id", "branch_id", "microbatch_id", "sample_id", "shard_id"):
        if not getattr(event, field):
            raise LedgerError(f"{field} is required")
    if not event.capability_lane:
        raise LedgerError("capability_lane is required")
    if not event.curriculum_stage:
        raise LedgerError("curriculum_stage is required")
    if event.model_phase not in MODEL_PHASES:
        raise LedgerError(f"invalid model_phase: {event.model_phase}")
    if event.attempt < 0:
        raise LedgerError("attempt must be non-negative")
    if event.global_step < 0:
        raise LedgerError("global_step must be non-negative")
    if event.ledger_offset < 0:
        raise LedgerError("ledger_offset must be non-negative")
    if event.loss_bearing_tokens <= 0:
        raise LedgerError("loss_bearing_tokens must be positive")
    if not math.isfinite(event.mean_loss) or event.mean_loss < 0:
        raise LedgerError(f"mean_loss must be finite and non-negative: {event.mean_loss}")
    # P8-T03 requires perplexity to be derivable from the recorded loss; enforce it here
    # so no reader has to trust that the two columns were computed together.
    if not math.isclose(event.perplexity, perplexity_from_loss(event.mean_loss), rel_tol=1e-9):
        raise LedgerError(
            f"perplexity {event.perplexity} does not match exp(mean_loss) for "
            f"mean_loss {event.mean_loss}"
        )
    if not event.opus_decision_id:
        raise LedgerError("opus_decision_id is required")
    if not event.batch_content_hash:
        raise LedgerError("batch_content_hash is required")
    return event


def learning_event_from_dict(payload: dict[str, object]) -> LearningLedgerEvent:
    """Parse one learning ledger event from JSON."""
    version = payload.get("schema_version", LEARNING_SCHEMA_VERSION)
    if version != LEARNING_SCHEMA_VERSION:
        raise LedgerError(f"unsupported schema_version: {version}")

    required = (
        "run_id",
        "branch_id",
        "global_step",
        "ledger_offset",
        "microbatch_id",
        "sample_id",
        "shard_id",
        "capability_lane",
        "curriculum_stage",
        "model_phase",
        "loss_bearing_tokens",
        "mean_loss",
        "perplexity",
        "opus_decision_id",
        "batch_content_hash",
    )
    for key in required:
        if key not in payload:
            raise LedgerError(f"learning ledger event missing key: {key}")

    raw_score = payload.get("opus_score")
    event = LearningLedgerEvent(
        run_id=str(payload["run_id"]),
        branch_id=str(payload["branch_id"]),
        global_step=int(payload["global_step"]),  # type: ignore[arg-type]
        ledger_offset=int(payload["ledger_offset"]),  # type: ignore[arg-type]
        microbatch_id=str(payload["microbatch_id"]),
        sample_id=str(payload["sample_id"]),
        shard_id=str(payload["shard_id"]),
        capability_lane=str(payload["capability_lane"]),
        curriculum_stage=str(payload["curriculum_stage"]),
        model_phase=str(payload["model_phase"]),
        loss_bearing_tokens=int(payload["loss_bearing_tokens"]),  # type: ignore[arg-type]
        mean_loss=float(payload["mean_loss"]),  # type: ignore[arg-type]
        perplexity=float(payload["perplexity"]),  # type: ignore[arg-type]
        opus_score=None if raw_score is None else float(raw_score),  # type: ignore[arg-type]
        opus_decision_id=str(payload["opus_decision_id"]),
        batch_content_hash=str(payload["batch_content_hash"]),
        # Rows written before the field existed are attempt 0: the first, uninterrupted pass.
        attempt=int(payload.get("attempt", 0)),  # type: ignore[arg-type]
    )
    return validate_learning_event(event)


def append_learning_events(
    path: Path,
    events: Sequence[LearningLedgerEvent],
) -> tuple[LearningLedgerEvent, ...]:
    """Append validated learning events to learning.jsonl."""
    if not events:
        return ()
    validated = tuple(validate_learning_event(event) for event in events)
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for event in validated:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return validated


def load_learning_ledger(path: Path) -> tuple[LearningLedgerEvent, ...]:
    """Load all learning ledger events from disk."""
    target = path.resolve()
    if not target.is_file():
        return ()

    records: list[LearningLedgerEvent] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"invalid JSON in {target} line {line_number}") from exc
        if not isinstance(payload, dict):
            raise LedgerError(f"learning ledger line {line_number} must be a JSON object")
        records.append(learning_event_from_dict(payload))

    _validate_non_decreasing_offsets(records)
    return tuple(records)


def _validate_non_decreasing_offsets(records: list[LearningLedgerEvent]) -> None:
    """Append-only ordering.

    One microbatch emits several rows, so offsets repeat within an attempt but must never
    go backwards. A resumed attempt rewinds the offset on purpose, so the rewind is only
    allowed when the attempt number goes up.
    """
    previous_attempt = 0
    previous_offset = -1
    for index, record in enumerate(records):
        if record.attempt < previous_attempt:
            raise LedgerError(
                f"learning ledger row {index} has attempt {record.attempt} after "
                f"{previous_attempt}; the ledger is append-only"
            )
        if record.attempt == previous_attempt and record.ledger_offset < previous_offset:
            raise LedgerError(
                f"learning ledger row {index} has ledger_offset {record.ledger_offset} "
                f"after {previous_offset} in attempt {record.attempt}; "
                "the ledger is append-only"
            )
        previous_attempt = record.attempt
        previous_offset = record.ledger_offset
