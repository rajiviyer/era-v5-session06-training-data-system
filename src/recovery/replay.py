"""Replay a historical step range and prove it reconstructs (P9-T05–T07).

Replay is not resume. It does not train, does not update weights, and appends nothing to
the consumption ledger. It answers one question: *can the run reproduce a stretch of
history it already recorded?*

Two independent checks per microbatch, which is what stops this from being circular:

1. **The planner is re-run** from `(run_id, branch_id, seed, step, microbatch_index)` and
   must produce the same sample IDs the ledger recorded. This proves the sampling stream
   is reproducible.
2. **The batch is rebuilt** from the ledger's recorded sample IDs by tokenizing, packing,
   and masking again, and the recomputed `batch_content_hash`, `loss_mask_hash`, and
   token spans must equal the recorded ones. This proves the data path is deterministic.

Reading a hash out of the ledger and comparing it to itself would prove nothing, so
neither check does that: both sides are regenerated and the ledger is only the record
they are held against.

When a range spans a crash, the *effective* stream is replayed: for each microbatch the
newest attempt wins, because that is the batch the model actually carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

from ledger import LedgerBoundDataLoader
from ledger.commit import flatten_token_span_ids, parse_microbatch_id
from ledger.reader import load_consumption_ledger
from ledger.rebuild import rebuild_batch
from ledger.types import ConsumptionLedgerEvent
from runlog import RunLogWriter

from .errors import RecoveryError

if TYPE_CHECKING:
    from trainer.loop import TrainingContext, TrainingPaths

REPLAY_VERIFICATION_FILENAME = "replay_verification.json"


@dataclass(frozen=True)
class ReplayComparison:
    """One historical microbatch against its reconstruction.

    Kept separate from `BatchComparison` (resume) on purpose: resume pairs two *recorded*
    rows, while replay pairs a recorded row against a *recomputed* batch and additionally
    re-runs the planner. Sharing one type would leave meaningless fields on both sides.
    """

    global_step: int
    microbatch_index: int
    microbatch_id: str
    ledger_offset: int
    attempt: int
    planned_sample_ids_match: bool
    recorded_batch_content_hash: str
    recomputed_batch_content_hash: str
    recorded_loss_mask_hash: str
    recomputed_loss_mask_hash: str
    token_spans_match: bool

    @property
    def matched(self) -> bool:
        return (
            self.planned_sample_ids_match
            and self.recorded_batch_content_hash == self.recomputed_batch_content_hash
            and self.recorded_loss_mask_hash == self.recomputed_loss_mask_hash
            and self.token_spans_match
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "microbatch_index": self.microbatch_index,
            "microbatch_id": self.microbatch_id,
            "ledger_offset": self.ledger_offset,
            "attempt": self.attempt,
            "planned_sample_ids_match": self.planned_sample_ids_match,
            "recorded_batch_content_hash": self.recorded_batch_content_hash,
            "recomputed_batch_content_hash": self.recomputed_batch_content_hash,
            "recorded_loss_mask_hash": self.recorded_loss_mask_hash,
            "recomputed_loss_mask_hash": self.recomputed_loss_mask_hash,
            "token_spans_match": self.token_spans_match,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class ReplayVerification:
    """Result of replaying `[start_step, end_step]`."""

    run_id: str
    branch_id: str
    start_step: int
    end_step: int
    comparisons: tuple[ReplayComparison, ...]
    steps_replayed: tuple[int, ...]
    steps_without_commits: tuple[int, ...]

    @property
    def mismatched(self) -> tuple[ReplayComparison, ...]:
        return tuple(item for item in self.comparisons if not item.matched)

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and not self.mismatched

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "batches_replayed": len(self.comparisons),
            "batches_matched": len(self.comparisons) - len(self.mismatched),
            "steps_replayed": list(self.steps_replayed),
            "steps_without_commits": list(self.steps_without_commits),
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


def replay_range(
    context: TrainingContext,
    paths: TrainingPaths,
    *,
    start_step: int,
    end_step: int,
    run_log: RunLogWriter | None = None,
) -> ReplayVerification:
    """Reconstruct every committed microbatch in `[start_step, end_step]`."""
    if end_step < start_step:
        raise RecoveryError(f"replay range {start_step}..{end_step} is empty")

    records = load_consumption_ledger(paths.ledger_path)
    if not records:
        raise RecoveryError(f"cannot replay an empty ledger: {paths.ledger_path}")

    rows = _effective_rows(records, start_step=start_step, end_step=end_step)
    if not rows:
        raise RecoveryError(
            f"no committed batches in replay range {start_step}..{end_step}"
        )

    if run_log is not None:
        run_log.emit(
            "replay_initiated",
            run_id=rows[0].run_id,
            branch_id=rows[0].branch_id,
            start_step=start_step,
            end_step=end_step,
            batches_to_replay=len(rows),
        )

    demo = context.demo
    comparisons = [_replay_one(row, context=context) for row in rows]
    replayed_steps = sorted({row.global_step for row in rows})
    missing = tuple(
        step
        for step in range(start_step, min(end_step, demo.training.total_steps - 1) + 1)
        if step not in set(replayed_steps)
    )

    return ReplayVerification(
        run_id=rows[0].run_id,
        branch_id=rows[0].branch_id,
        start_step=start_step,
        end_step=end_step,
        comparisons=tuple(comparisons),
        steps_replayed=tuple(replayed_steps),
        steps_without_commits=missing,
    )


def write_replay_verification(reports_dir: Path, verification: ReplayVerification) -> Path:
    """Write `reports/replay_verification.json` (P9-T07)."""
    from shards.io import write_json_atomic

    target = Path(reports_dir).resolve() / REPLAY_VERIFICATION_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, verification.to_dict())
    return target


def _replay_one(
    row: ConsumptionLedgerEvent,
    *,
    context: TrainingContext,
) -> ReplayComparison:
    global_step, microbatch_index = parse_microbatch_id(row.microbatch_id)
    demo = context.demo

    # Check 1: put a fresh dataloader at this exact position and re-plan.
    loader = LedgerBoundDataLoader(
        context.run_plan,
        microbatch_size=demo.training.microbatch_size,
        global_batch_size=demo.training.global_batch_size,
        run_id=row.run_id,
        branch_id=row.branch_id,
        next_global_step=global_step,
        next_microbatch_index=microbatch_index,
    )
    planned = loader.next_microbatch()
    planned_ids = tuple(sample.sample_id for sample in planned.samples)

    # Check 2: rebuild the batch from what the ledger says was consumed.
    assembled = rebuild_batch(
        row,
        documents_by_id=context.documents_by_id,
        tokenizer=context.tokenizer,
        seq_len=demo.training.seq_len,
    )

    return ReplayComparison(
        global_step=global_step,
        microbatch_index=microbatch_index,
        microbatch_id=row.microbatch_id,
        ledger_offset=row.ledger_offset,
        attempt=row.attempt,
        planned_sample_ids_match=planned_ids == row.packed_sample_ids,
        recorded_batch_content_hash=row.batch_content_hash,
        recomputed_batch_content_hash=assembled.batch.batch_content_hash,
        recorded_loss_mask_hash=row.loss_mask_hash,
        recomputed_loss_mask_hash=assembled.batch.loss_mask_hash,
        token_spans_match=flatten_token_span_ids(assembled.batch) == row.token_span_ids,
    )


def _effective_rows(
    records: Sequence[ConsumptionLedgerEvent],
    *,
    start_step: int,
    end_step: int,
) -> tuple[ConsumptionLedgerEvent, ...]:
    """Rows in range, newest attempt per microbatch, in consumption order."""
    newest: dict[str, ConsumptionLedgerEvent] = {}
    for record in records:
        if not start_step <= record.global_step <= end_step:
            continue
        current = newest.get(record.microbatch_id)
        if current is None or record.attempt > current.attempt:
            newest[record.microbatch_id] = record
    return tuple(
        sorted(newest.values(), key=lambda record: (record.global_step, record.microbatch_id))
    )
