"""Fork a new branch from an earlier checkpoint (P9-T08).

A fork restores the same weights as resume, then diverges on purpose. The planner is
seeded on `(run_id, branch_id, seed, step, slot)`, so handing it a new `branch_id` makes
it draw a different stream from the same pool, with the same schedule and the same
admitted shards. Nothing else has to change for the branches to separate.

**Branches get their own lineage on disk.** The fork writes to
`branches/<branch_id>/ledgers/consumption.jsonl` and its own checkpoint directory rather
than interleaving into the parent's ledger. A branch is a different history, not a later
part of the same one, and keeping them apart means the parent's append-only ordering
rules stay simple: one file, one lineage, offsets from 0.

The link between them lives in `ledgers/forks.jsonl` on the parent: parent branch, child
branch, the checkpoint forked from, the parent offset at that point, and the step where
the two streams first diverge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import torch

from checkpoint.io import dataloader_state_from_checkpoint, load_checkpoint, restore_rng_state
from ledger import LedgerWriter
from ledger.reader import load_consumption_ledger
from ledger.types import ConsumptionLedgerEvent
from runlog import RunLogWriter
from schedule import plan_run
from schedule.pool import SampleCandidate

from .errors import RecoveryError

if TYPE_CHECKING:
    from trainer.loop import TrainingContext, TrainingPaths, TrainingRunner

FORK_LOG_FILENAME = "forks.jsonl"
FORK_VERIFICATION_FILENAME = "fork_verification.json"


@dataclass(frozen=True)
class ForkEvent:
    """Divergence record written to the parent's `ledgers/forks.jsonl`."""

    run_id: str
    parent_branch_id: str
    child_branch_id: str
    forked_from_checkpoint_id: str
    forked_from_step: int
    parent_ledger_offset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "fork",
            "run_id": self.run_id,
            "parent_branch_id": self.parent_branch_id,
            "child_branch_id": self.child_branch_id,
            "forked_from_checkpoint_id": self.forked_from_checkpoint_id,
            "forked_from_step": self.forked_from_step,
            "parent_ledger_offset": self.parent_ledger_offset,
        }


@dataclass(frozen=True)
class ForkedRun:
    """A runner on a new branch, plus where it split from the parent."""

    runner: TrainingRunner
    paths: TrainingPaths
    event: ForkEvent

    @property
    def branch_id(self) -> str:
        return self.event.child_branch_id


@dataclass(frozen=True)
class ForkVerification:
    """Evidence that the forked stream really is a different history."""

    run_id: str
    parent_branch_id: str
    child_branch_id: str
    forked_from_step: int
    parent_ledger_offset: int
    divergence_step: int | None
    compared_steps: tuple[int, ...]
    diverged_steps: tuple[int, ...]
    child_batches: int

    @property
    def passed(self) -> bool:
        """A fork that reproduced the parent exactly would not be a fork."""
        return (
            self.child_branch_id != self.parent_branch_id
            and self.child_batches > 0
            and self.divergence_step is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parent_branch_id": self.parent_branch_id,
            "child_branch_id": self.child_branch_id,
            "forked_from_step": self.forked_from_step,
            "parent_ledger_offset": self.parent_ledger_offset,
            "divergence_step": self.divergence_step,
            "compared_steps": list(self.compared_steps),
            "diverged_steps": list(self.diverged_steps),
            "child_batches": self.child_batches,
            "passed": self.passed,
        }


def branch_paths(paths: TrainingPaths, branch_id: str) -> TrainingPaths:
    """Artifact paths for one branch, rooted at `branches/<branch_id>/`."""
    from trainer.loop import TrainingPaths as _TrainingPaths

    root = paths.ledger_path.parent.parent / "branches" / branch_id
    return _TrainingPaths.under(root)


def fork_from_checkpoint(
    context: TrainingContext,
    paths: TrainingPaths,
    *,
    checkpoint_step: int,
    new_branch_id: str,
    pool: Sequence[SampleCandidate],
    device: torch.device | None = None,
    run_log: RunLogWriter | None = None,
) -> ForkedRun:
    """Restore `checkpoint_step` onto a new branch and log the divergence point.

    `pool` is the admitted sample pool the parent planned from. The fork has to re-plan
    rather than reuse `context.run_plan`, since that plan is bound to the parent branch.
    """
    from trainer.loop import TrainingContext as _TrainingContext, build_training_runner

    payload = load_checkpoint(paths.checkpoints_dir, checkpoint_step)
    parent_branch = payload.branch_id
    if new_branch_id == parent_branch:
        raise RecoveryError(
            f"fork branch_id must differ from the parent branch {parent_branch!r}"
        )
    if payload.model_state is None or payload.optimizer_state is None:
        raise RecoveryError(
            f"checkpoint at step {checkpoint_step} has no model state; the fork would "
            "start from random weights instead of the parent's"
        )

    # Same schedule, same pool, new branch: the planner diverges because it is seeded on
    # branch_id, so the fork is a genuine alternative history rather than a copy.
    forked_plan = plan_run(
        context.schedule.steps,
        pool,
        run_id=context.demo.run.run_id,
        branch_id=new_branch_id,
        seed=context.demo.run.seed,
        global_batch_size=context.demo.training.global_batch_size,
    )
    forked_context = _TrainingContext(
        demo=context.demo,
        curriculum=context.curriculum,
        schedule=context.schedule,
        run_plan=forked_plan,
        tokenizer=context.tokenizer,
        documents_by_id=context.documents_by_id,
        registry=context.registry,
    )

    child_paths = branch_paths(paths, new_branch_id)
    runner = build_training_runner(
        forked_context,
        child_paths,
        device=device,
        writer=LedgerWriter(child_paths.ledger_path, next_offset=0, attempt=0),
    )
    runner.trainer.load_state_dicts(payload.model_state, payload.optimizer_state)
    restore_rng_state(payload.rng_state)

    # The branch has consumed nothing yet, so its own offsets start at 0. The parent
    # offset it split from is recorded in the fork event, not carried into the child.
    state = dataloader_state_from_checkpoint(payload)
    runner.dataloader.restore_state(
        replace(state, branch_id=new_branch_id, ledger_offset=-1)
    )

    event = ForkEvent(
        run_id=payload.run_id,
        parent_branch_id=parent_branch,
        child_branch_id=new_branch_id,
        forked_from_checkpoint_id=payload.checkpoint_id,
        forked_from_step=payload.next_global_step,
        parent_ledger_offset=payload.ledger_offset,
    )
    append_fork_event(paths.ledger_path.parent / FORK_LOG_FILENAME, event)
    # The divergence is an event in the *parent's* history: the child's log starts at
    # the fork point and cannot record what it branched away from.
    (run_log or RunLogWriter.open(paths.run_log_path)).emit(
        "fork_initiated",
        run_id=event.run_id,
        parent_branch_id=event.parent_branch_id,
        child_branch_id=event.child_branch_id,
        forked_from_checkpoint_id=event.forked_from_checkpoint_id,
        forked_from_step=event.forked_from_step,
        parent_ledger_offset=event.parent_ledger_offset,
        # Relative to submission_artifacts/, so the log does not carry a local path.
        child_ledger_path=child_paths.ledger_path.relative_to(
            paths.run_log_path.parent
        ).as_posix(),
    )
    return ForkedRun(runner=runner, paths=child_paths, event=event)


def append_fork_event(path: Path, event: ForkEvent) -> ForkEvent:
    """Append one divergence record to the parent's fork log."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    return event


def verify_fork(
    parent_paths: TrainingPaths,
    forked: ForkedRun,
) -> ForkVerification:
    """Compare the two branches after the fork point and find where they separate.

    Compared over the **union** of steps each branch committed at, not the intersection.
    The branches draw different sample streams, so they also gate differently: one can
    commit at a step where the other committed nothing. That is a real difference in what
    was consumed, and intersecting would silently drop it (and can leave nothing to
    compare at all when the two happen to miss each other).
    """
    parent_rows = _rows_by_step(load_consumption_ledger(parent_paths.ledger_path))
    child_rows = _rows_by_step(load_consumption_ledger(forked.paths.ledger_path))
    if not child_rows:
        raise RecoveryError(
            f"forked branch {forked.branch_id} committed no batches; nothing to compare"
        )

    fork_step = forked.event.forked_from_step
    steps = sorted(
        {step for step in parent_rows if step >= fork_step}
        | {step for step in child_rows if step >= fork_step}
    )

    compared: list[int] = []
    diverged: list[int] = []
    for step in steps:
        compared.append(step)
        if _samples_at(child_rows, step) != _samples_at(parent_rows, step):
            diverged.append(step)

    return ForkVerification(
        run_id=forked.event.run_id,
        parent_branch_id=forked.event.parent_branch_id,
        child_branch_id=forked.branch_id,
        forked_from_step=forked.event.forked_from_step,
        parent_ledger_offset=forked.event.parent_ledger_offset,
        divergence_step=diverged[0] if diverged else None,
        compared_steps=tuple(compared),
        diverged_steps=tuple(diverged),
        child_batches=sum(len(rows) for rows in child_rows.values()),
    )


def write_fork_verification(reports_dir: Path, verification: ForkVerification) -> Path:
    """Write `reports/fork_verification.json` (P9-T08)."""
    from shards.io import write_json_atomic

    target = Path(reports_dir).resolve() / FORK_VERIFICATION_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, verification.to_dict())
    return target


def _rows_by_step(
    records: tuple[ConsumptionLedgerEvent, ...],
) -> dict[int, list[ConsumptionLedgerEvent]]:
    """Newest attempt per microbatch, grouped by step."""
    newest: dict[str, ConsumptionLedgerEvent] = {}
    for record in records:
        current = newest.get(record.microbatch_id)
        if current is None or record.attempt > current.attempt:
            newest[record.microbatch_id] = record

    grouped: dict[int, list[ConsumptionLedgerEvent]] = {}
    for record in newest.values():
        grouped.setdefault(record.global_step, []).append(record)
    return grouped


def _samples_at(
    rows_by_step: dict[int, list[ConsumptionLedgerEvent]],
    step: int,
) -> tuple[str, ...]:
    """Sample IDs consumed at one step, order-independent; empty if nothing committed."""
    samples: list[str] = []
    for record in rows_by_step.get(step, ()):
        samples.extend(record.packed_sample_ids)
    return tuple(sorted(samples))
