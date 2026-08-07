"""Resume a crashed run from a checkpoint plus ledger offset (P9-T02).

Resume restores four things, and all four have to move together or the run is not
actually where it claims to be:

1. **Model and optimizer weights** from `model.pt` / `optimizer.pt`
2. **RNG state**, so dropout and any sampling continue the same stream
3. **The data cursor** (`next_global_step`, `next_microbatch_index`) from the checkpoint
4. **The ledger position**, opened as a new attempt starting at `ledger_offset + 1`

Point 4 is the part a naive resume gets wrong. The crashed attempt committed batches
*after* the checkpoint, so the ledger tail is ahead of the restored cursor. Those rows
are not deleted (the ledger is append-only) and not skipped over (that would lose the
batches between). The resumed attempt re-commits the same offsets under a higher
`attempt` number, which is what lets P9-T03 compare the re-derived batch against the
original byte for byte.

Nothing here reads the `simulated_crash` line from `run.log`. Resume works from the
checkpoint and the ledger alone; if it needed the crash record it would be a simulation
of recovery rather than recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from checkpoint.io import dataloader_state_from_checkpoint, load_checkpoint, restore_rng_state
from checkpoint.types import CheckpointPayload
from ledger import LedgerWriter
from runlog import RunLogWriter

from .crash import CrashPolicy
from .errors import RecoveryError

if TYPE_CHECKING:
    # `trainer.loop` imports `recovery.crash`, so importing it here at module level would
    # deadlock the two packages. Annotations are strings under `from __future__ import
    # annotations`, and the one runtime symbol is imported inside the function.
    from trainer.loop import TrainingContext, TrainingPaths, TrainingRunner


@dataclass(frozen=True)
class ResumedRun:
    """A runner restored from a checkpoint, plus where it was restored to."""

    runner: TrainingRunner
    checkpoint: CheckpointPayload
    prior_attempt: int
    resumed_attempt: int

    @property
    def resume_step(self) -> int:
        return self.checkpoint.next_global_step

    @property
    def resume_ledger_offset(self) -> int:
        """Last offset the checkpoint had consumed; the run re-commits from the next one."""
        return self.checkpoint.ledger_offset


def resume_from_checkpoint(
    context: TrainingContext,
    paths: TrainingPaths,
    *,
    checkpoint_step: int,
    device: torch.device | None = None,
    crash_policy: CrashPolicy | None = None,
    run_log: RunLogWriter | None = None,
) -> ResumedRun:
    """Rebuild a training runner positioned exactly where the checkpoint left off."""
    from trainer.loop import build_training_runner

    payload = load_checkpoint(paths.checkpoints_dir, checkpoint_step)

    demo = context.demo
    if payload.run_id != demo.run.run_id:
        raise RecoveryError(
            f"checkpoint run_id {payload.run_id} does not match {demo.run.run_id}"
        )
    if payload.branch_id != context.run_plan.branch_id:
        raise RecoveryError(
            f"checkpoint branch_id {payload.branch_id} does not match "
            f"{context.run_plan.branch_id}"
        )
    if payload.model_state is None or payload.optimizer_state is None:
        raise RecoveryError(
            f"checkpoint at step {checkpoint_step} has no model/optimizer state; "
            "resume would silently restart from random weights"
        )

    writer = LedgerWriter.resume_at(
        paths.ledger_path,
        ledger_offset=payload.ledger_offset + 1,
    )
    runner = build_training_runner(
        context,
        paths,
        device=device,
        crash_policy=crash_policy,
        writer=writer,
        run_log=run_log,
    )
    runner.run_log.emit(
        "resume_initiated",
        run_id=payload.run_id,
        branch_id=payload.branch_id,
        checkpoint_id=payload.checkpoint_id,
        resume_from_step=payload.next_global_step,
        resume_ledger_offset=payload.ledger_offset,
        prior_attempt=writer.attempt - 1,
        resumed_attempt=writer.attempt,
    )

    runner.trainer.load_state_dicts(payload.model_state, payload.optimizer_state)
    restore_rng_state(payload.rng_state)
    runner.dataloader.restore_state(dataloader_state_from_checkpoint(payload))

    # The dataloader counts the last consumed offset; the writer stamps the next one.
    # If these disagree the run would either overwrite a row or leave a hole.
    if runner.dataloader.next_ledger_offset != writer.next_offset:
        raise RecoveryError(
            f"dataloader resumes at offset {runner.dataloader.next_ledger_offset} but "
            f"the ledger writer is at {writer.next_offset}"
        )

    return ResumedRun(
        runner=runner,
        checkpoint=payload,
        prior_attempt=writer.attempt - 1,
        resumed_attempt=writer.attempt,
    )


def checkpoints_available(checkpoints_dir: Path) -> tuple[int, ...]:
    """Steps that have a complete checkpoint on disk, oldest first."""
    if not checkpoints_dir.is_dir():
        return ()
    steps: list[int] = []
    for entry in sorted(checkpoints_dir.glob("ckpt-*")):
        if not (entry / "checkpoint.json").is_file():
            continue
        try:
            steps.append(int(entry.name.split("-", 1)[1]))
        except (IndexError, ValueError) as exc:
            raise RecoveryError(f"malformed checkpoint directory name: {entry.name}") from exc
    return tuple(steps)
