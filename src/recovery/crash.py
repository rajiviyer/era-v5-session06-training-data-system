"""Deliberate crash injection for the recovery demo (P9-T01).

The assignment requires a run that *actually* stops mid-flight, not a run that pretends
to. So this module aborts the training loop by raising out of it: the runner never
returns a summary, in-memory model and optimizer state are lost, accumulated gradients
are discarded unapplied, and everything the next phase can use is whatever already
reached disk.

**What a crash leaves behind:**

- `consumption.jsonl` and `learning.jsonl` rows for every microbatch committed before
  the abort, including a partially consumed step
- `opus_audit.jsonl` rows for every gate decision, committed or not
- checkpoints only at completed intervals, so the newest checkpoint is *behind* the
  ledger tail
- one `simulated_crash` line in `run.log`

That gap between the last checkpoint and the ledger tail is the whole point: resume has
to reconcile them (P9-T02), and it must do so from the checkpoint and the ledger alone.
Nothing in the recovery path is allowed to read the crash log line; it is evidence for
the reader, not an input to recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.schemas import RecoveryConfig
from ledger.types import DataLoaderState
from runlog import RunLogWriter

from .errors import RecoveryError

CRASH_EVENT_TYPE = "simulated_crash"


class SimulatedCrash(RuntimeError):
    """Raised to abort a run at the configured crash point.

    Carries the position the run died at so the caller can report it. Recovery must not
    depend on this object: after a real crash the process is gone and only disk remains.
    """

    def __init__(
        self,
        *,
        run_id: str,
        branch_id: str,
        global_step: int,
        microbatch_index: int,
        ledger_offset: int,
    ) -> None:
        super().__init__(
            f"simulated crash in run {run_id} branch {branch_id} at step {global_step} "
            f"microbatch {microbatch_index} (ledger_offset {ledger_offset})"
        )
        self.run_id = run_id
        self.branch_id = branch_id
        self.global_step = global_step
        self.microbatch_index = microbatch_index
        self.ledger_offset = ledger_offset


@dataclass(frozen=True)
class CrashPolicy:
    """Where in the run the deliberate crash fires.

    `crash_after_microbatches` counts microbatches of `crash_at_step` that are attempted
    before the abort. The default of 1 kills the run **mid-step**, which is the case
    worth proving: the ledger ends part-way through a step, so resume cannot simply
    restart at a clean step boundary. Set it to 0 to crash on the step boundary instead.
    """

    crash_at_step: int
    crash_after_microbatches: int = 1

    def __post_init__(self) -> None:
        if self.crash_at_step < 0:
            raise RecoveryError("crash_at_step must be non-negative")
        if self.crash_after_microbatches < 0:
            raise RecoveryError("crash_after_microbatches must be non-negative")

    @classmethod
    def from_config(
        cls,
        recovery: RecoveryConfig,
        *,
        crash_after_microbatches: int = 1,
    ) -> CrashPolicy:
        """Build the demo crash policy from `demo.yaml`'s recovery block."""
        return cls(
            crash_at_step=recovery.crash_at_step,
            crash_after_microbatches=crash_after_microbatches,
        )

    def should_crash(self, state: DataLoaderState) -> bool:
        """True when the dataloader cursor has reached the crash point.

        The cursor is the authority: `next_microbatch_index` already counts every
        microbatch of this step that was attempted, committed or gated, so no separate
        counter can drift away from it.
        """
        return (
            state.next_global_step == self.crash_at_step
            and state.next_microbatch_index >= self.crash_after_microbatches
        )


@dataclass(frozen=True)
class CrashEvent:
    """Structured `simulated_crash` record for run.log (P9-T01)."""

    event_type: str
    run_id: str
    branch_id: str
    global_step: int
    microbatch_index: int
    ledger_offset: int
    last_checkpoint_id: str | None

    def to_log_fields(self) -> dict[str, object]:
        """Payload for the `simulated_crash` run.log event, without the envelope keys."""
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "microbatch_index": self.microbatch_index,
            "ledger_offset": self.ledger_offset,
            "last_checkpoint_id": self.last_checkpoint_id,
        }


def log_crash_event(
    run_log: RunLogWriter,
    state: DataLoaderState,
    *,
    last_checkpoint_id: str | None,
) -> CrashEvent:
    """Append the crash record to run.log and return it."""
    event = CrashEvent(
        event_type=CRASH_EVENT_TYPE,
        run_id=state.run_id,
        branch_id=state.branch_id,
        global_step=state.next_global_step,
        microbatch_index=state.next_microbatch_index,
        ledger_offset=state.ledger_offset,
        last_checkpoint_id=last_checkpoint_id,
    )
    run_log.emit(CRASH_EVENT_TYPE, **event.to_log_fields())
    return event


def crash_from_state(state: DataLoaderState) -> SimulatedCrash:
    """Build the abort exception for a dataloader position."""
    return SimulatedCrash(
        run_id=state.run_id,
        branch_id=state.branch_id,
        global_step=state.next_global_step,
        microbatch_index=state.next_microbatch_index,
        ledger_offset=state.ledger_offset,
    )
