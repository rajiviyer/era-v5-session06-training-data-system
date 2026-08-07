"""Ledger-bound dataloader state (P6-T07)."""

from __future__ import annotations

from dataclasses import dataclass

from schedule.types import PlannedSample, RunPlan, StepPlan

from .types import DataLoaderState, DATALOADER_VERSION


@dataclass(frozen=True)
class MicrobatchPlan:
    """Planned samples for one microbatch."""

    global_step: int
    microbatch_index: int
    curriculum_stage: str
    samples: tuple[PlannedSample, ...]
    candidate_id: str


class LedgerBoundDataLoader:
    """Deterministic microbatch planner bound to ledger_offset progression."""

    def __init__(
        self,
        run_plan: RunPlan,
        *,
        microbatch_size: int,
        global_batch_size: int,
        run_id: str,
        branch_id: str,
        ledger_offset: int = -1,
        next_global_step: int = 0,
        next_microbatch_index: int = 0,
    ) -> None:
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if global_batch_size % microbatch_size != 0:
            raise ValueError("global_batch_size must be divisible by microbatch_size")

        self.run_plan = run_plan
        self.microbatch_size = microbatch_size
        self.global_batch_size = global_batch_size
        self.microbatches_per_step = global_batch_size // microbatch_size
        self.run_id = run_id
        self.branch_id = branch_id
        self._ledger_offset = ledger_offset
        self._next_global_step = next_global_step
        self._next_microbatch_index = next_microbatch_index
        self._step_plans = {step.step: step for step in run_plan.steps}

    @property
    def ledger_offset(self) -> int:
        return self._ledger_offset

    @property
    def next_ledger_offset(self) -> int:
        return self._ledger_offset + 1

    def state(self) -> DataLoaderState:
        """Current dataloader position for checkpoint binding."""
        return DataLoaderState(
            run_id=self.run_id,
            branch_id=self.branch_id,
            ledger_offset=self._ledger_offset,
            next_global_step=self._next_global_step,
            next_microbatch_index=self._next_microbatch_index,
            dataloader_version=DATALOADER_VERSION,
        )

    def restore_state(self, state: DataLoaderState) -> None:
        """Restore dataloader position after checkpoint load."""
        if state.run_id != self.run_id:
            raise ValueError(f"run_id mismatch: {state.run_id} != {self.run_id}")
        if state.branch_id != self.branch_id:
            raise ValueError(f"branch_id mismatch: {state.branch_id} != {self.branch_id}")
        if state.dataloader_version != DATALOADER_VERSION:
            raise ValueError(
                f"dataloader_version mismatch: {state.dataloader_version} != {DATALOADER_VERSION}"
            )
        self._ledger_offset = state.ledger_offset
        self._next_global_step = state.next_global_step
        self._next_microbatch_index = state.next_microbatch_index

    def next_microbatch(self) -> MicrobatchPlan:
        """Return the next planned microbatch without advancing state."""
        step_plan = self._require_step_plan(self._next_global_step)
        start = self._next_microbatch_index * self.microbatch_size
        end = start + self.microbatch_size
        if end > len(step_plan.samples):
            raise ValueError(
                f"microbatch slice out of range for step {self._next_global_step}: "
                f"{start}:{end} of {len(step_plan.samples)}"
            )
        samples = step_plan.samples[start:end]
        return MicrobatchPlan(
            global_step=self._next_global_step,
            microbatch_index=self._next_microbatch_index,
            curriculum_stage=step_plan.phase,
            samples=samples,
            candidate_id=self._candidate_id(self._next_global_step, self._next_microbatch_index),
        )

    def advance_after_commit(self) -> None:
        """Advance plan cursor after one successful ledger commit."""
        self._ledger_offset += 1
        self._advance_cursor()

    def advance_after_skip(self) -> None:
        """Advance the plan cursor for a microbatch that never reached the ledger.

        Firewall blocks and OPUS rejections must not consume a `ledger_offset`: the
        offset counts committed batches only. The plan cursor still moves so the run
        does not re-offer the same rejected microbatch forever, and both values are
        checkpointed independently, so resume stays exact.
        """
        self._advance_cursor()

    def _advance_cursor(self) -> None:
        self._next_microbatch_index += 1
        if self._next_microbatch_index >= self.microbatches_per_step:
            self._next_microbatch_index = 0
            self._next_global_step += 1

    def _require_step_plan(self, global_step: int) -> StepPlan:
        step_plan = self._step_plans.get(global_step)
        if step_plan is None:
            raise ValueError(f"no plan available for global_step {global_step}")
        return step_plan

    def _candidate_id(self, global_step: int, microbatch_index: int) -> str:
        return (
            f"{self.run_id}:{self.branch_id}:step{global_step}:"
            f"mb{microbatch_index}"
        )
