"""Training loop wiring plan -> gates -> batch -> training step -> ledger (P7-T04, P7-T05).

Per microbatch the order is fixed and must stay fixed:

1. `LedgerBoundDataLoader` yields the planned samples for the next microbatch.
2. The samples are tokenized, packed, and masked into a hashed `Batch`.
3. The eval firewall runs, then OPUS. Both can stop the microbatch **before** any
   gradient is computed, which is the whole point of gating before loss assignment.
4. Only a committed microbatch reaches forward/backward.
5. Only a trained microbatch appends a consumption ledger row.

Blocked and rejected microbatches advance the plan cursor but not the ledger offset,
so `ledger_offset` always counts exactly the batches the model actually learned from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from batch import AssembledBatch, assemble_microbatch
from checkpoint import build_checkpoint_payload, save_checkpoint
from config.schemas import CurriculumConfig, DemoConfig
from firewall import assert_no_eval_loss, candidate_from_planned_samples
from firewall.log import log_firewall_rejection
from firewall.types import EvalRegistry
from metrics.timing import TIMINGS_FILENAME, StepClock, StepTiming, append_step_timing
from ledger import (
    LEARNING_LEDGER_FILENAME,
    LEDGER_FILENAME,
    LearningLedgerEvent,
    LedgerBoundDataLoader,
    LedgerWriter,
    MicrobatchPlan,
    append_learning_events,
    build_learning_events,
    commit_batch,
)
from opus import BatchPipelineResult, run_batch_gate
from opus.types import AUDIT_FILENAME
from recovery.crash import CrashPolicy, crash_from_state, log_crash_event
from runlog import RunLogWriter
from schedule.types import CompiledSchedule, PlannedSample, RunPlan, StepSchedule
from tokenizer.frozen import FrozenTokenizer

from .errors import TrainerError
from .model import TinyModelConfig, build_model
from .step import MicrobatchResult, StepResult, TinyTrainer

MicrobatchStatus = Literal[
    "committed",
    "firewall_blocked",
    "opus_rejected",
    "opus_deferred",
]


@dataclass(frozen=True)
class TrainingPaths:
    """Artifact paths the training loop writes to."""

    ledger_path: Path
    learning_path: Path
    opus_audit_path: Path
    run_log_path: Path
    checkpoints_dir: Path
    reports_dir: Path
    timings_path: Path

    @classmethod
    def under(cls, artifacts_dir: Path) -> TrainingPaths:
        """Standard submission_artifacts/ layout (SCOPE.md §9)."""
        root = Path(artifacts_dir).resolve()
        return cls(
            ledger_path=root / "ledgers" / LEDGER_FILENAME,
            learning_path=root / "ledgers" / LEARNING_LEDGER_FILENAME,
            opus_audit_path=root / "ledgers" / AUDIT_FILENAME,
            run_log_path=root / "run.log",
            checkpoints_dir=root / "checkpoints",
            reports_dir=root / "reports",
            timings_path=root / "reports" / TIMINGS_FILENAME,
        )


@dataclass(frozen=True)
class TrainingContext:
    """Everything the loop needs that is produced by P1–P6."""

    demo: DemoConfig
    curriculum: CurriculumConfig
    schedule: CompiledSchedule
    run_plan: RunPlan
    tokenizer: FrozenTokenizer
    documents_by_id: dict[str, dict[str, Any]]
    registry: EvalRegistry


@dataclass(frozen=True)
class MicrobatchOutcome:
    """What happened to one planned microbatch, committed or not."""

    global_step: int
    microbatch_index: int
    candidate_id: str
    status: MicrobatchStatus
    sample_ids: tuple[str, ...]
    capability_lane: str
    curriculum_stage: str
    ledger_offset: int | None = None
    batch_content_hash: str | None = None
    assembled: AssembledBatch | None = None
    training: MicrobatchResult | None = None
    learning: tuple[LearningLedgerEvent, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSummary:
    """Result of one contiguous training segment."""

    run_id: str
    branch_id: str
    start_step: int
    stop_at_step: int
    steps: tuple[StepResult, ...]
    outcomes: tuple[MicrobatchOutcome, ...]
    checkpoint_steps: tuple[int, ...]
    final_ledger_offset: int

    @property
    def committed_microbatches(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "committed")


class TrainingRunner:
    """Runs training steps against the ledger-bound dataloader."""

    def __init__(
        self,
        context: TrainingContext,
        paths: TrainingPaths,
        *,
        trainer: TinyTrainer,
        dataloader: LedgerBoundDataLoader,
        crash_policy: CrashPolicy | None = None,
        writer: LedgerWriter | None = None,
        run_log: RunLogWriter | None = None,
    ) -> None:
        self.context = context
        self.paths = paths
        self.trainer = trainer
        self.dataloader = dataloader
        self.crash_policy = crash_policy
        # One writer per log file, or `seq` numbers collide (see runlog/writer.py). The
        # demo passes its own writer in; a standalone runner opens the log itself.
        self.run_log = run_log or RunLogWriter.open(paths.run_log_path)
        # Resume and fork inject a writer positioned for their lineage; a fresh run
        # starts a new ledger at offset 0.
        if writer is None:
            writer = LedgerWriter.open(paths.ledger_path)
            if writer.next_offset > 0 and dataloader.state().next_global_step == 0:
                # The ledger already holds a run but the plan cursor is at step 0, so
                # this runner would re-train from the beginning while appending to the
                # existing history: the same steps twice, silently. Continuing a run is
                # what resume_from_checkpoint and fork_from_checkpoint are for.
                raise TrainerError(
                    f"{paths.ledger_path} already holds "
                    f"{writer.next_offset} committed batches but the dataloader is at "
                    "step 0; use resume_from_checkpoint or fork_from_checkpoint to "
                    "continue a run"
                )
        self.writer = writer
        self._schedule_by_step: dict[int, StepSchedule] = {
            step.step: step for step in context.schedule.steps
        }
        self._last_checkpoint_id: str | None = None
        # Only a stage *change* within this segment is a transition. A resumed runner
        # starts with no previous stage, and logging its first step as a transition
        # would invent a curriculum change the schedule never contained.
        self._last_stage: str | None = None

    def run(self, *, stop_at_step: int) -> RunSummary:
        """Train until `stop_at_step` (exclusive), committing and checkpointing as it goes."""
        demo = self.context.demo
        start_step = self.dataloader.state().next_global_step
        if stop_at_step > demo.training.total_steps:
            raise TrainerError(
                f"stop_at_step {stop_at_step} exceeds total_steps {demo.training.total_steps}"
            )

        steps: list[StepResult] = []
        outcomes: list[MicrobatchOutcome] = []
        checkpoint_steps: list[int] = []

        clock = StepClock()
        while self.dataloader.state().next_global_step < stop_at_step:
            global_step = self.dataloader.state().next_global_step
            self._maybe_log_stage_transition(global_step)
            # The clock spans gating too: a microbatch the firewall or OPUS discarded
            # still cost wall time, and hiding that would overstate throughput.
            clock.start()
            # Consume microbatches until the cursor rolls to the next step rather than
            # a fixed count, so a mid-step restore finishes its partial step correctly.
            while self.dataloader.state().next_global_step == global_step:
                outcomes.append(self._run_microbatch())

            step_schedule = self._schedule_by_step.get(global_step)
            lr_multiplier = step_schedule.lr_multiplier if step_schedule else None
            steps.append(self.trainer.finish_step(global_step, lr_multiplier=lr_multiplier))
            self._record_timing(global_step, clock.stop())

            saved = self._maybe_checkpoint()
            if saved is not None:
                checkpoint_steps.append(saved)

        return RunSummary(
            run_id=demo.run.run_id,
            branch_id=self.dataloader.branch_id,
            start_step=start_step,
            stop_at_step=stop_at_step,
            steps=tuple(steps),
            outcomes=tuple(outcomes),
            checkpoint_steps=tuple(checkpoint_steps),
            final_ledger_offset=self.dataloader.ledger_offset,
        )

    def _run_microbatch(self) -> MicrobatchOutcome:
        demo = self.context.demo
        self._maybe_crash()
        plan = self.dataloader.next_microbatch()
        sample_ids = tuple(sample.sample_id for sample in plan.samples)
        shard_ids = tuple(sample.shard_id for sample in plan.samples)
        primary = plan.samples[0]

        assembled = assemble_microbatch(
            sample_ids,
            shard_ids,
            documents_by_id=self.context.documents_by_id,
            tokenizer=self.context.tokenizer,
            seq_len=demo.training.seq_len,
        )
        candidate = candidate_from_planned_samples(
            candidate_id=plan.candidate_id,
            global_step=plan.global_step,
            sample_ids=sample_ids,
            shard_ids=shard_ids,
            documents_by_id=self.context.documents_by_id,
            batch=assembled.batch,
        )

        pipeline_result = run_batch_gate(
            candidate,
            registry=self.context.registry,
            run_id=demo.run.run_id,
            branch_id=self.dataloader.branch_id,
            seed=demo.run.seed,
            curriculum_stage=plan.curriculum_stage,
            capability_lane=primary.capability_lane,
            path=primary.path,  # type: ignore[arg-type]
            opus_config=demo.opus,
            protected_floor_lanes=self.context.curriculum.protected_floors.lanes,
            audit_path=self.paths.opus_audit_path,
            documents_by_id=self.context.documents_by_id,
        )
        self._log_opus_decision(pipeline_result)

        if not pipeline_result.committed:
            return self._skip_microbatch(plan, pipeline_result, sample_ids, shard_ids, primary)

        # Last line of defense before any gradient exists: the gates above match on
        # sample IDs and hashes, this reads the assembled loss mask itself. A
        # never-train document with loss_mask=1 here means an upstream gate leaked.
        assert_no_eval_loss(candidate, self.context.registry)

        training = self.trainer.train_microbatch(
            assembled.batch,
            global_step=plan.global_step,
            microbatch_index=plan.microbatch_index,
        )
        event = commit_batch(
            self.writer,
            pipeline_result,
            assembled.batch,
            run_id=demo.run.run_id,
            branch_id=self.dataloader.branch_id,
            global_step=plan.global_step,
            curriculum_stage=plan.curriculum_stage,
            mixture_lane=primary.capability_lane,
            tokenizer_hash=self.context.tokenizer.tokenizer_hash,
            microbatch_index=plan.microbatch_index,
            checkpoint_id=self._last_checkpoint_id,
        )

        # The learning ledger is written after forward/backward, from the consumption
        # event that was just committed: a loss can only be recorded against a batch the
        # ledger already says the model consumed (P8-T02, P8-T04).
        learning = append_learning_events(
            self.paths.learning_path,
            build_learning_events(
                training.per_document_loss,
                event,
                lanes_by_sample={
                    sample.sample_id: sample.capability_lane for sample in plan.samples
                },
                shards_by_sample=dict(zip(sample_ids, shard_ids)),
                opus_score=pipeline_result.opus.opus_score if pipeline_result.opus else None,
                total_steps=demo.training.total_steps,
            ),
        )
        self.dataloader.advance_after_commit()

        self.run_log.emit(
            "batch_committed",
            run_id=demo.run.run_id,
            branch_id=self.dataloader.branch_id,
            global_step=event.global_step,
            attempt=event.attempt,
            ledger_offset=event.ledger_offset,
            microbatch_id=event.microbatch_id,
            batch_content_hash=event.batch_content_hash,
            loss_mask_hash=event.loss_mask_hash,
            curriculum_stage=event.curriculum_stage,
            mixture_lane=event.mixture_lane,
            opus_decision_id=event.opus_decision_id,
        )

        return MicrobatchOutcome(
            global_step=plan.global_step,
            microbatch_index=plan.microbatch_index,
            candidate_id=plan.candidate_id,
            status="committed",
            sample_ids=sample_ids,
            capability_lane=primary.capability_lane,
            curriculum_stage=plan.curriculum_stage,
            ledger_offset=event.ledger_offset,
            batch_content_hash=event.batch_content_hash,
            assembled=assembled,
            training=training,
            learning=learning,
        )

    def _skip_microbatch(
        self,
        plan: MicrobatchPlan,
        pipeline_result: BatchPipelineResult,
        sample_ids: tuple[str, ...],
        shard_ids: tuple[str, ...],
        primary: PlannedSample,
    ) -> MicrobatchOutcome:
        """Record a blocked or rejected microbatch, then move the plan cursor."""
        if pipeline_result.firewall.decision == "blocked":
            status = "firewall_blocked"
            reasons = pipeline_result.firewall.reasons
            log_firewall_rejection(
                self.run_log,
                pipeline_result.firewall,
                run_id=self.context.demo.run.run_id,
                branch_id=self.dataloader.branch_id,
                shard_ids=shard_ids,
                sample_ids=sample_ids,
            )
        else:
            decision = pipeline_result.opus.decision
            status = "opus_deferred" if decision == "deferred" else "opus_rejected"
            reason = pipeline_result.opus.audit.rejection_reason
            reasons = (reason,) if reason else ()

        self.dataloader.advance_after_skip()
        return MicrobatchOutcome(
            global_step=plan.global_step,
            microbatch_index=plan.microbatch_index,
            candidate_id=plan.candidate_id,
            status=status,
            sample_ids=sample_ids,
            capability_lane=primary.capability_lane,
            curriculum_stage=plan.curriculum_stage,
            reasons=reasons,
        )

    def _log_opus_decision(self, pipeline_result: BatchPipelineResult) -> None:
        """Log every OPUS verdict, including the ones that stopped the microbatch.

        A candidate the firewall already blocked never reaches OPUS, so there is nothing
        to log for it beyond the `firewall_block` event.
        """
        opus = pipeline_result.opus
        if opus is None:
            return
        audit = opus.audit
        self.run_log.emit(
            "opus_decision",
            run_id=audit.run_id,
            branch_id=audit.branch_id,
            global_step=audit.global_step,
            candidate_id=audit.candidate_id,
            opus_decision_id=audit.opus_decision_id,
            decision=audit.decision,
            opus_score=audit.opus_score,
            protected_floor_override=audit.protected_floor_override,
            opus_bypassed=audit.opus_bypassed,
            rejection_reason=audit.rejection_reason,
            capability_lane=audit.capability_lane,
            curriculum_stage=audit.curriculum_stage,
            committed=opus.committed,
        )

    def _maybe_log_stage_transition(self, global_step: int) -> None:
        """Log the step where the curriculum moves from one stage to the next."""
        step_schedule = self._schedule_by_step.get(global_step)
        if step_schedule is None:
            return
        stage = step_schedule.phase
        if self._last_stage is not None and stage != self._last_stage:
            self.run_log.emit(
                "stage_transition",
                run_id=self.context.demo.run.run_id,
                branch_id=self.dataloader.branch_id,
                global_step=global_step,
                from_stage=self._last_stage,
                to_stage=stage,
                lr_multiplier=step_schedule.lr_multiplier,
            )
        self._last_stage = stage

    def _record_timing(self, global_step: int, wall_seconds: float) -> None:
        """Append this step's wall time (P10-T03).

        Written per step rather than at the end so a crashed run still leaves timings for
        every step it finished.
        """
        append_step_timing(
            self.paths.timings_path,
            StepTiming(
                run_id=self.context.demo.run.run_id,
                branch_id=self.dataloader.branch_id,
                attempt=self.writer.attempt,
                global_step=global_step,
                wall_seconds=wall_seconds,
            ),
        )

    def _maybe_crash(self) -> None:
        """Abort the run at the configured crash point (P9-T01).

        The check runs at a microbatch boundary, before any work for that microbatch, so
        the crash never leaves a half-written batch: the ledger tail is a whole number of
        committed microbatches, and the step it lands in is genuinely incomplete.
        """
        if self.crash_policy is None:
            return
        state = self.dataloader.state()
        if not self.crash_policy.should_crash(state):
            return
        log_crash_event(
            self.run_log,
            state,
            last_checkpoint_id=self._last_checkpoint_id,
        )
        raise crash_from_state(state)

    def _maybe_checkpoint(self) -> int | None:
        """Save a checkpoint when the next step lands on the configured interval (P7-T05)."""
        state = self.dataloader.state()
        interval = self.context.demo.training.checkpoint_interval
        if state.next_global_step == 0 or state.next_global_step % interval != 0:
            return None

        model_state, optimizer_state = self.trainer.state_dicts()
        payload = build_checkpoint_payload(
            run_id=self.context.demo.run.run_id,
            branch_id=self.dataloader.branch_id,
            # The checkpoint is named for the step training resumes at, so
            # resume_from_checkpoint_step in demo.yaml reads as "restart at step N".
            global_step=state.next_global_step,
            seed=self.context.demo.run.seed,
            dataloader_state=state,
            model_state=model_state,
            optimizer_state=optimizer_state,
        )
        save_checkpoint(self.paths.checkpoints_dir, payload)
        self._last_checkpoint_id = payload.checkpoint_id
        self.run_log.emit(
            "checkpoint_saved",
            run_id=payload.run_id,
            branch_id=payload.branch_id,
            checkpoint_id=payload.checkpoint_id,
            global_step=payload.next_global_step,
            ledger_offset=payload.ledger_offset,
        )
        return state.next_global_step


def build_training_runner(
    context: TrainingContext,
    paths: TrainingPaths,
    *,
    device: torch.device | None = None,
    crash_policy: CrashPolicy | None = None,
    writer: LedgerWriter | None = None,
    run_log: RunLogWriter | None = None,
) -> TrainingRunner:
    """Wire model, optimizer, and ledger-bound dataloader from loaded configs."""
    demo = context.demo
    model_config = TinyModelConfig.from_demo_config(
        demo.model,
        vocab_size=context.tokenizer.vocab_size,
        max_seq_len=demo.training.seq_len,
    )
    model = build_model(model_config, seed=demo.run.seed, device=device)
    trainer = TinyTrainer(
        model,
        demo.optimizer,
        gradient_accumulation_steps=demo.training.gradient_accumulation_steps,
        device=device,
    )
    dataloader = LedgerBoundDataLoader(
        context.run_plan,
        microbatch_size=demo.training.microbatch_size,
        global_batch_size=demo.training.global_batch_size,
        run_id=demo.run.run_id,
        branch_id=context.run_plan.branch_id,
    )
    return TrainingRunner(
        context,
        paths,
        trainer=trainer,
        dataloader=dataloader,
        crash_policy=crash_policy,
        writer=writer,
        run_log=run_log,
    )
