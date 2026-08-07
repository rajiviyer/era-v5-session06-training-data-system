"""One-command demo orchestrator (P11-T02, P11-T03).

Runs the phases the assignment grades, in order, against a clean `submission_artifacts/`
directory. Every phase either produces an artifact or checks one that a previous phase
produced; none of them assert a result they computed themselves.

**Phases implemented so far**

1. Build tokenized shards, manifests, and the admission-gated registry
2. Compile the mixture schedule
3. Train until the configured crash step, then crash for real
4. Resume from the checkpoint behind the ledger tail and train to the end
5. Verify the resumed batches against the pre-crash record
6. Replay a historical step range and prove it reconstructs
7. Fork a new branch from an earlier checkpoint and show the streams separate
8. Recompute packing utilization and throughput from the artifacts
9. Report firewall and OPUS gate activity from the audit trail
10. Audit the structured `run.log` against the event vocabulary SCOPE.md §9.1 requires
11. Collect the evidence bundle: `evidence.json` and `evidence.md`

The last two phases are audits of the run rather than parts of it. They read the finished
artifacts back off disk, which is the only way a demo can claim its own output is correct
without asserting it.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.loader import load_configs
from config.schemas import CurriculumConfig, DemoConfig
from corpus import load_corpus
from evidence import (
    REQUIREMENT_KEYS,
    collect_evidence,
    write_evidence_json,
    write_evidence_markdown,
)
from firewall import REGISTRY_FILENAME as EVAL_REGISTRY_FILENAME, build_eval_registry
from firewall.registry import write_eval_registry
from ledger import load_consumption_ledger, load_learning_ledger, verify_learning_links
from ledger.reader import events_for_attempt
from metrics import (
    compute_packing_utilization,
    compute_throughput,
    load_step_timings,
    write_packing_report,
    write_throughput_report,
)
from opus import load_opus_audit
from recovery import CrashPolicy, ResumedRun, SimulatedCrash, resume_from_checkpoint
from recovery.fork import fork_from_checkpoint, verify_fork, write_fork_verification
from recovery.replay import replay_range, write_replay_verification
from recovery.verify import verify_resume, write_resume_verification
from runlog import (
    RunLogWriter,
    event_type_counts,
    events_of_type,
    load_run_log,
    missing_event_types,
)
from schedule import build_sample_pool, compile_schedule, plan_run, write_schedule_json
from schedule.pool import SampleCandidate
from shards.admission import evaluate_admission
from shards.pipeline import build_shards_with_manifests
from tokenizer.frozen import FrozenTokenizer
from tokenizer.manifest import (
    MANIFEST_FILENAME as TOKENIZER_MANIFEST_FILENAME,
    write_tokenizer_manifest,
)
from trainer import TrainingContext, TrainingPaths, build_training_runner

PENDING_PHASES: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of one demo phase."""

    number: int
    name: str
    passed: bool
    detail: str
    artifacts: tuple[Path, ...] = ()


@dataclass
class DemoResult:
    """Everything the demo produced, in phase order."""

    artifacts_dir: Path
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.phases) and all(phase.passed for phase in self.phases)

    def record(self, phase: PhaseResult) -> PhaseResult:
        self.phases.append(phase)
        return phase


@dataclass(frozen=True)
class _Wiring:
    """Loaded inputs shared by the training phases."""

    demo: DemoConfig
    curriculum: CurriculumConfig
    tokenizer: FrozenTokenizer
    documents: list[dict[str, Any]]
    paths: TrainingPaths
    # One writer for the whole run: the trainer, the recovery phases, and this module all
    # append through it, so `seq` stays a single ordered stream (see runlog/writer.py).
    run_log: RunLogWriter


def run_demo(
    assignment_root: Path,
    *,
    artifacts_dir: Path | None = None,
    clean: bool = True,
) -> DemoResult:
    """Run the full demo sequence and return one result per phase."""
    root = Path(assignment_root).resolve()
    demo, curriculum = load_configs(root)
    output = (artifacts_dir or demo.paths.submission_artifacts).resolve()

    if clean:
        _clean_artifacts_dir(output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = FrozenTokenizer.load_default(root)
    _, documents = load_corpus(demo.paths.toy_corpus)
    paths = TrainingPaths.under(output)
    wiring = _Wiring(
        demo=demo,
        curriculum=curriculum,
        tokenizer=tokenizer,
        documents=documents,
        paths=paths,
        run_log=RunLogWriter.open(paths.run_log_path),
    )

    result = DemoResult(artifacts_dir=output)
    _log_run_start(wiring, root)

    shards = result.record(_phase_build_shards(wiring, output))
    if not shards.passed:
        return _finish(wiring, result)

    schedule_phase = result.record(_phase_compile_schedule(wiring, output))
    if not schedule_phase.passed:
        return _finish(wiring, result)

    context, pool = _build_context(wiring, output)

    crashed = result.record(_phase_train_to_crash(wiring, context))
    if not crashed.passed:
        return _finish(wiring, result)

    resumed_run, resume_phase = _phase_resume(wiring, context)
    result.record(resume_phase)
    if not resume_phase.passed or resumed_run is None:
        return _finish(wiring, result)

    result.record(_phase_verify_resume(wiring, resumed_run))
    result.record(_phase_replay(wiring, context))
    result.record(_phase_fork(wiring, context, pool))
    result.record(_phase_metrics(wiring, context))
    result.record(_phase_gate_activity(wiring))
    return _finish(wiring, result)


def _finish(wiring: _Wiring, result: DemoResult) -> DemoResult:
    """Close the log with `run_complete`, then audit what the run actually logged.

    `run_complete` is written before the audit so that it is genuinely the last line of
    the file and the audit can see it. The audit reads the finished `run.log` back from
    disk: whether the required events were emitted is a claim about the artifact, not
    about the code that was supposed to emit them.
    """
    wiring.run_log.emit(
        "run_complete",
        run_id=wiring.demo.run.run_id,
        branch_id=wiring.demo.run.branch_id,
        phases_run=len(result.phases),
        phases_passed=sum(1 for phase in result.phases if phase.passed),
        failed_phases=[phase.name for phase in result.phases if not phase.passed],
    )
    result.record(_phase_audit_run_log(wiring))
    result.record(_phase_evidence(wiring))
    return result


def _log_run_start(wiring: _Wiring, assignment_root: Path) -> None:
    """Open the log with what this run is and what it was configured from.

    The config hashes are of the YAML files as they were read, so a grader can tell
    whether two runs of the demo were driven by the same inputs without diffing them.
    """
    demo = wiring.demo
    wiring.run_log.emit(
        "run_start",
        run_id=demo.run.run_id,
        branch_id=demo.run.branch_id,
        seed=demo.run.seed,
        total_steps=demo.training.total_steps,
        seq_len=demo.training.seq_len,
        tokenizer_hash=wiring.tokenizer.tokenizer_hash,
        demo_config_hash=_file_hash(assignment_root / "configs" / "demo.yaml"),
        curriculum_config_hash=_file_hash(demo.paths.curriculum_config),
        corpus_documents=len(wiring.documents),
    )


def _file_hash(path: Path) -> str:
    """`sha256:` fingerprint of a config file's bytes."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _phase_build_shards(wiring: _Wiring, output: Path) -> PhaseResult:
    """Phase 1: immutable shards, manifests, admission gate, registry."""
    built = build_shards_with_manifests(
        wiring.documents,
        tokenizer=wiring.tokenizer,
        shards_dir=output / "shards",
        manifests_dir=output / "manifests",
    )
    # The tokenizer manifest travels with the shards it sealed (SCOPE.md §9): a shard's
    # tokenizer_hash is meaningless without the manifest that says what that hash covers.
    write_tokenizer_manifest(
        wiring.tokenizer.artifact_path,
        manifest_path=output / "manifests" / TOKENIZER_MANIFEST_FILENAME,
    )
    admitted = built.registry.get("admitted_shard_ids", [])
    blocked = [
        manifest["shard_id"]
        for manifest in built.manifests
        if manifest.get("admission") != "admitted"
    ]

    # The manifest is where the gate's decision is recorded; the log says when the run
    # saw it. Both are read from the manifests the builder just wrote, never asserted.
    for manifest in built.manifests:
        admission, reasons = evaluate_admission(manifest)
        wiring.run_log.emit(
            "shard_admitted" if admission == "admitted" else "shard_blocked",
            shard_id=manifest["shard_id"],
            capability_lane=manifest.get("capability_lane"),
            token_count=manifest.get("token_count"),
            content_hash=manifest.get("content_hash"),
            tokenizer_hash=manifest.get("tokenizer_hash"),
            admission=admission,
            reasons=reasons,
        )

    return PhaseResult(
        number=1,
        name="Build shards and manifests",
        passed=bool(admitted),
        detail=(
            f"{len(built.shards)} shards built, {len(admitted)} admitted, "
            f"{len(blocked)} blocked by the admission gate"
        ),
        artifacts=(output / "shards", output / "manifests"),
    )


def _phase_compile_schedule(wiring: _Wiring, output: Path) -> PhaseResult:
    """Phase 2: curriculum.yaml to per-step lane quotas."""
    schedule = compile_schedule(
        wiring.curriculum,
        total_steps=wiring.demo.training.total_steps,
    )
    target = output / "schedule.json"
    write_schedule_json(target, schedule)
    stages = sorted({step.phase for step in schedule.steps})
    return PhaseResult(
        number=2,
        name="Compile mixture schedule",
        passed=target.is_file() and bool(schedule.steps),
        detail=(
            f"{schedule.total_steps} steps across stages {stages}; "
            f"{len(schedule.warnings)} supply warnings"
        ),
        artifacts=(target,),
    )


def _phase_train_to_crash(wiring: _Wiring, context: TrainingContext) -> PhaseResult:
    """Phase 3: train, checkpoint on interval, then crash at the configured step."""
    recovery = wiring.demo.recovery
    runner = build_training_runner(
        context,
        wiring.paths,
        crash_policy=CrashPolicy.from_config(recovery),
        run_log=wiring.run_log,
    )
    try:
        runner.run(stop_at_step=wiring.demo.training.total_steps)
    except SimulatedCrash as crash:
        ledger = load_consumption_ledger(wiring.paths.ledger_path)
        return PhaseResult(
            number=3,
            name="Train, checkpoint, and crash",
            passed=bool(ledger),
            detail=(
                f"crashed at step {crash.global_step} microbatch "
                f"{crash.microbatch_index}; {len(ledger)} committed batches, "
                f"ledger_offset {crash.ledger_offset}"
            ),
            artifacts=(wiring.paths.ledger_path, wiring.paths.checkpoints_dir),
        )

    return PhaseResult(
        number=3,
        name="Train, checkpoint, and crash",
        passed=False,
        detail=(
            f"run reached total_steps without crashing; crash_at_step "
            f"{recovery.crash_at_step} was never hit"
        ),
    )


def _phase_resume(
    wiring: _Wiring,
    context: TrainingContext,
) -> tuple[ResumedRun | None, PhaseResult]:
    """Phase 4: restore the checkpoint behind the ledger tail and finish the run."""
    recovery = wiring.demo.recovery
    resumed = resume_from_checkpoint(
        context,
        wiring.paths,
        checkpoint_step=recovery.resume_from_checkpoint_step,
        run_log=wiring.run_log,
    )
    summary = resumed.runner.run(stop_at_step=wiring.demo.training.total_steps)
    ledger = load_consumption_ledger(wiring.paths.ledger_path)
    return resumed, PhaseResult(
        number=4,
        name="Resume from checkpoint",
        passed=summary.committed_microbatches > 0,
        detail=(
            f"resumed at ckpt-{resumed.resume_step:05d} from ledger_offset "
            f"{resumed.resume_ledger_offset}, attempt {resumed.prior_attempt} to "
            f"{resumed.resumed_attempt}; "
            f"{len(events_for_attempt(ledger, resumed.prior_attempt))} rows retained, "
            f"{len(events_for_attempt(ledger, resumed.resumed_attempt))} rows re-committed"
        ),
        artifacts=(wiring.paths.ledger_path,),
    )


def _phase_verify_resume(wiring: _Wiring, resumed: ResumedRun) -> PhaseResult:
    """Phase 5: prove the resumed batches match the pre-crash record."""
    verification = verify_resume(
        wiring.paths.ledger_path,
        resume_ledger_offset=resumed.resume_ledger_offset,
        prior_attempt=resumed.prior_attempt,
        resumed_attempt=resumed.resumed_attempt,
        resumed_from_checkpoint_step=resumed.resume_step,
    )
    report = write_resume_verification(wiring.paths.reports_dir, verification)
    _log_verification(
        wiring,
        "resume",
        passed=verification.passed,
        report=report,
        compared=len(verification.comparisons),
        matched=len(verification.comparisons) - len(verification.mismatched),
        hashes=[
            {
                "microbatch_id": item.microbatch_id,
                "expected_batch_content_hash": item.expected_batch_content_hash,
                "actual_batch_content_hash": item.actual_batch_content_hash,
                "matched": item.matched,
            }
            for item in verification.comparisons
        ],
        skipped_batches=list(verification.skipped_batches),
        repeated_batches=list(verification.repeated_batches),
    )
    return PhaseResult(
        number=5,
        name="Verify resume (no skip, no repeat, hashes match)",
        passed=verification.passed,
        detail=(
            f"{len(verification.comparisons)} batches compared, "
            f"{len(verification.comparisons) - len(verification.mismatched)} matched, "
            f"{len(verification.skipped_batches)} skipped, "
            f"{len(verification.repeated_batches)} repeated"
        ),
        artifacts=(report,),
    )


def _phase_replay(wiring: _Wiring, context: TrainingContext) -> PhaseResult:
    """Phase 6: re-derive a historical range and prove it reconstructs (P9-T05–T07)."""
    recovery = wiring.demo.recovery
    verification = replay_range(
        context,
        wiring.paths,
        start_step=recovery.replay_start_step,
        end_step=recovery.replay_end_step,
        run_log=wiring.run_log,
    )
    report = write_replay_verification(wiring.paths.reports_dir, verification)
    _log_verification(
        wiring,
        "replay",
        passed=verification.passed,
        report=report,
        compared=len(verification.comparisons),
        matched=len(verification.comparisons) - len(verification.mismatched),
        hashes=[
            {
                "microbatch_id": item.microbatch_id,
                "expected_batch_content_hash": item.recorded_batch_content_hash,
                "actual_batch_content_hash": item.recomputed_batch_content_hash,
                "matched": item.matched,
            }
            for item in verification.comparisons
        ],
        start_step=verification.start_step,
        end_step=verification.end_step,
    )
    return PhaseResult(
        number=6,
        name="Replay historical range",
        passed=verification.passed,
        detail=(
            f"steps {verification.start_step}..{verification.end_step}: "
            f"{len(verification.comparisons)} batches replayed, "
            f"{len(verification.comparisons) - len(verification.mismatched)} matched "
            "on planner, hashes, and token spans"
        ),
        artifacts=(report,),
    )


def _phase_fork(
    wiring: _Wiring,
    context: TrainingContext,
    pool: tuple[SampleCandidate, ...],
) -> PhaseResult:
    """Phase 7: branch from an earlier checkpoint and show the streams separate."""
    recovery = wiring.demo.recovery
    forked = fork_from_checkpoint(
        context,
        wiring.paths,
        checkpoint_step=recovery.fork_from_checkpoint_step,
        new_branch_id=recovery.fork_branch_id,
        pool=pool,
        run_log=wiring.run_log,
    )
    # Enough steps past the fork point for the streams to visibly separate.
    forked.runner.run(stop_at_step=min(
        recovery.fork_from_checkpoint_step + 5,
        wiring.demo.training.total_steps,
    ))

    verification = verify_fork(wiring.paths, forked)
    report = write_fork_verification(wiring.paths.reports_dir, verification)
    # A fork has no expected-versus-actual hashes to report: the point is that the two
    # streams *differ*, so what the log carries is where they first diverged.
    _log_verification(
        wiring,
        "fork",
        passed=verification.passed,
        report=report,
        compared=len(verification.compared_steps),
        parent_branch_id=verification.parent_branch_id,
        child_branch_id=verification.child_branch_id,
        divergence_step=verification.divergence_step,
        diverged_steps=list(verification.diverged_steps),
    )
    return PhaseResult(
        number=7,
        name="Fork from earlier checkpoint",
        passed=verification.passed,
        detail=(
            f"branch {verification.child_branch_id} from "
            f"ckpt-{verification.forked_from_step:05d} (parent offset "
            f"{verification.parent_ledger_offset}); {verification.child_batches} batches, "
            f"diverges at step {verification.divergence_step}"
        ),
        artifacts=(report, forked.paths.ledger_path),
    )


def _phase_metrics(wiring: _Wiring, context: TrainingContext) -> PhaseResult:
    """Phase 8: packing utilization and throughput, recomputed from the artifacts (P10)."""
    ledger = load_consumption_ledger(wiring.paths.ledger_path)
    packing = compute_packing_utilization(
        ledger,
        documents_by_id=context.documents_by_id,
        tokenizer=wiring.tokenizer,
        seq_len=wiring.demo.training.seq_len,
    )
    packing_report = write_packing_report(wiring.paths.reports_dir, packing)

    timings = load_step_timings(wiring.paths.timings_path)
    throughput = compute_throughput(packing.batches, timings)
    throughput_report = write_throughput_report(wiring.paths.reports_dir, throughput)

    return PhaseResult(
        number=8,
        name="Packing utilization and throughput",
        passed=bool(packing.batches) and bool(throughput.steps),
        detail=(
            f"utilization {packing.utilization:.3f} over {len(packing.batches)} batches "
            f"({dict((name, round(stats['utilization'], 3)) for name, stats in packing.by_policy().items())}); "
            f"{throughput.loss_bearing_tokens_per_second:.0f} loss-bearing tok/s, "
            f"{throughput.raw_tokens_per_second:.0f} raw tok/s over "
            f"{throughput.total_wall_seconds:.2f} s"
        ),
        artifacts=(packing_report, throughput_report),
    )


def _phase_gate_activity(wiring: _Wiring) -> PhaseResult:
    """Phase 9: report what the firewall and OPUS actually did during the run.

    Informational: the pass criterion is only that the audit trail exists and links up.
    Whether every OPUS decision type appeared is a grading claim, and that belongs to the
    evidence collector (P11-T05), not to the script that produced the run.
    """
    audit = load_opus_audit(wiring.paths.opus_audit_path)
    decisions = Counter(record.decision for record in audit)
    overrides = sum(1 for record in audit if record.protected_floor_override)

    learning = load_learning_ledger(wiring.paths.learning_path)
    consumption = load_consumption_ledger(wiring.paths.ledger_path)
    links = verify_learning_links(learning, consumption)

    firewall_blocks = len(
        events_of_type(load_run_log(wiring.paths.run_log_path), "firewall_block")
    )

    return PhaseResult(
        number=9,
        name="Gate activity and ledger links",
        passed=bool(audit) and links.linked,
        detail=(
            f"OPUS {dict(sorted(decisions.items()))}, {overrides} protected-floor "
            f"overrides, {firewall_blocks} firewall blocks; "
            f"{links.learning_rows} learning rows link to "
            f"{links.committed_batches} committed batches"
        ),
        artifacts=(wiring.paths.opus_audit_path, wiring.paths.learning_path),
    )


def _log_verification(
    wiring: _Wiring,
    verification: str,
    *,
    passed: bool,
    report: Path,
    **fields: Any,
) -> None:
    """Log one verification outcome (SCOPE.md §9.1: pass/fail with expected vs actual).

    The compact hash pairs go in the log itself rather than only in the report, so the
    event stream is readable end to end without opening a second file. `report_path`
    points at the full record for anything the summary drops.
    """
    wiring.run_log.emit(
        "verification_result",
        verification=verification,
        passed=passed,
        report_path=_artifact_path(wiring, report),
        **fields,
    )


def _artifact_path(wiring: _Wiring, path: Path) -> str:
    """Path relative to `submission_artifacts/`, so the log is machine-independent."""
    artifacts_dir = wiring.paths.run_log_path.parent
    return Path(path).resolve().relative_to(artifacts_dir).as_posix()


def _phase_audit_run_log(wiring: _Wiring) -> PhaseResult:
    """Phase 10: read `run.log` back and check it against SCOPE.md §9.1 (P11-T01).

    This is the only phase whose subject is the log itself. `load_run_log` also enforces
    that sequence numbers strictly increase, which is what proves the file is a single
    ordered stream rather than two writers interleaving.
    """
    events = load_run_log(wiring.paths.run_log_path)
    missing = missing_event_types(events)
    counts = event_type_counts(events)
    return PhaseResult(
        number=10,
        name="Structured run.log covers every required event type",
        passed=not missing,
        detail=(
            f"{len(events)} events; "
            + ", ".join(f"{name} {count}" for name, count in counts.items())
            + (f"; MISSING {list(missing)}" if missing else "")
        ),
        artifacts=(wiring.paths.run_log_path,),
    )


def _phase_evidence(wiring: _Wiring) -> PhaseResult:
    """Phase 11: collect `evidence.json` and `evidence.md` (P11-T04–T06).

    Runs last because it grades everything the earlier phases wrote. If it fails, the
    demo exits non-zero even when every producing phase succeeded: the artifacts, not
    the code path, are the submission.
    """
    artifacts_dir = wiring.paths.run_log_path.parent
    bundle = collect_evidence(artifacts_dir, wiring.demo.assignment_root)
    json_path = write_evidence_json(artifacts_dir, bundle)
    md_path = write_evidence_markdown(artifacts_dir, bundle)

    passed_count = sum(
        1 for key in REQUIREMENT_KEYS if bundle.requirements[key].passed
    )
    return PhaseResult(
        number=11,
        name="Evidence bundle",
        passed=bundle.passed,
        detail=(
            f"{passed_count}/{len(REQUIREMENT_KEYS)} requirements passed"
            + (f"; failed {list(bundle.failed_keys)}" if bundle.failed_keys else "")
            + (
                f"; missing evidence paths {list(bundle.missing_evidence_paths)}"
                if bundle.missing_evidence_paths
                else ""
            )
        ),
        artifacts=(json_path, md_path),
    )


def _build_context(
    wiring: _Wiring,
    output: Path,
) -> tuple[TrainingContext, tuple[SampleCandidate, ...]]:
    """Wire the planner and registry from the artifacts phase 1 just wrote.

    The pool is returned alongside the context because forking has to re-plan on a new
    branch, and `RunPlan` is bound to the branch it was planned for.
    """
    demo = wiring.demo
    schedule = compile_schedule(wiring.curriculum, total_steps=demo.training.total_steps)
    pool = build_sample_pool(output / "manifests", wiring.documents)
    run_plan = plan_run(
        schedule.steps,
        pool,
        run_id=demo.run.run_id,
        branch_id=demo.run.branch_id,
        seed=demo.run.seed,
        global_batch_size=demo.training.global_batch_size,
    )
    registry = build_eval_registry(wiring.documents, manifests_dir=output / "manifests")
    # The firewall's never-train list is part of the submission (SCOPE.md §9): a grader
    # checking "no eval token carried loss" needs to see which documents that covered.
    write_eval_registry(output / EVAL_REGISTRY_FILENAME, registry)

    context = TrainingContext(
        demo=demo,
        curriculum=wiring.curriculum,
        schedule=schedule,
        run_plan=run_plan,
        tokenizer=wiring.tokenizer,
        documents_by_id={doc["document_id"]: doc for doc in wiring.documents},
        registry=registry,
    )
    return context, tuple(pool)


def _clean_artifacts_dir(output: Path) -> None:
    """Empty submission_artifacts/ so the run proves regeneration from nothing.

    `.gitkeep` survives: the directory is tracked but its contents are not.
    """
    if not output.is_dir():
        return
    for entry in output.iterdir():
        if entry.name == ".gitkeep":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
