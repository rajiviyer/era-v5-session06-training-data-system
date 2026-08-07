"""Tests for crash resume and resume verification (P9-T02–T04)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
_CORPUS = _ASSIGNMENT / "data" / "toy_corpus"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import build_eval_registry  # noqa: E402
from ledger import (  # noqa: E402
    LedgerWriter,
    aggregate_by_shard,
    load_consumption_ledger,
    load_learning_ledger,
    verify_learning_links,
)
from ledger.errors import LedgerError  # noqa: E402
from ledger.reader import events_for_attempt, latest_attempt  # noqa: E402
from ledger.learning_aggregate import effective_events  # noqa: E402
from recovery import (  # noqa: E402
    CrashPolicy,
    RecoveryError,
    SimulatedCrash,
    checkpoints_available,
    resume_from_checkpoint,
    verify_resume,
    write_resume_verification,
)
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402
from trainer import TrainingContext, TrainingPaths, build_training_runner  # noqa: E402

# Checkpoint interval is 10, so a crash at 14 leaves ckpt-00010 behind the ledger tail.
CRASH_STEP = 14
RESUME_STEP = 10


class TestCrashThenResume(unittest.TestCase):
    """One crashed run, one resumed run, verified from the ledger they both wrote."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, cls.curriculum = load_configs(_ASSIGNMENT)
        cls.schedule = compile_schedule(
            cls.curriculum,
            total_steps=cls.demo.training.total_steps,
        )
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, cls.documents = load_corpus(_CORPUS)
        cls.documents_by_id = {doc["document_id"]: doc for doc in cls.documents}

        cls.temp_dir = Path(tempfile.mkdtemp())
        result = build_shards_with_manifests(
            cls.documents,
            tokenizer=cls.tokenizer,
            shards_dir=cls.temp_dir / "shards",
            manifests_dir=cls.temp_dir / "manifests",
        )
        cls.registry = build_eval_registry(cls.documents, manifests_dir=result.manifests_dir)
        pool = build_sample_pool(cls.temp_dir / "manifests", cls.documents)
        cls.run_plan = plan_run(
            cls.schedule.steps,
            pool,
            run_id=cls.demo.run.run_id,
            branch_id=cls.demo.run.branch_id,
            seed=cls.demo.run.seed,
            global_batch_size=cls.demo.training.global_batch_size,
        )

        cls.paths = TrainingPaths.under(Path(tempfile.mkdtemp(dir=cls.temp_dir)))

        # Attempt 0: train until the configured crash point.
        crashed = build_training_runner(
            cls._context(),
            cls.paths,
            crash_policy=CrashPolicy(crash_at_step=CRASH_STEP),
        )
        try:
            crashed.run(stop_at_step=cls.demo.training.total_steps)
        except SimulatedCrash as crash:
            cls.crash = crash
        else:  # pragma: no cover
            raise AssertionError("run completed without crashing")

        cls.ledger_at_crash = load_consumption_ledger(cls.paths.ledger_path)

        # Attempt 1: restore ckpt-00010 and train back through the crash point.
        cls.resumed = resume_from_checkpoint(
            cls._context(),
            cls.paths,
            checkpoint_step=RESUME_STEP,
        )
        cls.resumed.runner.run(stop_at_step=CRASH_STEP)

        cls.ledger = load_consumption_ledger(cls.paths.ledger_path)
        cls.verification = verify_resume(
            cls.paths.ledger_path,
            resume_ledger_offset=cls.resumed.resume_ledger_offset,
            prior_attempt=cls.resumed.prior_attempt,
            resumed_attempt=cls.resumed.resumed_attempt,
            resumed_from_checkpoint_step=RESUME_STEP,
        )

    @classmethod
    def _context(cls) -> TrainingContext:
        return TrainingContext(
            demo=cls.demo,
            curriculum=cls.curriculum,
            schedule=cls.schedule,
            run_plan=cls.run_plan,
            tokenizer=cls.tokenizer,
            documents_by_id=cls.documents_by_id,
            registry=cls.registry,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_resume_restores_cursor_and_opens_a_new_attempt(self) -> None:
        self.assertEqual(self.resumed.resume_step, RESUME_STEP)
        self.assertEqual(self.resumed.prior_attempt, 0)
        self.assertEqual(self.resumed.resumed_attempt, 1)
        # The checkpoint sat behind the crashed tail, which is what makes this a real
        # reconciliation rather than an append.
        self.assertLess(
            self.resumed.resume_ledger_offset,
            self.ledger_at_crash[-1].ledger_offset,
        )

    def test_crashed_rows_are_retained_not_overwritten(self) -> None:
        """Append-only: the crashed attempt stays readable as the expected record."""
        prior = events_for_attempt(self.ledger, 0)
        self.assertEqual(len(prior), len(self.ledger_at_crash))
        self.assertEqual(
            [record.batch_content_hash for record in prior],
            [record.batch_content_hash for record in self.ledger_at_crash],
        )
        self.assertEqual(latest_attempt(self.ledger), 1)

    def test_resume_no_skip_no_repeat(self) -> None:
        self.assertTrue(
            self.verification.comparisons,
            "resume re-consumed nothing, so nothing was proven",
        )
        self.assertEqual(self.verification.skipped_batches, ())
        self.assertEqual(self.verification.repeated_batches, ())
        self.assertTrue(self.verification.offsets_contiguous)
        self.assertTrue(self.verification.passed)

    def test_post_resume_batches_match_pre_crash_hashes(self) -> None:
        """P9-T03: same IDs, spans, and hashes at the same ledger position."""
        self.assertEqual(self.verification.mismatched, ())
        for comparison in self.verification.comparisons:
            self.assertEqual(
                comparison.expected_batch_content_hash,
                comparison.actual_batch_content_hash,
            )
            self.assertEqual(
                comparison.expected_loss_mask_hash,
                comparison.actual_loss_mask_hash,
            )
            self.assertTrue(comparison.sample_ids_match)
            self.assertTrue(comparison.token_spans_match)
            self.assertEqual(
                comparison.expected_ledger_offset,
                comparison.actual_ledger_offset,
            )

    def test_resumed_attempt_starts_at_the_checkpoint_offset(self) -> None:
        resumed_rows = events_for_attempt(self.ledger, 1)
        self.assertEqual(
            resumed_rows[0].ledger_offset,
            self.resumed.resume_ledger_offset + 1,
        )
        self.assertEqual(
            [record.ledger_offset for record in resumed_rows],
            list(
                range(
                    self.resumed.resume_ledger_offset + 1,
                    self.resumed.resume_ledger_offset + 1 + len(resumed_rows),
                )
            ),
        )

    def test_ledger_with_two_attempts_still_loads(self) -> None:
        """The ordering rules must accept a rewind that raises the attempt number."""
        reloaded = load_consumption_ledger(self.paths.ledger_path)
        self.assertEqual(len(reloaded), len(self.ledger))
        attempts = [record.attempt for record in reloaded]
        self.assertEqual(attempts, sorted(attempts))

    def test_learning_ledger_follows_the_resumed_attempt(self) -> None:
        learning = load_learning_ledger(self.paths.learning_path)
        consumption = load_consumption_ledger(self.paths.ledger_path)

        report = verify_learning_links(learning, consumption)
        self.assertEqual(report.orphan_offsets, ())
        self.assertEqual(report.unreported_offsets, ())
        self.assertEqual(report.mismatches, ())

        # Rows exist under both attempts, and the join key kept them apart.
        self.assertEqual({event.attempt for event in learning}, {0, 1})

    def test_aggregates_exclude_rolled_back_attempts(self) -> None:
        """Loss from the crashed attempt describes weight updates that no longer exist."""
        learning = load_learning_ledger(self.paths.learning_path)
        effective = effective_events(learning)
        self.assertLess(len(effective), len(learning))

        # Nothing an attempt-1 row supersedes may survive into the aggregate.
        superseded = {
            (event.global_step, event.microbatch_id, event.sample_id)
            for event in learning
            if event.attempt == 1
        }
        for event in effective:
            key = (event.global_step, event.microbatch_id, event.sample_id)
            if key in superseded:
                self.assertEqual(event.attempt, 1)

        aggregates = aggregate_by_shard(learning)
        self.assertTrue(aggregates)

    def test_writes_resume_verification_report(self) -> None:
        path = write_resume_verification(self.paths.reports_dir, self.verification)
        self.assertTrue(path.is_file())

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["skipped_batches"], [])
        self.assertEqual(payload["repeated_batches"], [])
        self.assertEqual(payload["resumed_from_checkpoint_step"], RESUME_STEP)
        self.assertEqual(payload["prior_attempt"], 0)
        self.assertEqual(payload["resumed_attempt"], 1)
        self.assertEqual(payload["batches_compared"], payload["batches_matched"])
        self.assertGreater(payload["batches_compared"], 0)

        # Every comparison carries both sides, so a reader can recheck the claim.
        for comparison in payload["comparisons"]:
            self.assertIn("expected_batch_content_hash", comparison)
            self.assertIn("actual_batch_content_hash", comparison)

    def test_verification_fails_when_a_batch_diverges(self) -> None:
        """The report must be capable of failing, not just of passing."""
        broken = self.verification.comparisons[0].__class__(
            **{
                **self.verification.comparisons[0].__dict__,
                "actual_batch_content_hash": "sha256:tampered",
            }
        )
        self.assertFalse(broken.matched)

        report = self.verification.__class__(
            **{**self.verification.__dict__, "comparisons": (broken,)}
        )
        self.assertFalse(report.passed)
        self.assertEqual(len(report.mismatched), 1)


class TestResumeGuards(unittest.TestCase):
    """Resume must refuse states it cannot honestly recover from."""

    def test_rejects_resume_offset_past_the_ledger_tail(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)
        ledger_path = temp_dir / "consumption.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("", encoding="utf-8")

        with self.assertRaises(LedgerError):
            LedgerWriter.resume_at(ledger_path, ledger_offset=0)

    def test_checkpoints_available_lists_complete_checkpoints_only(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)
        (temp_dir / "ckpt-00010").mkdir(parents=True)
        (temp_dir / "ckpt-00010" / "checkpoint.json").write_text("{}", encoding="utf-8")
        # A directory with no metadata is a torn write, not a resumable checkpoint.
        (temp_dir / "ckpt-00020").mkdir(parents=True)

        self.assertEqual(checkpoints_available(temp_dir), (10,))
        self.assertEqual(checkpoints_available(temp_dir / "missing"), ())

    def test_rejects_checkpoint_from_a_different_run(self) -> None:
        demo, curriculum = load_configs(_ASSIGNMENT)
        schedule = compile_schedule(curriculum, total_steps=demo.training.total_steps)
        tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, documents = load_corpus(_CORPUS)

        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)
        result = build_shards_with_manifests(
            documents,
            tokenizer=tokenizer,
            shards_dir=temp_dir / "shards",
            manifests_dir=temp_dir / "manifests",
        )
        pool = build_sample_pool(temp_dir / "manifests", documents)
        run_plan = plan_run(
            schedule.steps,
            pool,
            run_id=demo.run.run_id,
            branch_id="a-different-branch",
            seed=demo.run.seed,
            global_batch_size=demo.training.global_batch_size,
        )
        paths = TrainingPaths.under(temp_dir / "artifacts")
        runner = build_training_runner(
            TrainingContext(
                demo=demo,
                curriculum=curriculum,
                schedule=schedule,
                run_plan=run_plan,
                tokenizer=tokenizer,
                documents_by_id={doc["document_id"]: doc for doc in documents},
                registry=build_eval_registry(documents, manifests_dir=result.manifests_dir),
            ),
            paths,
        )
        runner.run(stop_at_step=demo.training.checkpoint_interval)

        # The plan's branch no longer matches the checkpoint's.
        mismatched_plan = plan_run(
            schedule.steps,
            pool,
            run_id=demo.run.run_id,
            branch_id=demo.run.branch_id,
            seed=demo.run.seed,
            global_batch_size=demo.training.global_batch_size,
        )
        with self.assertRaises(RecoveryError):
            resume_from_checkpoint(
                TrainingContext(
                    demo=demo,
                    curriculum=curriculum,
                    schedule=schedule,
                    run_plan=mismatched_plan,
                    tokenizer=tokenizer,
                    documents_by_id={doc["document_id"]: doc for doc in documents},
                    registry=build_eval_registry(documents, manifests_dir=result.manifests_dir),
                ),
                paths,
                checkpoint_step=demo.training.checkpoint_interval,
            )


if __name__ == "__main__":
    unittest.main()
