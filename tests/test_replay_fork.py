"""Tests for historical replay and forking (P9-T05–T08)."""

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
from ledger import load_consumption_ledger  # noqa: E402
from ledger.commit import make_microbatch_id, parse_microbatch_id  # noqa: E402
from recovery import RecoveryError  # noqa: E402
from recovery.fork import (  # noqa: E402
    FORK_LOG_FILENAME,
    fork_from_checkpoint,
    verify_fork,
    write_fork_verification,
)
from recovery.replay import replay_range, write_replay_verification  # noqa: E402
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402
from trainer import (  # noqa: E402
    TrainerError,
    TrainingContext,
    TrainingPaths,
    build_training_runner,
)

TRAIN_STEPS = 10  # equals checkpoint_interval, so ckpt-00010 exists to fork from


class TestMicrobatchId(unittest.TestCase):
    """Replay positions the dataloader from an ID, so the mapping must round-trip."""

    def test_round_trip(self) -> None:
        self.assertEqual(parse_microbatch_id(make_microbatch_id(20, 1)), (20, 1))
        self.assertEqual(parse_microbatch_id("mb-00000-0"), (0, 0))

    def test_rejects_malformed(self) -> None:
        for bad in ("mb-20", "step-00020-1", "mb-xx-1", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_microbatch_id(bad)


class TestReplayAndFork(unittest.TestCase):
    """One trained run, then replay it and fork from it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, cls.curriculum = load_configs(_ASSIGNMENT)
        cls.schedule = compile_schedule(
            cls.curriculum,
            total_steps=cls.demo.training.total_steps,
        )
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, cls.documents = load_corpus(_CORPUS)

        cls.temp_dir = Path(tempfile.mkdtemp())
        built = build_shards_with_manifests(
            cls.documents,
            tokenizer=cls.tokenizer,
            shards_dir=cls.temp_dir / "shards",
            manifests_dir=cls.temp_dir / "manifests",
        )
        cls.registry = build_eval_registry(cls.documents, manifests_dir=built.manifests_dir)
        cls.pool = build_sample_pool(cls.temp_dir / "manifests", cls.documents)
        cls.run_plan = plan_run(
            cls.schedule.steps,
            cls.pool,
            run_id=cls.demo.run.run_id,
            branch_id=cls.demo.run.branch_id,
            seed=cls.demo.run.seed,
            global_batch_size=cls.demo.training.global_batch_size,
        )

        cls.paths = TrainingPaths.under(Path(tempfile.mkdtemp(dir=cls.temp_dir)))
        runner = build_training_runner(cls._context(), cls.paths)
        # Train past the checkpoint the fork branches from, so parent and fork cover
        # overlapping steps and their streams can actually be compared.
        cls.summary = runner.run(stop_at_step=TRAIN_STEPS + 3)

    @classmethod
    def _context(cls) -> TrainingContext:
        return TrainingContext(
            demo=cls.demo,
            curriculum=cls.curriculum,
            schedule=cls.schedule,
            run_plan=cls.run_plan,
            tokenizer=cls.tokenizer,
            documents_by_id={doc["document_id"]: doc for doc in cls.documents},
            registry=cls.registry,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_replay_hash_match(self) -> None:
        """P9-T06: re-derived batches match recorded IDs, spans, and hashes."""
        verification = replay_range(
            self._context(),
            self.paths,
            start_step=0,
            end_step=TRAIN_STEPS - 1,
        )
        self.assertTrue(verification.comparisons)
        self.assertEqual(verification.mismatched, ())
        self.assertTrue(verification.passed)

        for comparison in verification.comparisons:
            # The planner independently produced the same samples...
            self.assertTrue(comparison.planned_sample_ids_match)
            # ...and re-tokenizing/packing/masking produced the same bytes.
            self.assertEqual(
                comparison.recorded_batch_content_hash,
                comparison.recomputed_batch_content_hash,
            )
            self.assertEqual(
                comparison.recorded_loss_mask_hash,
                comparison.recomputed_loss_mask_hash,
            )
            self.assertTrue(comparison.token_spans_match)

    def test_replay_covers_every_committed_batch_in_range(self) -> None:
        verification = replay_range(
            self._context(),
            self.paths,
            start_step=0,
            end_step=TRAIN_STEPS - 1,
        )
        committed = [
            record
            for record in load_consumption_ledger(self.paths.ledger_path)
            if record.global_step <= TRAIN_STEPS - 1
        ]
        self.assertEqual(len(verification.comparisons), len(committed))

    def test_replay_detects_a_tampered_ledger(self) -> None:
        """Replay recomputes rather than reads back, so an edited hash must fail."""
        tampered_root = Path(tempfile.mkdtemp(dir=self.temp_dir))
        tampered = TrainingPaths.under(tampered_root)
        tampered.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        lines = self.paths.ledger_path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["batch_content_hash"] = "sha256:" + "0" * 64
        lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        tampered.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        verification = replay_range(
            self._context(),
            tampered,
            start_step=0,
            end_step=TRAIN_STEPS - 1,
        )
        self.assertFalse(verification.passed)
        self.assertEqual(len(verification.mismatched), 1)

    def test_replay_rejects_an_empty_range(self) -> None:
        with self.assertRaises(RecoveryError):
            replay_range(self._context(), self.paths, start_step=5, end_step=4)

    def test_replay_report_is_written(self) -> None:
        verification = replay_range(
            self._context(), self.paths, start_step=0, end_step=TRAIN_STEPS - 1
        )
        path = write_replay_verification(self.paths.reports_dir, verification)
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["batches_replayed"], payload["batches_matched"])
        for comparison in payload["comparisons"]:
            self.assertIn("recorded_batch_content_hash", comparison)
            self.assertIn("recomputed_batch_content_hash", comparison)

    def _fork(self, branch_id: str = "run-a-fork-test"):
        forked = fork_from_checkpoint(
            self._context(),
            self.paths,
            checkpoint_step=TRAIN_STEPS,
            new_branch_id=branch_id,
            pool=self.pool,
        )
        forked.runner.run(stop_at_step=TRAIN_STEPS + 3)
        return forked

    def test_fork_new_branch_id(self) -> None:
        """P9-T08: a new branch, its own lineage, and a logged divergence point."""
        forked = self._fork()
        self.assertEqual(forked.branch_id, "run-a-fork-test")
        self.assertNotEqual(forked.event.parent_branch_id, forked.branch_id)
        self.assertEqual(forked.event.forked_from_step, TRAIN_STEPS)

        # The branch writes its own ledger, not into the parent's.
        self.assertNotEqual(forked.paths.ledger_path, self.paths.ledger_path)
        child = load_consumption_ledger(forked.paths.ledger_path)
        self.assertTrue(child)
        self.assertEqual({record.branch_id for record in child}, {forked.branch_id})
        # A new lineage starts its own offsets at 0.
        self.assertEqual(child[0].ledger_offset, 0)

    def test_fork_event_records_parent_and_offset(self) -> None:
        forked = self._fork("run-a-fork-log")
        log_path = self.paths.ledger_path.parent / FORK_LOG_FILENAME
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [e for e in events if e["child_branch_id"] == "run-a-fork-log"]
        self.assertEqual(len(matching), 1)

        event = matching[0]
        self.assertEqual(event["event_type"], "fork")
        self.assertEqual(event["parent_branch_id"], self.demo.run.branch_id)
        self.assertEqual(event["forked_from_step"], TRAIN_STEPS)
        self.assertIn("parent_ledger_offset", event)

    def test_forked_stream_diverges_from_the_parent(self) -> None:
        """The planner is seeded on branch_id, so a fork draws a different stream."""
        forked = self._fork("run-a-fork-diverge")
        verification = verify_fork(self.paths, forked)
        self.assertTrue(verification.compared_steps)
        self.assertIsNotNone(verification.divergence_step)
        self.assertTrue(verification.passed)

        report = write_fork_verification(forked.paths.reports_dir, verification)
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["child_branch_id"], "run-a-fork-diverge")
        self.assertGreater(payload["child_batches"], 0)

    def test_fork_rejects_reusing_the_parent_branch_id(self) -> None:
        with self.assertRaises(RecoveryError):
            fork_from_checkpoint(
                self._context(),
                self.paths,
                checkpoint_step=TRAIN_STEPS,
                new_branch_id=self.demo.run.branch_id,
                pool=self.pool,
            )

    def test_fresh_runner_refuses_to_retrain_over_an_existing_ledger(self) -> None:
        """Continuing a run is resume's job; a fresh runner would duplicate steps 0..N."""
        with self.assertRaises(TrainerError):
            build_training_runner(self._context(), self.paths)


if __name__ == "__main__":
    unittest.main()
