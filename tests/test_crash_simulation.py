"""Tests for deliberate crash injection (P9-T01)."""

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

from checkpoint.types import checkpoint_dir_for_step  # noqa: E402
from config import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import build_eval_registry  # noqa: E402
from ledger import load_consumption_ledger, load_learning_ledger  # noqa: E402
from ledger.types import DataLoaderState  # noqa: E402
from recovery import CrashPolicy, RecoveryError, SimulatedCrash  # noqa: E402
from recovery.crash import CRASH_EVENT_TYPE  # noqa: E402
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402
from trainer import TrainingContext, TrainingPaths, build_training_runner  # noqa: E402

# Past the first checkpoint interval (10), so a crash leaves a checkpoint behind it.
CRASH_STEP = 11


def _state(next_global_step: int, next_microbatch_index: int) -> DataLoaderState:
    return DataLoaderState(
        run_id="s6-demo",
        branch_id="run-a",
        ledger_offset=7,
        next_global_step=next_global_step,
        next_microbatch_index=next_microbatch_index,
    )


class TestCrashPolicy(unittest.TestCase):
    """P9-T01: the policy fires exactly once, at the configured cursor position."""

    def test_fires_mid_step_by_default(self) -> None:
        policy = CrashPolicy(crash_at_step=CRASH_STEP)
        self.assertFalse(policy.should_crash(_state(CRASH_STEP, 0)))
        self.assertTrue(policy.should_crash(_state(CRASH_STEP, 1)))

    def test_zero_microbatches_fires_on_the_step_boundary(self) -> None:
        policy = CrashPolicy(crash_at_step=CRASH_STEP, crash_after_microbatches=0)
        self.assertTrue(policy.should_crash(_state(CRASH_STEP, 0)))

    def test_does_not_fire_on_other_steps(self) -> None:
        policy = CrashPolicy(crash_at_step=CRASH_STEP)
        self.assertFalse(policy.should_crash(_state(CRASH_STEP - 1, 1)))
        self.assertFalse(policy.should_crash(_state(CRASH_STEP + 1, 1)))

    def test_rejects_invalid_policy(self) -> None:
        with self.assertRaises(RecoveryError):
            CrashPolicy(crash_at_step=-1)
        with self.assertRaises(RecoveryError):
            CrashPolicy(crash_at_step=5, crash_after_microbatches=-1)

    def test_from_config_uses_demo_recovery_block(self) -> None:
        demo, _ = load_configs(_ASSIGNMENT)
        policy = CrashPolicy.from_config(demo.recovery)
        self.assertEqual(policy.crash_at_step, demo.recovery.crash_at_step)
        # The demo must crash after the checkpoint it resumes from, or there is
        # nothing for resume to reconcile.
        self.assertGreater(policy.crash_at_step, demo.recovery.resume_from_checkpoint_step)


class TestCrashLeavesPartialState(unittest.TestCase):
    """The abort must leave real on-disk state, not a tidy summary."""

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
        runner = build_training_runner(
            cls._context(),
            cls.paths,
            crash_policy=CrashPolicy(crash_at_step=CRASH_STEP),
        )
        cls.microbatches_per_step = runner.dataloader.microbatches_per_step
        try:
            runner.run(stop_at_step=cls.demo.training.total_steps)
        except SimulatedCrash as crash:
            cls.crash = crash
        else:  # pragma: no cover - the policy must fire
            raise AssertionError("run completed without crashing")

        cls.consumption = load_consumption_ledger(cls.paths.ledger_path)
        cls.learning = load_learning_ledger(cls.paths.learning_path)

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

    def test_run_aborts_at_the_configured_step(self) -> None:
        self.assertEqual(self.crash.global_step, CRASH_STEP)
        self.assertEqual(self.crash.microbatch_index, 1)
        self.assertEqual(self.crash.run_id, self.demo.run.run_id)
        self.assertEqual(self.crash.branch_id, self.demo.run.branch_id)

    def test_ledger_stops_inside_the_crash_step(self) -> None:
        self.assertGreater(len(self.consumption), 0)
        steps = [record.global_step for record in self.consumption]
        self.assertLessEqual(max(steps), CRASH_STEP)

        # The crash step is genuinely partial: fewer committed microbatches than a
        # whole step contains. (Gating can lower this further, never raise it.)
        crash_step_rows = sum(1 for step in steps if step == CRASH_STEP)
        self.assertLess(crash_step_rows, self.microbatches_per_step)

        # Offsets stay contiguous through the abort: nothing half-written.
        self.assertEqual(
            [record.ledger_offset for record in self.consumption],
            list(range(len(self.consumption))),
        )
        self.assertEqual(self.crash.ledger_offset, len(self.consumption) - 1)

    def test_learning_ledger_matches_the_committed_tail(self) -> None:
        """Both ledgers must stop at the same batch; neither runs ahead of the other."""
        committed = {record.microbatch_id for record in self.consumption}
        self.assertEqual({event.microbatch_id for event in self.learning}, committed)

    def test_newest_checkpoint_is_behind_the_ledger_tail(self) -> None:
        """The gap between checkpoint and ledger is what resume has to reconcile."""
        interval = self.demo.training.checkpoint_interval
        saved = checkpoint_dir_for_step(self.paths.checkpoints_dir, interval)
        self.assertTrue((saved / "checkpoint.json").is_file())

        # No checkpoint at or after the crash step: the run died before reaching one.
        for step in range(CRASH_STEP, self.demo.training.total_steps + 1):
            missing = checkpoint_dir_for_step(self.paths.checkpoints_dir, step)
            self.assertFalse((missing / "checkpoint.json").is_file())

        checkpoint = json.loads((saved / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertLess(checkpoint["ledger_offset"], self.crash.ledger_offset)
        self.assertLess(checkpoint["next_global_step"], CRASH_STEP)

    def test_crash_event_written_to_run_log(self) -> None:
        lines = self.paths.run_log_path.read_text(encoding="utf-8").splitlines()
        crashes = [
            json.loads(line)
            for line in lines
            if line.strip() and json.loads(line).get("event_type") == CRASH_EVENT_TYPE
        ]
        self.assertEqual(len(crashes), 1)

        event = crashes[0]
        self.assertEqual(event["global_step"], CRASH_STEP)
        self.assertEqual(event["ledger_offset"], self.crash.ledger_offset)
        self.assertEqual(event["run_id"], self.demo.run.run_id)
        # Names the checkpoint the operator should resume from.
        self.assertIsNotNone(event["last_checkpoint_id"])


class TestCrashOnStepBoundary(unittest.TestCase):
    """`crash_after_microbatches=0` must leave only whole steps behind."""

    def test_no_rows_for_the_crash_step(self) -> None:
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
            branch_id=demo.run.branch_id,
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
            crash_policy=CrashPolicy(crash_at_step=4, crash_after_microbatches=0),
        )

        with self.assertRaises(SimulatedCrash) as raised:
            runner.run(stop_at_step=demo.training.total_steps)

        self.assertEqual(raised.exception.microbatch_index, 0)
        steps = [record.global_step for record in load_consumption_ledger(paths.ledger_path)]
        self.assertLessEqual(max(steps), 3)


if __name__ == "__main__":
    unittest.main()
