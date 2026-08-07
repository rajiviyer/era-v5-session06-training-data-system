"""Tests for packing utilization and throughput (P10-T01–T04)."""

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
from ledger.rebuild import rebuild_batch  # noqa: E402
from metrics import (  # noqa: E402
    StepTiming,
    UTILIZATION_FORMULA,
    compute_packing_utilization,
    compute_throughput,
    load_step_timings,
    write_packing_report,
    write_throughput_report,
)
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402
from trainer import TrainingContext, TrainingPaths, build_training_runner  # noqa: E402

TRAIN_STEPS = 6


class TestMetricsFromRealRun(unittest.TestCase):
    """Metrics must be reconstructible from the artifacts, not read back from a cache."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, cls.curriculum = load_configs(_ASSIGNMENT)
        cls.schedule = compile_schedule(
            cls.curriculum, total_steps=cls.demo.training.total_steps
        )
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, cls.documents = load_corpus(_CORPUS)
        cls.documents_by_id = {doc["document_id"]: doc for doc in cls.documents}

        cls.temp_dir = Path(tempfile.mkdtemp())
        built = build_shards_with_manifests(
            cls.documents,
            tokenizer=cls.tokenizer,
            shards_dir=cls.temp_dir / "shards",
            manifests_dir=cls.temp_dir / "manifests",
        )
        pool = build_sample_pool(cls.temp_dir / "manifests", cls.documents)
        run_plan = plan_run(
            cls.schedule.steps,
            pool,
            run_id=cls.demo.run.run_id,
            branch_id=cls.demo.run.branch_id,
            seed=cls.demo.run.seed,
            global_batch_size=cls.demo.training.global_batch_size,
        )
        cls.paths = TrainingPaths.under(Path(tempfile.mkdtemp(dir=cls.temp_dir)))
        runner = build_training_runner(
            TrainingContext(
                demo=cls.demo,
                curriculum=cls.curriculum,
                schedule=cls.schedule,
                run_plan=run_plan,
                tokenizer=cls.tokenizer,
                documents_by_id=cls.documents_by_id,
                registry=build_eval_registry(
                    cls.documents, manifests_dir=built.manifests_dir
                ),
            ),
            cls.paths,
        )
        runner.run(stop_at_step=TRAIN_STEPS)

        cls.ledger = load_consumption_ledger(cls.paths.ledger_path)
        cls.timings = load_step_timings(cls.paths.timings_path)
        cls.packing = compute_packing_utilization(
            cls.ledger,
            documents_by_id=cls.documents_by_id,
            tokenizer=cls.tokenizer,
            seq_len=cls.demo.training.seq_len,
        )
        cls.throughput = compute_throughput(cls.packing.batches, cls.timings)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_packing_utilization_recomputable(self) -> None:
        """P10-T02: an independent recomputation must reproduce every reported value."""
        self.assertEqual(len(self.packing.batches), len(self.ledger))

        for measured, row in zip(self.packing.batches, self.ledger):
            assembled = rebuild_batch(
                row,
                documents_by_id=self.documents_by_id,
                tokenizer=self.tokenizer,
                seq_len=self.demo.training.seq_len,
            )
            batch = assembled.batch
            capacity = batch.batch_size * batch.seq_len

            self.assertEqual(measured.capacity, capacity)
            self.assertEqual(measured.useful_tokens, assembled.useful_tokens)
            self.assertEqual(
                measured.loss_bearing_tokens,
                sum(sum(mask_row) for mask_row in batch.loss_mask),
            )
            # The documented formula, applied literally.
            self.assertAlmostEqual(
                measured.utilization,
                assembled.useful_tokens / capacity,
                places=9,
            )

    def test_run_utilization_is_token_weighted(self) -> None:
        """Averaging per-batch ratios would let a small batch outweigh a large one."""
        expected = self.packing.total_useful_tokens / self.packing.total_capacity
        self.assertAlmostEqual(self.packing.utilization, expected, places=9)
        self.assertGreater(self.packing.utilization, 0.0)
        self.assertLessEqual(self.packing.utilization, 1.0)

    def test_per_policy_split_covers_every_batch(self) -> None:
        by_policy = self.packing.by_policy()
        self.assertTrue(by_policy)
        self.assertEqual(
            sum(stats["batches"] for stats in by_policy.values()),
            len(self.packing.batches),
        )
        self.assertEqual(
            sum(stats["capacity"] for stats in by_policy.values()),
            self.packing.total_capacity,
        )

    def test_throughput_metrics_from_ledger(self) -> None:
        """P10-T04: rate is tokens (from the ledger) over seconds (from timings)."""
        self.assertTrue(self.throughput.steps)
        self.assertEqual(self.throughput.steps_without_timings, ())
        self.assertGreater(self.throughput.total_wall_seconds, 0.0)

        seconds_by_step = {
            (row.attempt, row.global_step): row.wall_seconds for row in self.timings
        }
        for step in self.throughput.steps:
            key = (step.attempt, step.global_step)
            self.assertEqual(step.wall_seconds, seconds_by_step[key])

            # Tokens come from the batches at that step, nowhere else.
            batches = [
                batch
                for batch in self.packing.batches
                if (batch.attempt, batch.global_step) == key
            ]
            self.assertEqual(
                step.raw_tokens, sum(batch.useful_tokens for batch in batches)
            )
            self.assertEqual(
                step.loss_bearing_tokens,
                sum(batch.loss_bearing_tokens for batch in batches),
            )
            self.assertAlmostEqual(
                step.loss_bearing_tokens_per_second,
                step.loss_bearing_tokens / step.wall_seconds,
                places=6,
            )

        # Loss-bearing tokens are a subset of raw tokens, so the rate cannot exceed it.
        self.assertLessEqual(
            self.throughput.loss_bearing_tokens_per_second,
            self.throughput.raw_tokens_per_second,
        )

    def test_missing_timings_are_reported_not_treated_as_free(self) -> None:
        """Dropping a duration must shrink the measured set, never inflate the rate."""
        partial = compute_throughput(self.packing.batches, self.timings[:1])
        self.assertEqual(len(partial.steps), 1)
        self.assertTrue(partial.steps_without_timings)
        self.assertLess(len(partial.steps), len(self.throughput.steps))

    def test_reports_are_written_with_formula_and_sources(self) -> None:
        packing_path = write_packing_report(self.paths.reports_dir, self.packing)
        throughput_path = write_throughput_report(self.paths.reports_dir, self.throughput)

        packing = json.loads(packing_path.read_text(encoding="utf-8"))
        self.assertEqual(packing["formula"], UTILIZATION_FORMULA)
        self.assertEqual(packing["batches_measured"], len(self.ledger))
        self.assertEqual(
            packing["total_useful_tokens"], self.packing.total_useful_tokens
        )
        # Per-batch rows are present, so a reader can recheck the aggregate.
        self.assertEqual(len(packing["batches"]), len(self.ledger))
        self.assertAlmostEqual(
            sum(row["useful_tokens"] for row in packing["batches"]),
            packing["total_useful_tokens"],
        )

        throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
        self.assertIn("consumption.jsonl", throughput["sources"]["tokens"])
        self.assertIn("step_timings.jsonl", throughput["sources"]["seconds"])
        self.assertEqual(
            sum(row["loss_bearing_tokens"] for row in throughput["steps"]),
            throughput["total_loss_bearing_tokens"],
        )

    def test_timings_cover_every_trained_step(self) -> None:
        timed = {(row.attempt, row.global_step) for row in self.timings}
        committed = {(row.attempt, row.global_step) for row in self.ledger}
        self.assertTrue(committed.issubset(timed))
        for row in self.timings:
            self.assertGreater(row.wall_seconds, 0.0)


class TestThroughputEdgeCases(unittest.TestCase):
    def test_rejects_empty_batches(self) -> None:
        with self.assertRaises(ValueError):
            compute_throughput([], [])

    def test_zero_duration_step_does_not_divide_by_zero(self) -> None:
        from metrics.packing import BatchUtilization

        batch = BatchUtilization(
            attempt=0,
            global_step=0,
            microbatch_id="mb-00000-0",
            ledger_offset=0,
            packing_policy="concat_and_chop",
            num_sequences=2,
            seq_len=16,
            useful_tokens=20,
            loss_bearing_tokens=18,
        )
        timing = StepTiming(
            run_id="r", branch_id="b", attempt=0, global_step=0, wall_seconds=0.0
        )
        report = compute_throughput([batch], [timing])
        self.assertEqual(report.raw_tokens_per_second, 0.0)
        self.assertEqual(report.loss_bearing_tokens_per_second, 0.0)


if __name__ == "__main__":
    unittest.main()
