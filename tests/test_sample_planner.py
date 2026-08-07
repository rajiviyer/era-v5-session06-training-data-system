"""Tests for Always-ON sampler, anneal filter, and sample planner (P3-T04–T06)."""

from __future__ import annotations

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
from schedule import (  # noqa: E402
    build_sample_pool,
    compile_schedule,
    filter_opus_candidates,
    plan_run,
    plan_step,
)
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestSamplePlanner(unittest.TestCase):
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
        result = build_shards_with_manifests(
            cls.documents,
            tokenizer=cls.tokenizer,
            shards_dir=cls.temp_dir / "shards",
            manifests_dir=cls.temp_dir / "manifests",
        )
        cls.pool = build_sample_pool(cls.temp_dir / "manifests", cls.documents)
        cls.run_kwargs = {
            "run_id": cls.demo.run.run_id,
            "branch_id": cls.demo.run.branch_id,
            "seed": cls.demo.run.seed,
            "global_batch_size": cls.demo.training.global_batch_size,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_planner_deterministic(self) -> None:
        first = plan_run(self.schedule.steps, self.pool, **self.run_kwargs)
        second = plan_run(self.schedule.steps, self.pool, **self.run_kwargs)
        self.assertEqual(
            [sample.sample_id for step in first.steps for sample in step.samples],
            [sample.sample_id for step in second.steps for sample in step.samples],
        )

    def test_planner_branch_changes_plan(self) -> None:
        baseline = plan_run(self.schedule.steps, self.pool, **self.run_kwargs)
        forked = plan_run(
            self.schedule.steps,
            self.pool,
            run_id=self.run_kwargs["run_id"],
            branch_id="run-a-fork-1",
            seed=self.run_kwargs["seed"],
            global_batch_size=self.run_kwargs["global_batch_size"],
        )
        self.assertNotEqual(
            [sample.sample_id for step in baseline.steps for sample in step.samples],
            [sample.sample_id for step in forked.steps for sample in step.samples],
        )

    def test_always_on_fraction_met(self) -> None:
        run_plan = plan_run(self.schedule.steps, self.pool, **self.run_kwargs)
        total_samples = sum(len(step.samples) for step in run_plan.steps)
        always_on_samples = sum(step.always_on_count for step in run_plan.steps)
        observed = always_on_samples / total_samples
        self.assertAlmostEqual(observed, 0.11, delta=0.03)

    def test_always_on_slots_bypass_opus_path(self) -> None:
        run_plan = plan_run(self.schedule.steps, self.pool, **self.run_kwargs)
        always_on_rows = [
            sample
            for step in run_plan.steps
            for sample in step.samples
            if sample.path == "always_on"
        ]
        self.assertGreater(len(always_on_rows), 0)
        self.assertTrue(all(sample.sub_share is not None for sample in always_on_rows))

    def test_anneal_reserve_excludes_holdback_before_anneal(self) -> None:
        pre_anneal = self.schedule.steps[10]
        opus_candidates = filter_opus_candidates(self.pool, pre_anneal)
        self.assertFalse(any(candidate.anneal_eligible for candidate in opus_candidates))

    def test_anneal_phase_uses_anneal_eligible_only(self) -> None:
        anneal_step = self.schedule.steps[45]
        self.assertTrue(anneal_step.anneal_eligible_only)
        opus_candidates = filter_opus_candidates(self.pool, anneal_step)
        self.assertGreater(len(opus_candidates), 0)
        self.assertTrue(all(candidate.anneal_eligible for candidate in opus_candidates))

    def test_planner_respects_lane_quotas(self) -> None:
        run_plan = plan_run(self.schedule.steps, self.pool, **self.run_kwargs)
        expected_web = 0.0
        for step in self.schedule.steps:
            expected_web += step.opus_lane_quotas["web"]
        expected_web /= len(self.schedule.steps)

        opus_samples = [
            sample
            for step in run_plan.steps
            for sample in step.samples
            if sample.path == "opus"
        ]
        observed_web = sum(1 for sample in opus_samples if sample.capability_lane == "web") / len(
            opus_samples
        )
        self.assertGreater(observed_web, 0.0)
        self.assertAlmostEqual(observed_web, expected_web, delta=0.12)

    def test_plan_step_sample_ids_are_stable(self) -> None:
        step = self.schedule.steps[0]
        first = plan_step(step, self.pool, **self.run_kwargs)
        second = plan_step(step, self.pool, **self.run_kwargs)
        self.assertEqual(
            [sample.sample_id for sample in first.samples],
            [sample.sample_id for sample in second.samples],
        )


if __name__ == "__main__":
    unittest.main()
