"""Tests for mixture schedule compiler (P3-T01–T03)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_configs, load_curriculum_config  # noqa: E402
from schedule import (  # noqa: E402
    compile_schedule,
    load_schedule_json,
    parse_stage_records,
    write_schedule_json,
)


class TestScheduleCompiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, cls.curriculum = load_configs(_ASSIGNMENT)
        cls.schedule = compile_schedule(
            cls.curriculum,
            total_steps=cls.demo.training.total_steps,
        )

    def test_parse_stage_records_matches_curriculum(self) -> None:
        stages = parse_stage_records(self.curriculum)
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0].name, "foundation")
        self.assertEqual(stages[-1].anneal_eligible_only, True)

    def test_schedule_compiler_stage_boundaries(self) -> None:
        steps = {entry.step: entry for entry in self.schedule.steps}
        self.assertEqual(steps[0].phase, "foundation")
        self.assertEqual(steps[19].phase, "foundation")
        self.assertEqual(steps[20].phase, "skill_build")
        self.assertEqual(steps[44].phase, "skill_build")
        self.assertEqual(steps[45].phase, "anneal")
        self.assertEqual(steps[49].phase, "anneal")
        self.assertFalse(steps[19].anneal_eligible_only)
        self.assertTrue(steps[45].anneal_eligible_only)

    def test_stage_mix_changes_across_phases(self) -> None:
        steps = {entry.step: entry for entry in self.schedule.steps}
        foundation_web = steps[10].opus_lane_quotas["web"]
        skill_build_web = steps[30].opus_lane_quotas["web"]
        anneal_web = steps[48].opus_lane_quotas["web"]
        self.assertGreater(foundation_web, skill_build_web)
        self.assertGreater(skill_build_web, anneal_web)

    def test_transition_warmup_blends_mixtures(self) -> None:
        steps = {entry.step: entry for entry in self.schedule.steps}
        self.assertFalse(steps[19].in_transition)
        self.assertTrue(steps[20].in_transition)
        self.assertTrue(steps[21].in_transition)
        self.assertFalse(steps[22].in_transition)

        foundation = self.curriculum.phases[0].opus_mixture["web"]
        skill_build = self.curriculum.phases[1].opus_mixture["web"]
        expected_blend = 0.6 * foundation + 0.4 * skill_build
        actual = steps[20].opus_lane_quotas["web"] / self.curriculum.batch.opus_fraction
        self.assertAlmostEqual(actual, expected_blend, places=6)

    def test_compiler_is_deterministic(self) -> None:
        again = compile_schedule(
            self.curriculum,
            total_steps=self.demo.training.total_steps,
        )
        self.assertEqual(self.schedule.to_dict(), again.to_dict())

    def test_always_on_fraction_constant(self) -> None:
        for entry in self.schedule.steps:
            self.assertAlmostEqual(entry.always_on_fraction, 0.11)
            self.assertAlmostEqual(entry.opus_fraction, 0.89)

    def test_schedule_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.json"
            write_schedule_json(path, self.schedule)
            loaded = load_schedule_json(path)
            self.assertEqual(loaded.to_dict(), self.schedule.to_dict())

    def test_written_schedule_has_phase_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.json"
            write_schedule_json(path, self.schedule)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_steps"], 50)
            self.assertEqual(len(payload["phase_boundaries"]), 3)
            self.assertEqual(len(payload["steps"]), 50)

    def test_supply_shortfall_emits_warning(self) -> None:
        schedule = compile_schedule(
            self.curriculum,
            total_steps=self.demo.training.total_steps,
            lane_supply={"web": 0, "code": 5},
        )
        self.assertTrue(any("web" in warning for warning in schedule.warnings))


class TestScheduleCompilerValidation(unittest.TestCase):
    def test_rejects_mismatched_total_steps(self) -> None:
        curriculum = load_curriculum_config(
            _ASSIGNMENT / "configs" / "curriculum.yaml",
            total_steps=50,
        )
        with self.assertRaises(Exception):
            compile_schedule(curriculum, total_steps=40)


if __name__ == "__main__":
    unittest.main()
