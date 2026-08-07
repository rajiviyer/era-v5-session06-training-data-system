"""Tests for configuration loading and validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import ConfigError, load_configs, load_curriculum_config, load_demo_config  # noqa: E402


class TestLoadDemoConfig(unittest.TestCase):
    def test_loads_default_demo_yaml(self) -> None:
        demo = load_demo_config(_ASSIGNMENT / "configs" / "demo.yaml")
        self.assertEqual(demo.run.seed, 42)
        self.assertEqual(demo.training.total_steps, 50)
        self.assertEqual(demo.recovery.crash_at_step, 25)
        self.assertEqual(demo.recovery.resume_from_checkpoint_step, 20)
        self.assertTrue(demo.paths.submission_artifacts.name == "submission_artifacts")
        self.assertTrue(demo.paths.toy_corpus.is_absolute())

    def test_rejects_invalid_global_batch(self) -> None:
        yaml_text = """
run:
  run_id: x
  branch_id: y
  seed: 1
training:
  seq_len: 256
  global_batch_size: 3
  microbatch_size: 2
  gradient_accumulation_steps: 2
  total_steps: 10
  checkpoint_interval: 5
recovery:
  crash_at_step: 8
  resume_from_checkpoint_step: 5
  replay_start_step: 5
  replay_end_step: 8
  fork_from_checkpoint_step: 5
  fork_branch_id: fork
paths:
  toy_corpus: data/toy_corpus
  curriculum_config: configs/curriculum.yaml
  submission_artifacts: submission_artifacts
model:
  n_layers: 2
  n_heads: 4
  d_model: 128
  d_ff: 512
  dropout: 0.0
opus:
  accept_threshold: 0.5
  expected_rejection_rate: 0.1
logging:
  log_every_step: true
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_demo_config(path, assignment_root=_ASSIGNMENT)


class TestLoadCurriculumConfig(unittest.TestCase):
    def test_loads_default_curriculum_yaml(self) -> None:
        curriculum = load_curriculum_config(
            _ASSIGNMENT / "configs" / "curriculum.yaml",
            total_steps=50,
        )
        self.assertAlmostEqual(curriculum.batch.always_on_fraction, 0.11)
        self.assertEqual(len(curriculum.phases), 3)
        self.assertEqual(curriculum.phases[-1].name, "anneal")
        self.assertTrue(curriculum.phases[-1].anneal_eligible_only)

    def test_rejects_non_contiguous_phases(self) -> None:
        yaml_text = """
batch:
  always_on_fraction: 0.11
  opus_fraction: 0.89
transitions:
  warmup_steps: 1
  blend: [0.6, 0.4]
protected_floors:
  lanes: [indic]
phases:
  - name: a
    step_start: 0
    step_end: 10
    opus_mixture: {web: 0.5, code: 0.5}
  - name: b
    step_start: 12
    step_end: 20
    opus_mixture: {web: 0.5, code: 0.5}
always_on:
  fraction: 0.11
  stage_invariant: true
  sub_shares: {indic_tier_a: 1.0}
anneal_reserve:
  holdback_until_phase: anneal
  manifest_tag: anneal_eligible
manifest_filters:
  difficulty_field: curriculum_band
  reasoning_length_field: reasoning_trace_band
  anneal_holdback_field: anneal_eligible
  opus_excludes: [always_on_eligible]
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curriculum.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_curriculum_config(path, total_steps=20)


class TestLoadConfigs(unittest.TestCase):
    def test_loads_demo_and_curriculum_together(self) -> None:
        demo, curriculum = load_configs(_ASSIGNMENT)
        self.assertEqual(demo.training.total_steps, curriculum.phases[-1].step_end)
        self.assertAlmostEqual(curriculum.batch.always_on_fraction, 0.11)


if __name__ == "__main__":
    unittest.main()
