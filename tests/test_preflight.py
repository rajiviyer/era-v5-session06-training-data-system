"""Tests for PX pre-flight scripts and reports."""

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

from config import load_configs  # noqa: E402
from preflight import (  # noqa: E402
    PreflightInputs,
    build_admission_audit,
    build_data_card,
    build_dataset_supply,
    verify_artifacts,
    write_data_card,
    write_preflight_reports,
)
from schedule import compile_schedule, write_schedule_json  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from corpus import load_corpus  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestPreflightReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.artifacts = cls.temp_dir / "submission_artifacts"
        cls.artifacts.mkdir(parents=True)

        demo, curriculum = load_configs(_ASSIGNMENT)
        _, documents = load_corpus(demo.paths.toy_corpus)
        tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        build_shards_with_manifests(
            documents,
            tokenizer=tokenizer,
            shards_dir=cls.artifacts / "shards",
            manifests_dir=cls.artifacts / "manifests",
        )
        schedule = compile_schedule(curriculum, total_steps=demo.training.total_steps)
        write_schedule_json(cls.artifacts / "schedule.json", schedule)

        cls.inputs = PreflightInputs(
            assignment_root=_ASSIGNMENT,
            artifacts_root=cls.artifacts,
            schedule_path=cls.artifacts / "schedule.json",
            manifests_dir=cls.artifacts / "manifests",
            corpus_dir=demo.paths.toy_corpus,
        )

    def test_dataset_supply_flags_stages(self) -> None:
        supply = build_dataset_supply(self.inputs)
        self.assertEqual(supply["total_steps"], 50)
        self.assertEqual(len(supply["stages"]), 3)
        self.assertIn("foundation", [stage["stage"] for stage in supply["stages"]])

    def test_admission_audit_lists_blocked_shard(self) -> None:
        audit = build_admission_audit(self.inputs)
        self.assertGreaterEqual(audit["admitted_count"], 1)
        self.assertGreaterEqual(audit["blocked_count"], 1)
        blocked_ids = {row["shard_id"] for row in audit["blocked"]}
        self.assertTrue(blocked_ids)

    def test_preflight_reports_are_written(self) -> None:
        reports_dir = self.artifacts / "reports"
        supply_path, admission_path = write_preflight_reports(self.inputs, reports_dir)
        self.assertTrue(supply_path.is_file())
        self.assertTrue(admission_path.is_file())
        payload = json.loads(supply_path.read_text(encoding="utf-8"))
        self.assertIn("stages", payload)

    def test_data_card_is_deterministic(self) -> None:
        first = build_data_card(self.inputs)
        second = build_data_card(self.inputs)
        self.assertEqual(first, second)
        self.assertTrue(first["tokenizer"]["tokenizer_hash"].startswith("tok_"))
        self.assertGreater(first["corpus"]["ready_document_total"], 0)
        self.assertGreater(first["shards"]["admitted_token_total"], 0)

    def test_data_card_files_written(self) -> None:
        reports_dir = self.artifacts / "reports"
        json_path, md_path = write_data_card(self.inputs, reports_dir)
        self.assertTrue(json_path.is_file())
        self.assertTrue(md_path.is_file())
        self.assertIn("Session 6 Data Card", md_path.read_text(encoding="utf-8"))


class TestVerifyArtifactsScript(unittest.TestCase):
    def test_verify_passes_on_generated_artifacts(self) -> None:
        artifacts = _ASSIGNMENT / "submission_artifacts"
        if not (artifacts / "evidence.json").is_file():
            self.skipTest("submission_artifacts not generated yet")
        result = verify_artifacts(artifacts, _ASSIGNMENT)
        self.assertTrue(result.passed, result.details)


if __name__ == "__main__":
    unittest.main()
