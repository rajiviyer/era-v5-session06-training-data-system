"""Evidence bundle collected from generated artifacts (P11-T04, P11-T05, P11-T06).

The demo test asserts the bundle passes on a good run. What matters more here is the
opposite: every requirement below is re-collected against a *tampered* copy of the
artifacts, and must flip to failed. A check that cannot fail is not evidence, and
hardcoded `passed: true` values would survive every one of these.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from demo import run_demo  # noqa: E402
from evidence import (  # noqa: E402
    REQUIREMENT_KEYS,
    EvidenceError,
    collect_evidence,
    render_evidence_markdown,
    write_evidence_json,
)
from evidence.types import EVIDENCE_JSON_FILENAME  # noqa: E402


class TestEvidenceBundle(unittest.TestCase):
    """One demo run, then the bundle collected from what it left on disk."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.artifacts = cls.temp_dir / "submission_artifacts"
        cls.artifacts.mkdir(parents=True)
        run_demo(_ASSIGNMENT, artifacts_dir=cls.artifacts, clean=True)
        cls.bundle = collect_evidence(cls.artifacts, _ASSIGNMENT)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_all_fourteen_requirement_keys_are_present_and_passing(self) -> None:
        self.assertEqual(tuple(self.bundle.to_dict()["requirements"]), REQUIREMENT_KEYS)
        self.assertTrue(
            self.bundle.passed,
            "failed requirements: "
            + "; ".join(
                f"{key}: "
                + " | ".join(
                    check.detail for check in self.bundle.requirements[key].failures
                )
                for key in self.bundle.failed_keys
            ),
        )

    def test_every_evidence_path_points_at_a_real_artifact(self) -> None:
        self.assertEqual(self.bundle.missing_evidence_paths, ())
        for key in REQUIREMENT_KEYS:
            result = self.bundle.requirements[key]
            with self.subTest(requirement=key):
                self.assertTrue((self.artifacts / result.evidence_path).is_file())
                for path in result.evidence_paths:
                    self.assertTrue((self.artifacts / path).exists())

    def test_every_requirement_reports_the_measurements_it_decided_on(self) -> None:
        """A check with no detail is a green tick, not evidence."""
        for key in REQUIREMENT_KEYS:
            result = self.bundle.requirements[key]
            with self.subTest(requirement=key):
                self.assertTrue(result.checks)
                for check in result.checks:
                    self.assertTrue(check.detail.strip())

    def test_the_written_bundle_round_trips(self) -> None:
        path = write_evidence_json(self.artifacts, self.bundle)
        self.assertEqual(path.name, EVIDENCE_JSON_FILENAME)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["passed"], self.bundle.passed)
        self.assertEqual(list(payload["requirements"]), list(REQUIREMENT_KEYS))
        self.assertTrue(payload["generated_at"])
        self.assertIn("run_demo.py", payload["demo_command"])

    def test_markdown_is_rendered_from_the_bundle(self) -> None:
        markdown = render_evidence_markdown(self.bundle)
        self.assertIn("# Session 6 Evidence Report", markdown)
        self.assertIn(f"{len(REQUIREMENT_KEYS)} requirements passed", markdown)
        for key in REQUIREMENT_KEYS:
            self.assertIn(f"`{key}`", markdown)


class TestEvidenceDetectsTampering(unittest.TestCase):
    """Each requirement must fail when the artifact it grades is corrupted.

    The demo runs once for the class; every test copies that output, damages one thing,
    and re-collects. This is what separates a collector from a constant.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.good = cls.temp_dir / "good"
        cls.good.mkdir(parents=True)
        run_demo(_ASSIGNMENT, artifacts_dir=cls.good, clean=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _tampered_copy(self) -> Path:
        target = Path(tempfile.mkdtemp(dir=self.temp_dir)) / "artifacts"
        shutil.copytree(self.good, target)
        return target

    def _assert_fails(self, artifacts: Path, key: str) -> None:
        bundle = collect_evidence(artifacts, _ASSIGNMENT)
        self.assertIn(
            key,
            bundle.failed_keys,
            f"{key} still passed after its evidence was corrupted",
        )
        self.assertFalse(bundle.passed)

    @staticmethod
    def _rewrite_json(path: Path, mutate) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_editing_a_shard_file_breaks_the_content_hash(self) -> None:
        artifacts = self._tampered_copy()
        shard = sorted((artifacts / "shards").glob("*"))[0]
        shard.write_bytes(shard.read_bytes() + b"\x00")
        self._assert_fails(artifacts, "immutable_shards_with_manifests")

    def test_a_wrong_tokenizer_hash_in_the_manifest_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        self._rewrite_json(
            artifacts / "manifests" / "tokenizer_manifest.json",
            lambda payload: payload.__setitem__("tokenizer_hash", "tok_000000000000"),
        )
        self._assert_fails(artifacts, "frozen_tokenizer_hashes")

    def test_dropping_a_packing_policy_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        report = artifacts / "reports" / "packing_utilization.json"
        self._rewrite_json(
            report,
            lambda payload: payload["by_packing_policy"].pop("structure_preserving"),
        )
        self._assert_fails(artifacts, "packing_policies")

    def test_a_batch_claiming_more_loss_tokens_than_capacity_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        report = artifacts / "reports" / "packing_utilization.json"
        self._rewrite_json(
            report,
            lambda payload: payload["batches"][0].__setitem__("loss_bearing_tokens", 10_000),
        )
        self._assert_fails(artifacts, "correct_masks")

    def test_a_step_below_the_always_on_floor_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        self._rewrite_json(
            artifacts / "schedule.json",
            lambda payload: payload["steps"][3].__setitem__("always_on_fraction", 0.0),
        )
        self._assert_fails(artifacts, "curriculum_and_floors")

    def test_a_never_train_document_in_the_ledger_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        ledger = artifacts / "ledgers" / "consumption.jsonl"
        rows = [json.loads(line) for line in ledger.read_text("utf-8").splitlines() if line.strip()]
        # Both lists have to grow: the ledger schema requires them to stay aligned, and
        # this test is about the firewall check, not about ledger validation.
        rows[0]["packed_sample_ids"] = list(rows[0]["packed_sample_ids"]) + ["doc-eval-001"]
        rows[0]["shard_ids"] = list(rows[0]["shard_ids"]) + [rows[0]["shard_ids"][0]]
        ledger.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        self._assert_fails(artifacts, "eval_firewall")

    def test_a_committed_batch_with_no_opus_record_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        audit = artifacts / "ledgers" / "opus_audit.jsonl"
        lines = [line for line in audit.read_text("utf-8").splitlines() if line.strip()]
        kept = [line for line in lines if json.loads(line)["decision"] != "accepted"]
        audit.write_text("\n".join(kept) + "\n", encoding="utf-8")
        self._assert_fails(artifacts, "opus_audit_trail")

    def test_a_resume_report_claiming_an_unsupported_pass_is_caught(self) -> None:
        """The report says it passed; re-verifying the ledger must disagree."""
        artifacts = self._tampered_copy()
        self._rewrite_json(
            artifacts / "reports" / "resume_verification.json",
            lambda payload: payload.__setitem__("resumed_attempt", 0),
        )
        self._assert_fails(artifacts, "crash_resume_no_skip_repeat")

    def test_a_replayed_hash_that_contradicts_the_ledger_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        self._rewrite_json(
            artifacts / "reports" / "replay_verification.json",
            lambda payload: payload["comparisons"][0].__setitem__(
                "recomputed_batch_content_hash", "sha256:" + "0" * 64
            ),
        )
        self._assert_fails(artifacts, "replay_hash_match")

    def test_a_fork_that_never_diverged_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        self._rewrite_json(
            artifacts / "reports" / "fork_verification.json",
            lambda payload: (
                payload.__setitem__("divergence_step", None),
                payload.__setitem__("diverged_steps", []),
            ),
        )
        self._assert_fails(artifacts, "fork_new_branch")

    def test_a_utilization_that_does_not_recompute_is_caught(self) -> None:
        artifacts = self._tampered_copy()
        self._rewrite_json(
            artifacts / "reports" / "packing_utilization.json",
            lambda payload: payload.__setitem__("utilization", 0.99),
        )
        self._assert_fails(artifacts, "packing_and_throughput")

    def test_a_missing_evidence_file_fails_the_bundle(self) -> None:
        artifacts = self._tampered_copy()
        (artifacts / "reports" / "throughput.json").unlink()
        with self.assertRaises(EvidenceError):
            collect_evidence(artifacts, _ASSIGNMENT)


if __name__ == "__main__":
    unittest.main()
