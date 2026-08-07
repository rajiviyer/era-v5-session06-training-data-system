"""Integration test for the one-command demo (P11-T02, P11-T03)."""

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

from demo import DemoResult, PhaseResult, run_demo  # noqa: E402
from ledger import load_consumption_ledger, load_learning_ledger  # noqa: E402
from ledger.reader import latest_attempt  # noqa: E402
from runlog import events_of_type, load_run_log, missing_event_types  # noqa: E402


class TestDemoEndToEnd(unittest.TestCase):
    """One unattended run from a clean directory, then check what it left behind."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.artifacts = cls.temp_dir / "submission_artifacts"
        cls.artifacts.mkdir(parents=True)

        # Stale output from an earlier run must not survive into this one.
        (cls.artifacts / "stale.json").write_text("{}", encoding="utf-8")
        (cls.artifacts / ".gitkeep").write_text("", encoding="utf-8")

        cls.result = run_demo(_ASSIGNMENT, artifacts_dir=cls.artifacts, clean=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_all_phases_pass(self) -> None:
        self.assertTrue(
            self.result.passed,
            "failed phases: "
            + "; ".join(
                f"{phase.name}: {phase.detail}"
                for phase in self.result.phases
                if not phase.passed
            ),
        )
        self.assertGreaterEqual(len(self.result.phases), 11)
        self.assertEqual(
            [phase.number for phase in self.result.phases],
            sorted(phase.number for phase in self.result.phases),
        )

    def test_clean_run_removes_stale_artifacts_but_keeps_gitkeep(self) -> None:
        self.assertFalse((self.artifacts / "stale.json").exists())
        self.assertTrue((self.artifacts / ".gitkeep").is_file())

    def test_regenerates_the_full_artifact_tree(self) -> None:
        for relative in (
            "schedule.json",
            "run.log",
            "evidence.json",
            "evidence.md",
            "eval_registry.json",
            "manifests/shard_registry.json",
            "manifests/tokenizer_manifest.json",
            "ledgers/consumption.jsonl",
            "ledgers/learning.jsonl",
            "ledgers/opus_audit.jsonl",
            "ledgers/forks.jsonl",
            "reports/resume_verification.json",
            "reports/replay_verification.json",
            "reports/fork_verification.json",
            "reports/packing_utilization.json",
            "reports/throughput.json",
            "reports/step_timings.jsonl",
        ):
            with self.subTest(relative=relative):
                self.assertTrue(
                    (self.artifacts / relative).is_file(),
                    f"demo did not produce {relative}",
                )

        checkpoints = sorted(p.name for p in (self.artifacts / "checkpoints").glob("ckpt-*"))
        self.assertTrue(checkpoints)
        for name in checkpoints:
            directory = self.artifacts / "checkpoints" / name
            self.assertTrue((directory / "checkpoint.json").is_file())
            self.assertTrue((directory / "model.pt").is_file())

    def test_run_crashed_and_resumed_for_real(self) -> None:
        """The demo must exercise recovery, not train straight through."""
        ledger = load_consumption_ledger(self.artifacts / "ledgers" / "consumption.jsonl")
        self.assertEqual(latest_attempt(ledger), 1)

        learning = load_learning_ledger(self.artifacts / "ledgers" / "learning.jsonl")
        self.assertEqual({event.attempt for event in learning}, {0, 1})

        report = json.loads(
            (self.artifacts / "reports" / "resume_verification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(report["passed"])
        self.assertGreater(report["batches_compared"], 0)
        self.assertEqual(report["skipped_batches"], [])
        self.assertEqual(report["repeated_batches"], [])

    def test_replay_and_fork_reports_pass(self) -> None:
        replay = json.loads(
            (self.artifacts / "reports" / "replay_verification.json").read_text("utf-8")
        )
        self.assertTrue(replay["passed"])
        self.assertEqual(replay["batches_replayed"], replay["batches_matched"])
        self.assertGreater(replay["batches_replayed"], 0)

        fork = json.loads(
            (self.artifacts / "reports" / "fork_verification.json").read_text("utf-8")
        )
        self.assertTrue(fork["passed"])
        self.assertNotEqual(fork["child_branch_id"], fork["parent_branch_id"])
        self.assertIsNotNone(fork["divergence_step"])

    def test_metrics_reports_are_recomputable_from_the_artifacts(self) -> None:
        """A grader must be able to redo the arithmetic from the generated files."""
        packing = json.loads(
            (self.artifacts / "reports" / "packing_utilization.json").read_text("utf-8")
        )
        rows = packing["batches"]
        self.assertEqual(len(rows), packing["batches_measured"])
        self.assertEqual(
            sum(row["capacity"] for row in rows), packing["total_capacity"]
        )
        self.assertAlmostEqual(
            packing["total_useful_tokens"] / packing["total_capacity"],
            packing["utilization"],
            places=5,
        )
        # Both packing policies were exercised by the run.
        self.assertGreaterEqual(len(packing["by_packing_policy"]), 2)

        throughput = json.loads(
            (self.artifacts / "reports" / "throughput.json").read_text("utf-8")
        )
        self.assertEqual(throughput["steps_without_timings"], [])
        self.assertAlmostEqual(
            sum(row["wall_seconds"] for row in throughput["steps"]),
            throughput["total_wall_seconds"],
            places=4,
        )
        self.assertAlmostEqual(
            throughput["total_loss_bearing_tokens"] / throughput["total_wall_seconds"],
            throughput["loss_bearing_tokens_per_second"],
            places=1,
        )
        self.assertLessEqual(
            throughput["loss_bearing_tokens_per_second"],
            throughput["raw_tokens_per_second"],
        )

    def test_firewall_blocks_a_contaminated_document_in_the_run(self) -> None:
        """P4-T04's demo criterion: a real block, on generated output."""
        blocks = [
            json.loads(line)
            for line in (self.artifacts / "run.log").read_text("utf-8").splitlines()
            if line.strip() and json.loads(line).get("event_type") == "firewall_block"
        ]
        self.assertTrue(blocks, "demo run produced no firewall block")

        # The corpus row that trips it is admitted and not never_train: its metadata says
        # clean, so only the firewall's content check can catch it.
        reasons = {reason for block in blocks for reason in block["reasons"]}
        self.assertIn("canary_string_match", reasons)

        ledger = (self.artifacts / "ledgers" / "consumption.jsonl").read_text("utf-8")
        self.assertNotIn(
            "doc-web-contaminated-001",
            ledger,
            "a firewall-blocked document reached the consumption ledger",
        )

    def test_run_log_covers_every_event_type_scope_requires(self) -> None:
        """P11-T01, checked on the generated log rather than on the emitting code."""
        events = load_run_log(self.artifacts / "run.log")
        self.assertEqual(missing_event_types(events), ())

        # The ordering the log claims must hold: the run opens, crashes, then resumes.
        positions = {
            name: events_of_type(events, name)[0].seq
            for name in ("run_start", "simulated_crash", "resume_initiated", "run_complete")
        }
        self.assertLess(positions["run_start"], positions["simulated_crash"])
        self.assertLess(positions["simulated_crash"], positions["resume_initiated"])
        self.assertEqual(positions["run_complete"], events[-1].seq)

    def test_run_log_batch_commits_match_the_consumption_ledger(self) -> None:
        """The log is a second, independent record of what the run consumed."""
        events = events_of_type(load_run_log(self.artifacts / "run.log"), "batch_committed")
        ledger = load_consumption_ledger(self.artifacts / "ledgers" / "consumption.jsonl")

        self.assertEqual(len(events), len(ledger))
        self.assertEqual(
            [
                (event.fields["attempt"], event.fields["ledger_offset"], event.fields["batch_content_hash"])
                for event in events
            ],
            [(row.attempt, row.ledger_offset, row.batch_content_hash) for row in ledger],
        )

    def test_run_log_verification_events_agree_with_the_reports(self) -> None:
        """A `verification_result` may not claim a pass the report does not record."""
        events = events_of_type(
            load_run_log(self.artifacts / "run.log"), "verification_result"
        )
        self.assertEqual(
            {event.fields["verification"] for event in events},
            {"resume", "replay", "fork"},
        )
        for event in events:
            with self.subTest(verification=event.fields["verification"]):
                report = json.loads(
                    (self.artifacts / event.fields["report_path"]).read_text("utf-8")
                )
                self.assertEqual(event.fields["passed"], report["passed"])

    def test_opus_audit_shows_every_decision_type(self) -> None:
        """P5-T04's deferred demo criterion, checked on generated output."""
        lines = (
            (self.artifacts / "ledgers" / "opus_audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        decisions = {json.loads(line)["decision"] for line in lines if line.strip()}
        self.assertEqual(
            decisions,
            {"accepted", "rejected", "deferred", "protected_override"},
            f"demo run did not exercise every OPUS decision: {sorted(decisions)}",
        )


class TestDemoResultGate(unittest.TestCase):
    """The runner's exit code must follow the phases, not the absence of exceptions."""

    def test_one_failed_phase_fails_the_demo(self) -> None:
        result = DemoResult(artifacts_dir=Path("."))
        result.record(PhaseResult(number=1, name="ok", passed=True, detail=""))
        self.assertTrue(result.passed)

        result.record(PhaseResult(number=2, name="broken", passed=False, detail=""))
        self.assertFalse(result.passed)

    def test_empty_result_is_not_a_pass(self) -> None:
        self.assertFalse(DemoResult(artifacts_dir=Path(".")).passed)


if __name__ == "__main__":
    unittest.main()
