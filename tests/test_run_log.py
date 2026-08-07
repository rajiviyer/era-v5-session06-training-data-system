"""Structured run.log writer and reader (P11-T01)."""

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

from runlog import (  # noqa: E402
    EVENT_TYPES,
    RunLogError,
    RunLogWriter,
    event_type_counts,
    events_of_type,
    load_run_log,
    missing_event_types,
)


class TestRunLogWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.path = self.temp_dir / "run.log"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_every_line_is_one_json_event_with_the_envelope(self) -> None:
        writer = RunLogWriter.open(self.path)
        writer.emit("run_start", run_id="s6-demo", branch_id="run-a")
        writer.emit("run_complete", run_id="s6-demo", branch_id="run-a")

        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        for expected_seq, line in enumerate(lines):
            payload = json.loads(line)
            self.assertEqual(payload["seq"], expected_seq)
            self.assertIn("ts", payload)
            self.assertIn("event_type", payload)
            self.assertEqual(payload["run_id"], "s6-demo")

    def test_unknown_event_type_is_rejected(self) -> None:
        """A typo must fail loudly, not create an event no reader looks for."""
        writer = RunLogWriter.open(self.path)
        with self.assertRaises(RunLogError):
            writer.emit("batch_commited", global_step=1)
        self.assertFalse(self.path.exists())

    def test_payload_may_not_shadow_the_envelope(self) -> None:
        """A field named `seq` or `ts` would overwrite the ordering the reader relies on."""
        writer = RunLogWriter.open(self.path)
        with self.assertRaises(RunLogError):
            writer.emit("run_start", seq=99)
        with self.assertRaises(RunLogError):
            writer.emit("run_start", ts="whenever")

    def test_reopening_continues_the_sequence(self) -> None:
        """Resume and fork extend the crashed run's log instead of restarting at zero."""
        first = RunLogWriter.open(self.path)
        first.emit("run_start", run_id="s6-demo")
        first.emit("simulated_crash", global_step=25)

        second = RunLogWriter.open(self.path)
        self.assertEqual(second.next_sequence, 2)
        second.emit("resume_initiated", resume_from_step=20)

        events = load_run_log(self.path)
        self.assertEqual([event.seq for event in events], [0, 1, 2])
        self.assertEqual(events[-1].event_type, "resume_initiated")


class TestRunLogReader(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.path = self.temp_dir / "run.log"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_duplicate_sequence_numbers_are_an_error(self) -> None:
        """Two writers open on one file would break every ordering claim in the log."""
        writer_a = RunLogWriter.open(self.path)
        writer_b = RunLogWriter.open(self.path)
        writer_a.emit("run_start", run_id="s6-demo")
        writer_b.emit("run_complete", run_id="s6-demo")

        with self.assertRaises(RunLogError):
            load_run_log(self.path)

    def test_missing_event_types_reports_the_scope_vocabulary_gap(self) -> None:
        writer = RunLogWriter.open(self.path)
        writer.emit("run_start", run_id="s6-demo")
        writer.emit("firewall_block", candidate_id="cand-1")
        writer.emit("firewall_block", candidate_id="cand-2")

        events = load_run_log(self.path)
        self.assertEqual(len(events_of_type(events, "firewall_block")), 2)
        self.assertEqual(event_type_counts(events)["firewall_block"], 2)
        self.assertEqual(event_type_counts(events)["batch_committed"], 0)

        missing = missing_event_types(events)
        self.assertIn("batch_committed", missing)
        self.assertNotIn("run_start", missing)
        self.assertEqual(len(missing), len(EVENT_TYPES) - 2)

    def test_a_missing_log_is_an_error_not_an_empty_run(self) -> None:
        with self.assertRaises(RunLogError):
            load_run_log(self.temp_dir / "never_written.log")


if __name__ == "__main__":
    unittest.main()
