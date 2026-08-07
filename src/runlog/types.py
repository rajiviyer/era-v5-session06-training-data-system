"""Run log event vocabulary and record type (P11-T01).

SCOPE.md §9.1 lists the events a run must log. That list is encoded here as
`EVENT_TYPES` rather than left implicit at each call site, for two reasons:

- A misspelled `event_type` would otherwise create a phantom event that no reader ever
  finds, and the log would look complete while a required event was silently absent.
- The evidence collector (P11-T04) checks a finished `run.log` against this exact set,
  so the requirement and the check read from one definition.

Every line is one JSON object: a three-field envelope (`seq`, `ts`, `event_type`) plus
the event's own fields flattened alongside it. Flat lines keep the log greppable
(`event_type` is at the top level of every record) and let a reader load the file with
one `json.loads` per line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

RUN_LOG_FILENAME = "run.log"

EVENT_TYPES: tuple[str, ...] = (
    "run_start",
    "shard_admitted",
    "shard_blocked",
    "stage_transition",
    "opus_decision",
    "firewall_block",
    "batch_committed",
    "checkpoint_saved",
    "simulated_crash",
    "resume_initiated",
    "replay_initiated",
    "fork_initiated",
    "verification_result",
    "run_complete",
)

_ENVELOPE_KEYS = ("seq", "ts", "event_type")


class RunLogError(ValueError):
    """Raised for an unknown event type, a bad envelope, or an out-of-order log."""


@dataclass(frozen=True)
class RunLogEvent:
    """One line of `run.log`.

    `seq` orders the stream across process boundaries: a crash, a resume, and a fork all
    append to the same file, and wall-clock timestamps alone would not prove which came
    first at millisecond resolution.
    """

    seq: int
    ts: str
    event_type: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "event_type": self.event_type,
            **self.fields,
        }

    def to_log_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunLogEvent:
        missing = [key for key in _ENVELOPE_KEYS if key not in payload]
        if missing:
            raise RunLogError(f"run.log record is missing envelope keys {missing}: {payload!r}")
        return cls(
            seq=int(payload["seq"]),
            ts=str(payload["ts"]),
            event_type=str(payload["event_type"]),
            fields={
                key: value for key, value in payload.items() if key not in _ENVELOPE_KEYS
            },
        )


def validate_event(event_type: str, fields: Mapping[str, Any]) -> None:
    """Reject an unknown event type or a payload that would shadow the envelope."""
    if event_type not in EVENT_TYPES:
        raise RunLogError(
            f"unknown run.log event_type {event_type!r}; "
            f"SCOPE.md §9.1 defines {list(EVENT_TYPES)}"
        )
    collisions = sorted(set(_ENVELOPE_KEYS).intersection(fields))
    if collisions:
        raise RunLogError(
            f"{event_type} payload may not set envelope keys {collisions}"
        )
