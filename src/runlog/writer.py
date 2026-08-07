"""Append-only `run.log` writer (P11-T01).

One writer owns one file. That is not an implementation detail: `seq` is assigned from
an in-memory counter, so two writers open on the same path would both hand out the same
numbers and the log would stop being an ordered stream. The demo therefore creates one
writer for the run and passes it into the trainer and the recovery phases; a fork gets
its own writer because it writes to its own branch directory.

`open` continues the numbering of an existing file, so a resumed run extends the
crashed run's log rather than restarting at zero.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import RunLogEvent, validate_event


def _utc_now() -> str:
    """Millisecond UTC stamp, e.g. `2026-08-07T09:41:02.318Z`."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class RunLogWriter:
    """Writes one JSON event per line, in emission order."""

    def __init__(self, path: Path, *, next_sequence: int = 0) -> None:
        self.path = Path(path).resolve()
        self._next_sequence = next_sequence

    @classmethod
    def open(cls, path: Path) -> RunLogWriter:
        """Open (or create) a log, continuing after whatever it already holds."""
        target = Path(path).resolve()
        next_sequence = 0
        if target.is_file():
            next_sequence = sum(
                1
                for line in target.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        return cls(target, next_sequence=next_sequence)

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def emit(self, event_type: str, **fields: Any) -> RunLogEvent:
        """Append one event and return it.

        Flushed per line rather than buffered: a crashed run's log has to survive the
        crash, and the `simulated_crash` event is the last thing written before the
        process dies.
        """
        validate_event(event_type, fields)
        event = RunLogEvent(
            seq=self._next_sequence,
            ts=_utc_now(),
            event_type=event_type,
            fields=dict(fields),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.to_log_line())
            handle.write("\n")
        self._next_sequence += 1
        return event
