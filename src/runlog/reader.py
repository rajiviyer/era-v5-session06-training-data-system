"""Read and audit a generated `run.log` (P11-T01).

The demo and the evidence collector both need to answer "did the run actually log what
SCOPE.md §9.1 requires?", and neither may answer it from what the code *would* have
written. Everything here reads the finished file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .types import EVENT_TYPES, RunLogEvent, RunLogError


def load_run_log(path: Path) -> tuple[RunLogEvent, ...]:
    """Parse `run.log`, enforcing that sequence numbers strictly increase.

    A duplicate or out-of-order `seq` means two writers were open on the file at once,
    which would make the ordering claims in the log (crash before resume, commit before
    checkpoint) unreliable. That is worth failing on rather than reporting.
    """
    target = Path(path).resolve()
    if not target.is_file():
        raise RunLogError(f"run.log not found: {target}")

    events: list[RunLogEvent] = []
    previous = -1
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunLogError(f"{target}:{line_number} is not valid JSON: {error}") from error
        event = RunLogEvent.from_dict(payload)
        if event.seq <= previous:
            raise RunLogError(
                f"{target}:{line_number} has seq {event.seq} after {previous}; "
                "run.log sequence numbers must strictly increase"
            )
        previous = event.seq
        events.append(event)
    return tuple(events)


def events_of_type(
    events: Sequence[RunLogEvent],
    event_type: str,
) -> tuple[RunLogEvent, ...]:
    return tuple(event for event in events if event.event_type == event_type)


def event_type_counts(events: Sequence[RunLogEvent]) -> dict[str, int]:
    """How many of each event type the run logged, in SCOPE.md §9.1 order."""
    counts = Counter(event.event_type for event in events)
    return {name: counts.get(name, 0) for name in EVENT_TYPES}


def missing_event_types(events: Sequence[RunLogEvent]) -> tuple[str, ...]:
    """Event types SCOPE.md §9.1 requires that this log never recorded."""
    present = {event.event_type for event in events}
    return tuple(name for name in EVENT_TYPES if name not in present)
