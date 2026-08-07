"""Append-only consumption ledger writer (P6-T02)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .errors import LedgerError
from .schema import validate_ledger_event
from .types import ConsumptionLedgerEvent


def append_ledger_event(path: Path, event: ConsumptionLedgerEvent) -> None:
    """Append one validated ledger event to consumption.jsonl."""
    validated = validate_ledger_event(event)
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(validated.to_dict(), sort_keys=True, separators=(",", ":"))
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


class LedgerWriter:
    """Assign monotonic ledger_offset values and append events.

    The writer owns both lineage coordinates, `attempt` and `ledger_offset`, so callers
    cannot set one without the other and drift apart.
    """

    def __init__(self, path: Path, *, next_offset: int, attempt: int = 0) -> None:
        self.path = path.resolve()
        self._next_offset = next_offset
        self._attempt = attempt

    @classmethod
    def open(cls, path: Path) -> LedgerWriter:
        """Open or create a ledger writer, continuing the newest attempt."""
        from .reader import load_consumption_ledger

        records = load_consumption_ledger(path)
        if not records:
            return cls(path, next_offset=0, attempt=0)
        return cls(
            path,
            next_offset=records[-1].ledger_offset + 1,
            attempt=records[-1].attempt,
        )

    @classmethod
    def resume_at(cls, path: Path, *, ledger_offset: int) -> LedgerWriter:
        """Open a writer for a **new attempt** starting at `ledger_offset` (P9-T02).

        Resume rewinds the data cursor to a checkpoint that sits behind the ledger tail.
        The rows in between are not deleted; the new attempt re-commits those offsets
        under a higher attempt number, so both the original and the re-derived batch stay
        on disk and can be compared.
        """
        from .reader import load_consumption_ledger

        records = load_consumption_ledger(path)
        if not records:
            raise LedgerError(f"cannot resume an empty ledger: {path}")
        tail = records[-1]
        if ledger_offset > tail.ledger_offset + 1:
            raise LedgerError(
                f"resume offset {ledger_offset} is past the ledger tail "
                f"{tail.ledger_offset}; batches would be skipped"
            )
        return cls(path, next_offset=ledger_offset, attempt=tail.attempt + 1)

    @property
    def next_offset(self) -> int:
        return self._next_offset

    @property
    def attempt(self) -> int:
        return self._attempt

    def append(self, event: ConsumptionLedgerEvent) -> ConsumptionLedgerEvent:
        """Append one event, stamped with this writer's attempt and next offset."""
        if event.ledger_offset != self._next_offset:
            raise LedgerError(
                f"expected ledger_offset {self._next_offset}, got {event.ledger_offset}"
            )
        stamped = replace(event, attempt=self._attempt)
        append_ledger_event(self.path, stamped)
        self._next_offset += 1
        return stamped
