"""Consumption ledger reader and step reconstruction (P6-T04).

The ledger can hold more than one attempt (see `ConsumptionLedgerEvent.attempt`). Rows
from a crashed attempt are kept for audit, so every lookup here answers from the
**newest** attempt that covers the requested position unless a caller asks for a
specific one. That keeps "reconstruct step N" meaning "reconstruct what the model is
actually carrying", not "reconstruct a batch whose weight update was rolled back".
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import LedgerError
from .schema import event_from_dict
from .types import ConsumptionLedgerEvent


def load_consumption_ledger(path: Path) -> tuple[ConsumptionLedgerEvent, ...]:
    """Load all consumption ledger events from disk."""
    target = path.resolve()
    if not target.is_file():
        return ()

    records: list[ConsumptionLedgerEvent] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"invalid JSON in {target} line {line_number}") from exc
        if not isinstance(payload, dict):
            raise LedgerError(f"ledger line {line_number} must be a JSON object")
        records.append(event_from_dict(payload))

    _validate_ledger_ordering(records)
    return tuple(records)


def latest_attempt(records: tuple[ConsumptionLedgerEvent, ...]) -> int:
    """Highest attempt number present in the ledger."""
    if not records:
        raise LedgerError("ledger is empty")
    return max(record.attempt for record in records)


def events_for_attempt(
    records: tuple[ConsumptionLedgerEvent, ...],
    attempt: int,
) -> tuple[ConsumptionLedgerEvent, ...]:
    """Return only the rows one attempt wrote."""
    return tuple(record for record in records if record.attempt == attempt)


def get_event_at_offset(
    records: tuple[ConsumptionLedgerEvent, ...],
    ledger_offset: int,
    *,
    attempt: int | None = None,
) -> ConsumptionLedgerEvent:
    """Return the ledger event at one offset, from the newest attempt holding it."""
    matched = [record for record in records if record.ledger_offset == ledger_offset]
    if attempt is not None:
        matched = [record for record in matched if record.attempt == attempt]
    if not matched:
        raise LedgerError(
            f"ledger_offset {ledger_offset} not found"
            + ("" if attempt is None else f" in attempt {attempt}")
        )
    return max(matched, key=lambda record: record.attempt)


def get_events_for_global_step(
    records: tuple[ConsumptionLedgerEvent, ...],
    global_step: int,
    *,
    attempt: int | None = None,
) -> tuple[ConsumptionLedgerEvent, ...]:
    """Return committed microbatches for one global step, newest attempt by default."""
    matched = [record for record in records if record.global_step == global_step]
    if not matched:
        raise LedgerError(f"global_step {global_step} not found in ledger")
    target = attempt if attempt is not None else max(r.attempt for r in matched)
    selected = tuple(record for record in matched if record.attempt == target)
    if not selected:
        raise LedgerError(f"global_step {global_step} not found in attempt {target}")
    return selected


def reconstruct_at_global_step(
    path: Path,
    global_step: int,
    *,
    attempt: int | None = None,
) -> tuple[ConsumptionLedgerEvent, ...]:
    """Reconstruct committed microbatches for one global training step (P6-T04).

    The one entry point that goes from a path to reconstructed rows. Callers that
    already hold the loaded ledger use `get_events_for_global_step` or
    `get_event_at_offset` directly rather than re-reading the file.
    """
    records = load_consumption_ledger(path)
    return get_events_for_global_step(records, global_step, attempt=attempt)


def _validate_ledger_ordering(records: list[ConsumptionLedgerEvent]) -> None:
    """Append-only ordering across attempts.

    Within an attempt offsets still increment by exactly one. Across attempts the number
    must never go backwards, and a resumed attempt must not start *beyond* the previous
    tail: that would mean batches between the two were never consumed by anyone, which is
    the skipped-batch failure the assignment fails a run for.
    """
    previous: ConsumptionLedgerEvent | None = None
    attempt_first_offset: dict[int, int] = {}

    for index, record in enumerate(records):
        if previous is None:
            attempt_first_offset[record.attempt] = record.ledger_offset
            previous = record
            continue

        if record.attempt < previous.attempt:
            raise LedgerError(
                f"ledger row {index}: attempt {record.attempt} follows "
                f"attempt {previous.attempt}; the ledger is append-only"
            )

        if record.attempt == previous.attempt:
            if record.ledger_offset != previous.ledger_offset + 1:
                raise LedgerError(
                    f"ledger row {index}: expected ledger_offset "
                    f"{previous.ledger_offset + 1} in attempt {record.attempt}, "
                    f"got {record.ledger_offset}"
                )
        else:
            if record.attempt in attempt_first_offset:
                raise LedgerError(
                    f"ledger row {index}: attempt {record.attempt} resumes after it "
                    "already ended; attempts must be contiguous"
                )
            if record.ledger_offset > previous.ledger_offset + 1:
                raise LedgerError(
                    f"ledger row {index}: attempt {record.attempt} starts at offset "
                    f"{record.ledger_offset}, past the previous tail "
                    f"{previous.ledger_offset}; batches would be skipped"
                )
            attempt_first_offset[record.attempt] = record.ledger_offset

        previous = record

    first = records[0] if records else None
    if first is not None and first.ledger_offset != 0:
        raise LedgerError(
            f"ledger must start at offset 0, got {first.ledger_offset}"
        )
