"""Per-step wall-clock timings (P10-T03).

Timings live in their own artifact, not in the consumption ledger. The ledger records
*what was consumed*, which is reproducible; a duration is a property of the machine that
ran it and changes on every run. Mixing them would put a non-reproducible field into the
record that resume and replay verify against.

Keeping them separate is also what makes the throughput report checkable: token counts
come from the ledger, seconds come from here, and the reported rate is the join of two
independently generated files.

Rows carry `attempt` because a crash-resume re-runs the same steps. Both passes really
happened and both really cost wall time, so throughput counts both.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMINGS_FILENAME = "step_timings.jsonl"


@dataclass(frozen=True)
class StepTiming:
    """Wall time for one global step, including gated microbatches it discarded."""

    run_id: str
    branch_id: str
    attempt: int
    global_step: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "attempt": self.attempt,
            "global_step": self.global_step,
            # Microsecond resolution is far finer than a CPU step needs and keeps the
            # file stable to read.
            "wall_seconds": round(self.wall_seconds, 6),
        }


class StepClock:
    """Measures one global step, gating and all."""

    def __init__(self) -> None:
        self._started: float | None = None

    def start(self) -> None:
        self._started = time.perf_counter()

    def stop(self) -> float:
        if self._started is None:
            raise RuntimeError("StepClock.stop() called before start()")
        elapsed = time.perf_counter() - self._started
        self._started = None
        return elapsed


def append_step_timing(path: Path, timing: StepTiming) -> StepTiming:
    """Append one timing row, so a crashed run still leaves the steps it completed."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(timing.to_dict(), sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    return timing


def load_step_timings(path: Path) -> tuple[StepTiming, ...]:
    """Load all recorded step timings."""
    target = Path(path).resolve()
    if not target.is_file():
        return ()

    rows: list[StepTiming] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {target} line {line_number}") from exc
        rows.append(
            StepTiming(
                run_id=str(payload["run_id"]),
                branch_id=str(payload["branch_id"]),
                attempt=int(payload.get("attempt", 0)),
                global_step=int(payload["global_step"]),
                wall_seconds=float(payload["wall_seconds"]),
            )
        )
    return tuple(rows)
