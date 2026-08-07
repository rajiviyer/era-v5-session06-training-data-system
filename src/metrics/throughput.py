"""Throughput: raw and useful loss-bearing tokens per second (P10-T03, P10-T04).

The report is a **join of two independently generated files**:

- token counts come from `consumption.jsonl`, by rebuilding each batch and counting
- seconds come from `reports/step_timings.jsonl`, written as the run executed

Neither file records a rate, so the reported numbers cannot be stale or invented: a
reader recomputes them from the same two sources. That separation is the point. If
throughput were recorded during the run, there would be nothing to check it against.

**Useful loss-bearing tokens/sec is the honest number.** Raw tokens/sec counts every
non-pad token the model saw, including positions masked out of the loss. Only the
loss-bearing ones moved a gradient, so a run can look fast on raw tokens while learning
from a fraction of them.

Steps with no timing row are excluded and reported as `steps_without_timings` rather
than being silently treated as instantaneous, which would inflate the rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .packing import BatchUtilization
from .timing import StepTiming

THROUGHPUT_REPORT_FILENAME = "throughput.json"


@dataclass(frozen=True)
class StepThroughput:
    """Tokens and seconds for one (attempt, step)."""

    attempt: int
    global_step: int
    wall_seconds: float
    raw_tokens: int
    loss_bearing_tokens: int
    microbatches: int

    @property
    def raw_tokens_per_second(self) -> float:
        return self.raw_tokens / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def loss_bearing_tokens_per_second(self) -> float:
        return (
            self.loss_bearing_tokens / self.wall_seconds if self.wall_seconds > 0 else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "global_step": self.global_step,
            "wall_seconds": round(self.wall_seconds, 6),
            "raw_tokens": self.raw_tokens,
            "loss_bearing_tokens": self.loss_bearing_tokens,
            "microbatches": self.microbatches,
            "raw_tokens_per_second": round(self.raw_tokens_per_second, 3),
            "loss_bearing_tokens_per_second": round(
                self.loss_bearing_tokens_per_second, 3
            ),
        }


@dataclass(frozen=True)
class ThroughputReport:
    """Run-level throughput assembled from ledger tokens and recorded seconds."""

    run_id: str
    steps: tuple[StepThroughput, ...]
    steps_without_timings: tuple[tuple[int, int], ...]
    """`(attempt, global_step)` pairs that committed batches but recorded no duration."""

    @property
    def total_wall_seconds(self) -> float:
        return sum(step.wall_seconds for step in self.steps)

    @property
    def total_raw_tokens(self) -> int:
        return sum(step.raw_tokens for step in self.steps)

    @property
    def total_loss_bearing_tokens(self) -> int:
        return sum(step.loss_bearing_tokens for step in self.steps)

    @property
    def raw_tokens_per_second(self) -> float:
        seconds = self.total_wall_seconds
        return self.total_raw_tokens / seconds if seconds > 0 else 0.0

    @property
    def loss_bearing_tokens_per_second(self) -> float:
        seconds = self.total_wall_seconds
        return self.total_loss_bearing_tokens / seconds if seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sources": {
                "tokens": "ledgers/consumption.jsonl (batches rebuilt and re-counted)",
                "seconds": "reports/step_timings.jsonl",
            },
            "steps_measured": len(self.steps),
            "steps_without_timings": [list(pair) for pair in self.steps_without_timings],
            "total_wall_seconds": round(self.total_wall_seconds, 6),
            "total_raw_tokens": self.total_raw_tokens,
            "total_loss_bearing_tokens": self.total_loss_bearing_tokens,
            "raw_tokens_per_second": round(self.raw_tokens_per_second, 3),
            "loss_bearing_tokens_per_second": round(
                self.loss_bearing_tokens_per_second, 3
            ),
            "steps": [step.to_dict() for step in self.steps],
        }


def compute_throughput(
    batches: Sequence[BatchUtilization],
    timings: Sequence[StepTiming],
) -> ThroughputReport:
    """Join per-batch token counts with per-step durations."""
    if not batches:
        raise ValueError("cannot compute throughput without measured batches")

    seconds_by_step = {(row.attempt, row.global_step): row.wall_seconds for row in timings}

    totals: dict[tuple[int, int], list[int]] = {}
    for batch in batches:
        key = (batch.attempt, batch.global_step)
        entry = totals.setdefault(key, [0, 0, 0])
        entry[0] += batch.useful_tokens
        entry[1] += batch.loss_bearing_tokens
        entry[2] += 1

    steps: list[StepThroughput] = []
    missing: list[tuple[int, int]] = []
    for key in sorted(totals):
        wall_seconds = seconds_by_step.get(key)
        if wall_seconds is None:
            missing.append(key)
            continue
        raw_tokens, loss_tokens, microbatches = totals[key]
        steps.append(
            StepThroughput(
                attempt=key[0],
                global_step=key[1],
                wall_seconds=wall_seconds,
                raw_tokens=raw_tokens,
                loss_bearing_tokens=loss_tokens,
                microbatches=microbatches,
            )
        )

    run_id = "unknown"
    if timings:
        run_id = timings[0].run_id

    return ThroughputReport(
        run_id=run_id,
        steps=tuple(steps),
        steps_without_timings=tuple(missing),
    )


def write_throughput_report(reports_dir: Path, report: ThroughputReport) -> Path:
    """Write `reports/throughput.json` (P10-T04)."""
    from shards.io import write_json_atomic

    target = Path(reports_dir).resolve() / THROUGHPUT_REPORT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, report.to_dict())
    return target
