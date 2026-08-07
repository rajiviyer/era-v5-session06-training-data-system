"""Packing utilization, recomputed from the ledger (P10-T01, P10-T02).

    utilization = useful_tokens / (seq_len * num_sequences)

`useful_tokens` counts non-pad positions; the denominator is the slot capacity the batch
occupied whether or not it was filled. A run that pads heavily is paying for compute it
throws away, and this is the number that says so.

**Nothing here is read back from a recorded metric.** Every batch is rebuilt from its
ledger row (same path replay uses) and re-counted, so the report is reproducible from
`consumption.jsonl` plus the corpus and tokenizer. That is the assignment's bar: a
packing claim that cannot be reconstructed earns no credit.

**Every attempt counts.** Unlike the learning aggregates, which drop rolled-back work,
utilization describes batches the machine actually built. A crashed attempt's batches
were built and padded exactly like any other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

from ledger.rebuild import rebuild_batch
from ledger.types import ConsumptionLedgerEvent

if TYPE_CHECKING:
    from tokenizer.frozen import FrozenTokenizer

PACKING_REPORT_FILENAME = "packing_utilization.json"
UTILIZATION_FORMULA = "useful_tokens / (seq_len * num_sequences)"


@dataclass(frozen=True)
class BatchUtilization:
    """Slot occupancy for one committed microbatch."""

    attempt: int
    global_step: int
    microbatch_id: str
    ledger_offset: int
    packing_policy: str
    num_sequences: int
    seq_len: int
    useful_tokens: int
    loss_bearing_tokens: int

    @property
    def capacity(self) -> int:
        return self.num_sequences * self.seq_len

    @property
    def utilization(self) -> float:
        return self.useful_tokens / self.capacity if self.capacity else 0.0

    @property
    def loss_bearing_fraction(self) -> float:
        """Share of capacity that actually produced gradient."""
        return self.loss_bearing_tokens / self.capacity if self.capacity else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "global_step": self.global_step,
            "microbatch_id": self.microbatch_id,
            "ledger_offset": self.ledger_offset,
            "packing_policy": self.packing_policy,
            "num_sequences": self.num_sequences,
            "seq_len": self.seq_len,
            "capacity": self.capacity,
            "useful_tokens": self.useful_tokens,
            "loss_bearing_tokens": self.loss_bearing_tokens,
            "utilization": round(self.utilization, 6),
            "loss_bearing_fraction": round(self.loss_bearing_fraction, 6),
        }


@dataclass(frozen=True)
class PackingReport:
    """Run-level utilization, with a per-policy split."""

    run_id: str
    formula: str
    batches: tuple[BatchUtilization, ...]

    @property
    def total_capacity(self) -> int:
        return sum(batch.capacity for batch in self.batches)

    @property
    def total_useful_tokens(self) -> int:
        return sum(batch.useful_tokens for batch in self.batches)

    @property
    def total_loss_bearing_tokens(self) -> int:
        return sum(batch.loss_bearing_tokens for batch in self.batches)

    @property
    def utilization(self) -> float:
        """Token-weighted, not the mean of per-batch ratios: batches differ in size."""
        return self.total_useful_tokens / self.total_capacity if self.total_capacity else 0.0

    def by_policy(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[BatchUtilization]] = {}
        for batch in self.batches:
            grouped.setdefault(batch.packing_policy, []).append(batch)

        summary: dict[str, dict[str, Any]] = {}
        for policy in sorted(grouped):
            entries = grouped[policy]
            capacity = sum(entry.capacity for entry in entries)
            useful = sum(entry.useful_tokens for entry in entries)
            summary[policy] = {
                "batches": len(entries),
                "capacity": capacity,
                "useful_tokens": useful,
                "utilization": round(useful / capacity, 6) if capacity else 0.0,
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "formula": self.formula,
            "batches_measured": len(self.batches),
            "total_capacity": self.total_capacity,
            "total_useful_tokens": self.total_useful_tokens,
            "total_loss_bearing_tokens": self.total_loss_bearing_tokens,
            "utilization": round(self.utilization, 6),
            "loss_bearing_fraction": (
                round(self.total_loss_bearing_tokens / self.total_capacity, 6)
                if self.total_capacity
                else 0.0
            ),
            "by_packing_policy": self.by_policy(),
            "batches": [batch.to_dict() for batch in self.batches],
        }


def compute_packing_utilization(
    records: Sequence[ConsumptionLedgerEvent],
    *,
    documents_by_id: dict[str, dict[str, Any]],
    tokenizer: FrozenTokenizer,
    seq_len: int,
) -> PackingReport:
    """Rebuild every committed batch and measure how much of its capacity was real."""
    if not records:
        raise ValueError("cannot compute packing utilization from an empty ledger")

    batches: list[BatchUtilization] = []
    for row in records:
        assembled = rebuild_batch(
            row,
            documents_by_id=documents_by_id,
            tokenizer=tokenizer,
            seq_len=seq_len,
        )
        batch = assembled.batch
        batches.append(
            BatchUtilization(
                attempt=row.attempt,
                global_step=row.global_step,
                microbatch_id=row.microbatch_id,
                ledger_offset=row.ledger_offset,
                packing_policy=assembled.packing_policy,
                num_sequences=batch.batch_size,
                seq_len=batch.seq_len,
                useful_tokens=assembled.useful_tokens,
                loss_bearing_tokens=sum(sum(row_mask) for row_mask in batch.loss_mask),
            )
        )

    return PackingReport(
        run_id=records[0].run_id,
        formula=UTILIZATION_FORMULA,
        batches=tuple(batches),
    )


def write_packing_report(reports_dir: Path, report: PackingReport) -> Path:
    """Write `reports/packing_utilization.json` (P10-T02)."""
    from shards.io import write_json_atomic

    target = Path(reports_dir).resolve() / PACKING_REPORT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, report.to_dict())
    return target
