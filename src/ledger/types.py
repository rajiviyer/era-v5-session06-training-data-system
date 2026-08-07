"""Consumption ledger event types (P6-T01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.0"
LEDGER_FILENAME = "consumption.jsonl"
DATALOADER_VERSION = "s6-v1"
ATTENTION_POLICY = "causal"


@dataclass(frozen=True)
class ConsumptionLedgerEvent:
    """One append-only consumption ledger row (SCOPE.md §6.7).

    `attempt` numbers the crash-resume generation that wrote the row. A crash leaves
    committed rows *ahead* of the checkpoint resume restores, and the ledger is
    append-only, so those rows cannot be deleted or overwritten. Instead the resumed run
    writes the same `ledger_offset` values again under the next attempt: uniqueness is
    `(attempt, ledger_offset)`, the superseded rows stay readable as the expected record
    that P9-T03 compares against, and the newest attempt is the live lineage.
    """

    run_id: str
    branch_id: str
    global_step: int
    ledger_offset: int
    checkpoint_id: str | None
    microbatch_id: str
    packed_sample_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    token_span_ids: tuple[str, ...]
    loss_mask_hash: str
    attention_policy: str
    position_policy: str
    mixture_lane: str
    curriculum_stage: str
    tokenizer_hash: str
    dataloader_version: str
    opus_decision_id: str
    batch_content_hash: str
    candidate_id: str
    attempt: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "attempt": self.attempt,
            "global_step": self.global_step,
            "ledger_offset": self.ledger_offset,
            "microbatch_id": self.microbatch_id,
            "packed_sample_ids": list(self.packed_sample_ids),
            "shard_ids": list(self.shard_ids),
            "token_span_ids": list(self.token_span_ids),
            "loss_mask_hash": self.loss_mask_hash,
            "attention_policy": self.attention_policy,
            "position_policy": self.position_policy,
            "mixture_lane": self.mixture_lane,
            "curriculum_stage": self.curriculum_stage,
            "tokenizer_hash": self.tokenizer_hash,
            "dataloader_version": self.dataloader_version,
            "opus_decision_id": self.opus_decision_id,
            "batch_content_hash": self.batch_content_hash,
            "candidate_id": self.candidate_id,
        }
        if self.checkpoint_id is not None:
            payload["checkpoint_id"] = self.checkpoint_id
        return payload


@dataclass(frozen=True)
class DataLoaderState:
    """Ledger-bound dataloader position (P6-T07)."""

    run_id: str
    branch_id: str
    ledger_offset: int
    next_global_step: int
    next_microbatch_index: int
    dataloader_version: str = DATALOADER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "ledger_offset": self.ledger_offset,
            "next_global_step": self.next_global_step,
            "next_microbatch_index": self.next_microbatch_index,
            "dataloader_version": self.dataloader_version,
        }
