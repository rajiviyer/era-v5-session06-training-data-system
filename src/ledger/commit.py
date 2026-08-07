"""Build and commit consumption ledger events (P6-T03)."""

from __future__ import annotations

from batch.types import Batch
from opus.types import BatchPipelineResult

from .types import ATTENTION_POLICY, ConsumptionLedgerEvent, DATALOADER_VERSION
from .writer import LedgerWriter


def flatten_token_span_ids(batch: Batch) -> tuple[str, ...]:
    """Serialize span IDs from a batch into ledger token_span_ids."""
    spans: list[str] = []
    for row_index, row in enumerate(batch.span_ids):
        for token_index, span_id in enumerate(row):
            spans.append(f"{row_index}:{token_index}:{span_id}")
    return tuple(spans)


def make_microbatch_id(global_step: int, microbatch_index: int) -> str:
    """Stable microbatch identifier."""
    return f"mb-{global_step:05d}-{microbatch_index}"


def parse_microbatch_id(microbatch_id: str) -> tuple[int, int]:
    """Recover `(global_step, microbatch_index)` from a microbatch identifier.

    Replay positions the dataloader at a historical microbatch, and the ledger row names
    it only by ID. Inverting `make_microbatch_id` keeps that mapping in one place.
    """
    parts = microbatch_id.split("-")
    if len(parts) != 3 or parts[0] != "mb":
        raise ValueError(f"malformed microbatch_id: {microbatch_id!r}")
    try:
        return int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(f"malformed microbatch_id: {microbatch_id!r}") from exc


def build_ledger_event(
    pipeline_result: BatchPipelineResult,
    batch: Batch,
    *,
    run_id: str,
    branch_id: str,
    global_step: int,
    ledger_offset: int,
    curriculum_stage: str,
    mixture_lane: str,
    tokenizer_hash: str,
    microbatch_index: int,
    checkpoint_id: str | None = None,
) -> ConsumptionLedgerEvent:
    """Build one consumption ledger event from a committed batch."""
    if not pipeline_result.committed:
        raise ValueError("build_ledger_event requires a committed pipeline result")
    if pipeline_result.opus is None:
        raise ValueError("build_ledger_event requires an OPUS result")
    if batch is None:
        raise ValueError("build_ledger_event requires a built batch")

    candidate = pipeline_result.candidate
    return ConsumptionLedgerEvent(
        run_id=run_id,
        branch_id=branch_id,
        global_step=global_step,
        ledger_offset=ledger_offset,
        checkpoint_id=checkpoint_id,
        microbatch_id=make_microbatch_id(global_step, microbatch_index),
        packed_sample_ids=candidate.sample_ids,
        shard_ids=candidate.shard_ids,
        token_span_ids=flatten_token_span_ids(batch),
        loss_mask_hash=batch.loss_mask_hash,
        attention_policy=ATTENTION_POLICY,
        position_policy=batch.position_id_policy,
        mixture_lane=mixture_lane,
        curriculum_stage=curriculum_stage,
        tokenizer_hash=tokenizer_hash,
        dataloader_version=DATALOADER_VERSION,
        opus_decision_id=pipeline_result.opus.audit.opus_decision_id,
        batch_content_hash=batch.batch_content_hash,
        candidate_id=candidate.candidate_id,
    )


def commit_batch(
    writer: LedgerWriter,
    pipeline_result: BatchPipelineResult,
    batch: Batch,
    *,
    run_id: str,
    branch_id: str,
    global_step: int,
    curriculum_stage: str,
    mixture_lane: str,
    tokenizer_hash: str,
    microbatch_index: int,
    checkpoint_id: str | None = None,
) -> ConsumptionLedgerEvent:
    """Append one committed batch to the consumption ledger."""
    event = build_ledger_event(
        pipeline_result,
        batch,
        run_id=run_id,
        branch_id=branch_id,
        global_step=global_step,
        ledger_offset=writer.next_offset,
        curriculum_stage=curriculum_stage,
        mixture_lane=mixture_lane,
        tokenizer_hash=tokenizer_hash,
        microbatch_index=microbatch_index,
        checkpoint_id=checkpoint_id,
    )
    return writer.append(event)
