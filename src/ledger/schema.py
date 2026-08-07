"""Consumption ledger validation (P6-T01)."""

from __future__ import annotations

from .errors import LedgerError
from .types import ATTENTION_POLICY, ConsumptionLedgerEvent, DATALOADER_VERSION, SCHEMA_VERSION


def validate_ledger_event(event: ConsumptionLedgerEvent) -> ConsumptionLedgerEvent:
    """Validate one consumption ledger event."""
    if not event.run_id:
        raise LedgerError("run_id is required")
    if not event.branch_id:
        raise LedgerError("branch_id is required")
    if event.attempt < 0:
        raise LedgerError("attempt must be non-negative")
    if event.global_step < 0:
        raise LedgerError("global_step must be non-negative")
    if event.ledger_offset < 0:
        raise LedgerError("ledger_offset must be non-negative")
    if not event.microbatch_id:
        raise LedgerError("microbatch_id is required")
    if not event.packed_sample_ids:
        raise LedgerError("packed_sample_ids must be non-empty")
    if len(event.shard_ids) != len(event.packed_sample_ids):
        raise LedgerError("shard_ids must align with packed_sample_ids")
    if not event.loss_mask_hash:
        raise LedgerError("loss_mask_hash is required")
    if event.attention_policy != ATTENTION_POLICY:
        raise LedgerError(f"attention_policy must be {ATTENTION_POLICY!r}")
    if not event.position_policy:
        raise LedgerError("position_policy is required")
    if not event.mixture_lane:
        raise LedgerError("mixture_lane is required")
    if not event.curriculum_stage:
        raise LedgerError("curriculum_stage is required")
    if not event.tokenizer_hash.startswith("tok_"):
        raise LedgerError("tokenizer_hash must be a tok_* string")
    if event.dataloader_version != DATALOADER_VERSION:
        raise LedgerError(f"dataloader_version must be {DATALOADER_VERSION!r}")
    if not event.opus_decision_id:
        raise LedgerError("opus_decision_id is required")
    if not event.batch_content_hash:
        raise LedgerError("batch_content_hash is required")
    if not event.candidate_id:
        raise LedgerError("candidate_id is required")
    return event


def event_from_dict(payload: dict[str, object]) -> ConsumptionLedgerEvent:
    """Parse one ledger event from JSON."""
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise LedgerError(f"unsupported schema_version: {version}")

    required = (
        "run_id",
        "branch_id",
        "global_step",
        "ledger_offset",
        "microbatch_id",
        "packed_sample_ids",
        "shard_ids",
        "token_span_ids",
        "loss_mask_hash",
        "attention_policy",
        "position_policy",
        "mixture_lane",
        "curriculum_stage",
        "tokenizer_hash",
        "dataloader_version",
        "opus_decision_id",
        "batch_content_hash",
        "candidate_id",
    )
    for key in required:
        if key not in payload:
            raise LedgerError(f"consumption ledger event missing key: {key}")

    checkpoint_id = payload.get("checkpoint_id")
    if checkpoint_id is not None:
        checkpoint_id = str(checkpoint_id)

    # A row written before the field existed is attempt 0, which is exactly what it was:
    # the first, uninterrupted pass. No migration needed.
    attempt = int(payload.get("attempt", 0))  # type: ignore[arg-type]

    event = ConsumptionLedgerEvent(
        run_id=str(payload["run_id"]),
        branch_id=str(payload["branch_id"]),
        global_step=int(payload["global_step"]),
        ledger_offset=int(payload["ledger_offset"]),
        checkpoint_id=checkpoint_id,
        microbatch_id=str(payload["microbatch_id"]),
        packed_sample_ids=tuple(str(item) for item in payload["packed_sample_ids"]),
        shard_ids=tuple(str(item) for item in payload["shard_ids"]),
        token_span_ids=tuple(str(item) for item in payload["token_span_ids"]),
        loss_mask_hash=str(payload["loss_mask_hash"]),
        attention_policy=str(payload["attention_policy"]),
        position_policy=str(payload["position_policy"]),
        mixture_lane=str(payload["mixture_lane"]),
        curriculum_stage=str(payload["curriculum_stage"]),
        tokenizer_hash=str(payload["tokenizer_hash"]),
        dataloader_version=str(payload["dataloader_version"]),
        opus_decision_id=str(payload["opus_decision_id"]),
        batch_content_hash=str(payload["batch_content_hash"]),
        candidate_id=str(payload["candidate_id"]),
        attempt=attempt,
    )
    return validate_ledger_event(event)
