"""Build training microbatches from packed sequences (P2-T04–T07)."""

from __future__ import annotations

from packing.types import PackedSequence

from .errors import BatchError
from .hash import batch_content_hash, loss_mask_hash
from .masks import (
    agentic_loss_mask,
    build_position_ids,
    causal_attention_mask,
    pretrain_loss_mask,
)
from .types import Batch, BatchBuildConfig


def build_batch(
    sequences: list[PackedSequence],
    config: BatchBuildConfig,
    *,
    packing_policy: str,
    loss_eligible: list[tuple[bool, ...] | None] | None = None,
) -> Batch:
    """Build one microbatch from packed sequences of equal length."""
    if not sequences:
        raise BatchError("batch requires at least one packed sequence")

    seq_len = len(sequences[0].token_ids)
    if any(len(sequence.token_ids) != seq_len for sequence in sequences):
        raise BatchError("all sequences in a batch must share seq_len")

    policies = {sequence.policy for sequence in sequences}
    if len(policies) != 1:
        raise BatchError("all sequences in a batch must share packing policy")
    if packing_policy not in policies:
        raise BatchError(f"packing_policy {packing_policy!r} does not match sequences")

    if config.mode not in ("pretrain", "agentic"):
        raise BatchError(f"invalid batch mode: {config.mode}")

    if loss_eligible is not None and len(loss_eligible) != len(sequences):
        raise BatchError("loss_eligible length must match number of sequences")

    input_rows: list[tuple[int, ...]] = []
    loss_rows: list[tuple[int, ...]] = []
    attention_rows: list[tuple[tuple[int, ...], ...]] = []
    position_rows: list[tuple[int, ...]] = []
    document_rows: list[tuple[str, ...]] = []
    span_rows: list[tuple[int, ...]] = []

    for row_index, sequence in enumerate(sequences):
        token_ids = sequence.token_ids
        input_rows.append(token_ids)

        if config.mode == "pretrain":
            loss_rows.append(pretrain_loss_mask(token_ids, pad_token_id=config.pad_token_id))
        else:
            eligibility = None if loss_eligible is None else loss_eligible[row_index]
            if eligibility is None:
                raise BatchError("agentic batches require loss_eligible per sequence")
            if len(eligibility) != len(token_ids):
                raise BatchError("loss_eligible must align with token_ids")
            loss_rows.append(
                agentic_loss_mask(
                    token_ids,
                    eligibility,
                    pad_token_id=config.pad_token_id,
                )
            )

        attention_rows.append(causal_attention_mask(token_ids, pad_token_id=config.pad_token_id))
        position_rows.append(
            build_position_ids(
                token_ids,
                sequence.document_ids,
                pad_token_id=config.pad_token_id,
                policy=config.position_id_policy,
            )
        )
        document_rows.append(sequence.document_ids)
        span_rows.append(sequence.span_ids)

    loss_rows_tuple = tuple(loss_rows)
    input_rows_tuple = tuple(input_rows)
    attention_rows_tuple = tuple(attention_rows)
    position_rows_tuple = tuple(position_rows)

    return Batch(
        input_ids=input_rows_tuple,
        loss_mask=loss_rows_tuple,
        attention_mask=attention_rows_tuple,
        position_ids=position_rows_tuple,
        document_ids=tuple(document_rows),
        span_ids=tuple(span_rows),
        mode=config.mode,
        packing_policy=packing_policy,
        position_id_policy=config.position_id_policy,
        loss_mask_hash=loss_mask_hash(loss_rows_tuple),
        batch_content_hash=batch_content_hash(
            input_ids=input_rows_tuple,
            loss_mask=loss_rows_tuple,
            attention_mask=attention_rows_tuple,
            position_ids=position_rows_tuple,
        ),
    )
