"""Batch tensors and masked next-token loss (P7-T02).

The batch builder already decided which positions are loss-bearing (`loss_mask[i] == 1`
means "position i predicts token i+1"). This module must respect that decision exactly:
padding, user turns, and tool outputs contribute zero gradient, and the denominator is
the number of loss-bearing tokens, not the number of slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from batch.types import Batch

from .errors import TrainerError

PAD_DOCUMENT_ID = ""


@dataclass(frozen=True)
class BatchTensors:
    """One microbatch converted to tensors for the model."""

    input_ids: Tensor
    loss_mask: Tensor
    attention_mask: Tensor
    position_ids: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass(frozen=True)
class MaskedLoss:
    """Masked loss for one microbatch, plus the per-token detail the ledger attributes."""

    loss: Tensor
    loss_bearing_tokens: int
    token_loss: Tensor
    """Detached `[batch, seq_len - 1]` loss per prediction; zero where `loss_mask == 0`."""

    @property
    def value(self) -> float:
        return float(self.loss.detach())


@dataclass(frozen=True)
class DocumentLoss:
    """Loss attributed to one document inside a microbatch (learning ledger input)."""

    document_id: str
    loss_bearing_tokens: int
    mean_loss: float


def batch_to_tensors(batch: Batch, *, device: torch.device | None = None) -> BatchTensors:
    """Convert a hashed Batch into model input tensors."""
    if batch.batch_size == 0:
        raise TrainerError("batch has no sequences")
    target = device or torch.device("cpu")
    return BatchTensors(
        input_ids=torch.tensor(batch.input_ids, dtype=torch.long, device=target),
        loss_mask=torch.tensor(batch.loss_mask, dtype=torch.long, device=target),
        attention_mask=torch.tensor(batch.attention_mask, dtype=torch.long, device=target),
        position_ids=torch.tensor(batch.position_ids, dtype=torch.long, device=target),
    )


def masked_causal_loss(logits: Tensor, tensors: BatchTensors) -> MaskedLoss:
    """Cross-entropy over `loss_mask == 1` positions only."""
    if logits.dim() != 3:
        raise TrainerError("logits must be [batch, seq_len, vocab_size]")
    if logits.shape[:2] != tensors.input_ids.shape:
        raise TrainerError("logits and input_ids must share batch and seq_len")
    if tensors.seq_len < 2:
        raise TrainerError("next-token loss needs at least 2 positions")

    # Position i predicts token i+1; the final column can never be loss-bearing.
    shifted_logits = logits[:, :-1, :]
    targets = tensors.input_ids[:, 1:]
    weights = tensors.loss_mask[:, :-1].to(logits.dtype)

    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    token_loss = token_loss * weights

    per_sequence_tokens = weights.sum(dim=1)
    total_tokens = per_sequence_tokens.sum()
    if float(total_tokens) == 0.0:
        raise TrainerError(
            "batch has zero loss-bearing tokens; it should not have been committed"
        )

    loss = token_loss.sum() / total_tokens

    return MaskedLoss(
        loss=loss,
        loss_bearing_tokens=int(total_tokens.item()),
        token_loss=token_loss.detach(),
    )


def per_document_losses(masked: MaskedLoss, batch: Batch) -> tuple[DocumentLoss, ...]:
    """Attribute the masked loss to the document that owns each predicted token.

    Position `i` predicts token `i + 1`, so the loss at `i` belongs to the document at
    `document_ids[i + 1]`, not `document_ids[i]`. The distinction is real: concat-and-chop
    packs several documents into one sequence, and the prediction that straddles a
    boundary is scored against the *next* document's first token. Attributing it to the
    previous document would credit shard A for tokens only shard B could explain.

    Documents come back in first-appearance order so the learning ledger rows for one
    microbatch are deterministic.
    """
    token_loss = masked.token_loss
    if token_loss.shape[0] != batch.batch_size or token_loss.shape[1] != batch.seq_len - 1:
        raise TrainerError(
            f"token_loss shape {tuple(token_loss.shape)} does not match batch "
            f"[{batch.batch_size}, {batch.seq_len - 1}]"
        )

    order: list[str] = []
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for row_index, document_row in enumerate(batch.document_ids):
        mask_row = batch.loss_mask[row_index]
        for index in range(len(document_row) - 1):
            if not mask_row[index]:
                continue
            document_id = document_row[index + 1]
            if document_id == PAD_DOCUMENT_ID:
                raise TrainerError(
                    f"loss-bearing position {row_index}:{index} has no document id; "
                    "the batch builder and packing policy disagree"
                )
            if document_id not in totals:
                order.append(document_id)
                totals[document_id] = 0.0
                counts[document_id] = 0
            totals[document_id] += float(token_loss[row_index, index])
            counts[document_id] += 1

    return tuple(
        DocumentLoss(
            document_id=document_id,
            loss_bearing_tokens=counts[document_id],
            mean_loss=totals[document_id] / counts[document_id],
        )
        for document_id in order
    )
