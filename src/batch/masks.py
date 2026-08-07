"""Loss, attention, and position mask builders."""

from __future__ import annotations

from .types import PositionIdPolicy


def pretrain_loss_mask(token_ids: tuple[int, ...], *, pad_token_id: int) -> tuple[int, ...]:
    """Next-token loss on non-pad targets (P2-T05)."""
    length = len(token_ids)
    mask = [0] * length
    for index in range(length - 1):
        if token_ids[index] == pad_token_id:
            continue
        if token_ids[index + 1] != pad_token_id:
            mask[index] = 1
    return tuple(mask)


def agentic_loss_mask(
    token_ids: tuple[int, ...],
    loss_eligible: tuple[bool, ...],
    *,
    pad_token_id: int,
) -> tuple[int, ...]:
    """Next-token loss only where the target token is assistant/tool-call (P2-T06)."""
    if len(loss_eligible) != len(token_ids):
        raise ValueError("loss_eligible must align with token_ids")
    length = len(token_ids)
    mask = [0] * length
    for index in range(length - 1):
        if token_ids[index] == pad_token_id:
            continue
        if token_ids[index + 1] == pad_token_id:
            continue
        if loss_eligible[index + 1]:
            mask[index] = 1
    return tuple(mask)


def causal_attention_mask(token_ids: tuple[int, ...], *, pad_token_id: int) -> tuple[tuple[int, ...], ...]:
    """Lower-triangular causal mask; pad rows and columns are zero (P2-T04)."""
    length = len(token_ids)
    rows: list[tuple[int, ...]] = []
    for row in range(length):
        if token_ids[row] == pad_token_id:
            rows.append(tuple(0 for _ in range(length)))
            continue
        row_mask = []
        for col in range(length):
            if col > row:
                row_mask.append(0)
            elif token_ids[col] == pad_token_id:
                row_mask.append(0)
            else:
                row_mask.append(1)
        rows.append(tuple(row_mask))
    return tuple(rows)


def build_position_ids(
    token_ids: tuple[int, ...],
    document_ids: tuple[str, ...],
    *,
    pad_token_id: int,
    policy: PositionIdPolicy,
) -> tuple[int, ...]:
    """Absolute or document-boundary-reset position IDs."""
    length = len(token_ids)
    if policy == "absolute":
        return tuple(index if token_ids[index] != pad_token_id else 0 for index in range(length))

    position = 0
    positions: list[int] = []
    previous_doc = ""
    for index in range(length):
        doc_id = document_ids[index]
        if token_ids[index] == pad_token_id or not doc_id:
            positions.append(0)
            continue
        if previous_doc and doc_id != previous_doc:
            position = 0
        positions.append(position)
        position += 1
        previous_doc = doc_id
    return tuple(positions)
