"""Batch output types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionIdPolicy = Literal["absolute", "reset_at_document_boundary"]
BatchMode = Literal["pretrain", "agentic"]

POSITION_ID_POLICIES: frozenset[str] = frozenset({"absolute", "reset_at_document_boundary"})
BATCH_MODES: frozenset[str] = frozenset({"pretrain", "agentic"})


@dataclass(frozen=True)
class BatchBuildConfig:
    """Configuration for building one microbatch."""

    pad_token_id: int = 0
    mode: BatchMode = "pretrain"
    position_id_policy: PositionIdPolicy = "absolute"


@dataclass(frozen=True)
class Batch:
    """One training microbatch with masks, metadata, and content hashes."""

    input_ids: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[tuple[int, ...], ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    document_ids: tuple[tuple[str, ...], ...]
    span_ids: tuple[tuple[int, ...], ...]
    mode: str
    packing_policy: str
    position_id_policy: str
    loss_mask_hash: str
    batch_content_hash: str

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)

    @property
    def seq_len(self) -> int:
        if not self.input_ids:
            return 0
        return len(self.input_ids[0])
