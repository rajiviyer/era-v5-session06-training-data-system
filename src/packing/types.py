"""Packing input and output types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PackingPolicyName = Literal["concat_and_chop", "structure_preserving", "pad_only"]

PACKING_POLICY_NAMES: frozenset[str] = frozenset(
    {"concat_and_chop", "structure_preserving", "pad_only"}
)


@dataclass(frozen=True)
class PackDocument:
    """One logical document ready for packing (already tokenized)."""

    document_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PackingConfig:
    """Parameters shared by all packing policies."""

    seq_len: int
    pad_token_id: int = 0


@dataclass(frozen=True)
class PackedSequence:
    """One fixed-length packed window produced by a policy."""

    token_ids: tuple[int, ...]
    document_ids: tuple[str, ...]
    span_ids: tuple[int, ...]
    useful_tokens: int
    policy: str

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        if length != len(self.document_ids) or length != len(self.span_ids):
            raise ValueError("token_ids, document_ids, and span_ids must have equal length")
        if self.useful_tokens < 0 or self.useful_tokens > length:
            raise ValueError("useful_tokens must be between 0 and sequence length")


@dataclass(frozen=True)
class PackResult:
    """All sequences emitted for one pack() call."""

    sequences: tuple[PackedSequence, ...]
    policy: str
    seq_len: int

    @property
    def utilization(self) -> float:
        """Fraction of slot capacity filled with non-pad tokens."""
        if not self.sequences:
            return 0.0
        capacity = self.seq_len * len(self.sequences)
        useful = sum(sequence.useful_tokens for sequence in self.sequences)
        return useful / capacity
