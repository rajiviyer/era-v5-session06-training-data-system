"""Packing policy interface and validation (P2-T01)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .errors import PackingError
from .types import PackDocument, PackResult, PackingConfig

__all__ = [
    "PackingPolicy",
    "pack_documents",
    "validate_pack_documents",
]


@runtime_checkable
class PackingPolicy(Protocol):
    """Pack tokenized documents into fixed-length sequences."""

    @property
    def name(self) -> str:
        """Policy identifier (e.g. concat_and_chop)."""
        ...

    def pack(self, docs: list[PackDocument], config: PackingConfig) -> PackResult:
        """Pack docs into one or more sequences of length config.seq_len."""
        ...


def validate_pack_documents(docs: list[PackDocument]) -> list[PackDocument]:
    """Validate documents before packing."""
    if not docs:
        raise PackingError("packing requires at least one document")
    seen: set[str] = set()
    for document in docs:
        if not document.document_id.strip():
            raise PackingError("document_id must be non-empty")
        if document.document_id in seen:
            raise PackingError(f"duplicate document_id in pack batch: {document.document_id}")
        seen.add(document.document_id)
        if not document.token_ids:
            raise PackingError(f"{document.document_id}: token_ids must be non-empty")
        for token_id in document.token_ids:
            if not isinstance(token_id, int) or token_id < 0:
                raise PackingError(f"{document.document_id}: invalid token id {token_id!r}")
    return docs


def validate_packing_config(config: PackingConfig) -> PackingConfig:
    if config.seq_len <= 0:
        raise PackingError("seq_len must be positive")
    if config.pad_token_id < 0:
        raise PackingError("pad_token_id must be non-negative")
    return config


def pack_documents(policy: PackingPolicy, docs: list[PackDocument], config: PackingConfig) -> PackResult:
    """Validate inputs, invoke a policy, and validate outputs."""
    validate_pack_documents(docs)
    validate_packing_config(config)
    result = policy.pack(docs, config)
    if result.policy != policy.name:
        raise PackingError(
            f"policy {policy.name!r} returned result labeled {result.policy!r}"
        )
    if result.seq_len != config.seq_len:
        raise PackingError("result seq_len does not match config")
    if not result.sequences:
        raise PackingError("packing must emit at least one sequence")
    for sequence in result.sequences:
        if len(sequence.token_ids) != config.seq_len:
            raise PackingError(
                f"sequence length {len(sequence.token_ids)} != seq_len {config.seq_len}"
            )
        if sequence.policy != policy.name:
            raise PackingError("sequence policy label mismatch")
    return result
