"""Turn planned samples into a training microbatch (P7-T04).

This is the single path from `PlannedSample` to `Batch`, so the training loop and the
P9 replay/fork paths rebuild byte-identical batches from the same planner output.

**Policy selection.** One microbatch produces exactly one `Batch`, which in turn
produces exactly one consumption ledger row. A `Batch` cannot mix packing policies, so
a microbatch that contains any agentic document is packed structure-preserving: losing
a little padding efficiency is acceptable, silently concatenating a tool-call
trajectory into an unrelated document is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from packing import (
    ConcatAndChopPolicy,
    PackDocument,
    PackingConfig,
    StructurePreservingPolicy,
    pack_documents,
)

from .agentic import encode_agentic_turns
from .builder import build_batch
from .errors import BatchError
from .types import Batch, BatchBuildConfig

if TYPE_CHECKING:
    from tokenizer.frozen import FrozenTokenizer

AGENTIC_DATA_TYPE = "agentic"


@dataclass(frozen=True)
class AssembledBatch:
    """One microbatch plus the provenance the ledger and metrics need."""

    batch: Batch
    sample_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    packing_policy: str
    mode: str
    useful_tokens: int

    @property
    def utilization(self) -> float:
        """Fraction of slot capacity filled with non-pad tokens (P10 input)."""
        capacity = self.batch.batch_size * self.batch.seq_len
        if capacity == 0:
            return 0.0
        return self.useful_tokens / capacity


def _tokenize_document(
    document: dict[str, Any],
    tokenizer: FrozenTokenizer,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Encode one corpus document into token IDs plus per-token loss eligibility."""
    text = document.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise BatchError(f"{document.get('document_id')}: document text is empty")

    if document.get("data_type") == AGENTIC_DATA_TYPE:
        return encode_agentic_turns(text, tokenizer)

    token_ids = tuple(tokenizer.encode(text))
    if not token_ids:
        raise BatchError(f"{document.get('document_id')}: encoded to zero tokens")
    # Pretrain documents are fully loss-bearing; padding is masked by the builder.
    return token_ids, tuple(True for _ in token_ids)


def _align_eligibility(
    eligible: tuple[bool, ...],
    seq_len: int,
) -> tuple[bool, ...]:
    """Truncate or pad eligibility to the packed sequence length (pad is never eligible)."""
    if len(eligible) >= seq_len:
        return eligible[:seq_len]
    return eligible + tuple(False for _ in range(seq_len - len(eligible)))


def assemble_microbatch(
    sample_ids: tuple[str, ...] | list[str],
    shard_ids: tuple[str, ...] | list[str],
    *,
    documents_by_id: dict[str, dict[str, Any]],
    tokenizer: FrozenTokenizer,
    seq_len: int,
    pad_token_id: int = 0,
    position_id_policy: str = "reset_at_document_boundary",
) -> AssembledBatch:
    """Tokenize, pack, and mask one planned microbatch."""
    if not sample_ids:
        raise BatchError("microbatch requires at least one planned sample")
    if len(sample_ids) != len(shard_ids):
        raise BatchError("sample_ids and shard_ids must have equal length")

    documents: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        document = documents_by_id.get(sample_id)
        if document is None:
            raise BatchError(f"unknown sample_id in microbatch: {sample_id}")
        documents.append(document)

    encoded = [_tokenize_document(document, tokenizer) for document in documents]
    has_agentic = any(
        document.get("data_type") == AGENTIC_DATA_TYPE for document in documents
    )

    pack_docs = [
        PackDocument(document["document_id"], token_ids)
        for document, (token_ids, _) in zip(documents, encoded)
    ]
    policy = StructurePreservingPolicy() if has_agentic else ConcatAndChopPolicy()
    packed = pack_documents(
        policy,
        pack_docs,
        PackingConfig(seq_len=seq_len, pad_token_id=pad_token_id),
    )

    loss_eligible: list[tuple[bool, ...] | None] | None = None
    if has_agentic:
        # Structure-preserving packing emits one sequence per document in input order,
        # so sequence i carries document i's eligibility.
        loss_eligible = [
            _align_eligibility(eligible, seq_len) for _, eligible in encoded
        ]
        if len(loss_eligible) != len(packed.sequences):
            raise BatchError(
                "structure-preserving packing must emit one sequence per document "
                f"(got {len(packed.sequences)} for {len(loss_eligible)} documents)"
            )

    batch = build_batch(
        list(packed.sequences),
        BatchBuildConfig(
            pad_token_id=pad_token_id,
            mode="agentic" if has_agentic else "pretrain",
            position_id_policy=position_id_policy,  # type: ignore[arg-type]
        ),
        packing_policy=policy.name,
        loss_eligible=loss_eligible,
    )

    if not any(any(row) for row in batch.loss_mask):
        # Most likely cause: seq_len truncated every agentic trajectory before its
        # first assistant turn, leaving nothing to learn from.
        raise BatchError(
            f"microbatch {tuple(sample_ids)} has no loss-bearing tokens at "
            f"seq_len={seq_len}"
        )

    return AssembledBatch(
        batch=batch,
        sample_ids=tuple(sample_ids),
        shard_ids=tuple(shard_ids),
        packing_policy=policy.name,
        mode=batch.mode,
        useful_tokens=sum(sequence.useful_tokens for sequence in packed.sequences),
    )
