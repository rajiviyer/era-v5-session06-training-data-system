"""Structure-preserving packing for agentic / SFT lanes (P2-T03)."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PackingError
from .types import PackDocument, PackResult, PackedSequence, PackingConfig

POLICY_NAME = "structure_preserving"


@dataclass(frozen=True)
class StructurePreservingPolicy:
    """Emit one fixed-length window per document without cross-doc concat or splits."""

    name: str = POLICY_NAME
    truncate_long_documents: bool = True

    def pack(self, docs: list[PackDocument], config: PackingConfig) -> PackResult:
        sequences: list[PackedSequence] = []
        for document in docs:
            if len(document.token_ids) > config.seq_len and not self.truncate_long_documents:
                raise PackingError(
                    f"{document.document_id}: token length ({len(document.token_ids)}) "
                    f"exceeds seq_len ({config.seq_len}); structure-preserving packing "
                    "cannot split documents"
                )

            tokens = list(document.token_ids[: config.seq_len])
            useful = len(tokens)
            if useful < config.seq_len:
                pad_count = config.seq_len - useful
                tokens.extend([config.pad_token_id] * pad_count)

            sequences.append(
                PackedSequence(
                    token_ids=tuple(tokens),
                    document_ids=tuple(
                        document.document_id if index < useful else ""
                        for index in range(config.seq_len)
                    ),
                    span_ids=tuple(
                        index if index < useful else -1 for index in range(config.seq_len)
                    ),
                    useful_tokens=useful,
                    policy=self.name,
                )
            )

        return PackResult(
            sequences=tuple(sequences),
            policy=self.name,
            seq_len=config.seq_len,
        )
