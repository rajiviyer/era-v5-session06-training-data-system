"""Concat-and-chop packing for pretrain / web lanes (P2-T02)."""

from __future__ import annotations

from dataclasses import dataclass

from .types import PackDocument, PackResult, PackedSequence, PackingConfig

POLICY_NAME = "concat_and_chop"


@dataclass(frozen=True)
class ConcatAndChopPolicy:
    """Concatenate document tokens, then chop into fixed-length windows."""

    name: str = POLICY_NAME

    def pack(self, docs: list[PackDocument], config: PackingConfig) -> PackResult:
        tokens: list[int] = []
        document_ids: list[str] = []
        span_ids: list[int] = []

        for document in docs:
            for span_index, token_id in enumerate(document.token_ids):
                tokens.append(token_id)
                document_ids.append(document.document_id)
                span_ids.append(span_index)

        sequences: list[PackedSequence] = []
        for start in range(0, len(tokens), config.seq_len):
            window_tokens = tokens[start : start + config.seq_len]
            window_docs = document_ids[start : start + config.seq_len]
            window_spans = span_ids[start : start + config.seq_len]
            useful = len(window_tokens)

            if useful < config.seq_len:
                pad_count = config.seq_len - useful
                window_tokens = window_tokens + [config.pad_token_id] * pad_count
                window_docs = window_docs + [""] * pad_count
                window_spans = window_spans + [-1] * pad_count

            sequences.append(
                PackedSequence(
                    token_ids=tuple(window_tokens),
                    document_ids=tuple(window_docs),
                    span_ids=tuple(window_spans),
                    useful_tokens=useful,
                    policy=self.name,
                )
            )

        return PackResult(
            sequences=tuple(sequences),
            policy=self.name,
            seq_len=config.seq_len,
        )
