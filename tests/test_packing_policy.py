"""Tests for packing policy interface (P2-T01)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass

_ASSIGNMENT = __import__("pathlib").Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from packing import (  # noqa: E402
    PackDocument,
    PackingConfig,
    PackingPolicy,
    pack_documents,
)
from packing.errors import PackingError  # noqa: E402
from packing.policy import validate_pack_documents  # noqa: E402


@dataclass
class PadOnlyPolicy:
    """Minimal reference policy: one document per sequence, pad to seq_len."""

    name: str = "pad_only"

    def pack(self, docs: list[PackDocument], config: PackingConfig):
        from packing.types import PackResult, PackedSequence

        sequences: list[PackedSequence] = []
        for document in docs:
            tokens = list(document.token_ids[: config.seq_len])
            useful = len(tokens)
            if useful < config.seq_len:
                tokens.extend([config.pad_token_id] * (config.seq_len - useful))
            sequences.append(
                PackedSequence(
                    token_ids=tuple(tokens),
                    document_ids=tuple(
                        document.document_id if index < useful else ""
                        for index in range(config.seq_len)
                    ),
                    span_ids=tuple(index if index < useful else -1 for index in range(config.seq_len)),
                    useful_tokens=useful,
                    policy=self.name,
                )
            )
        return PackResult(sequences=tuple(sequences), policy=self.name, seq_len=config.seq_len)


class TestPackingPolicyInterface(unittest.TestCase):
    def test_policy_protocol_is_runtime_checkable(self) -> None:
        policy = PadOnlyPolicy()
        self.assertIsInstance(policy, PackingPolicy)

    def test_pack_documents_returns_fixed_length_sequences(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2, 3, 4, 5)),
            PackDocument("doc-b", (10, 11)),
        ]
        result = pack_documents(PadOnlyPolicy(), docs, PackingConfig(seq_len=8, pad_token_id=0))
        self.assertEqual(result.policy, "pad_only")
        self.assertEqual(result.seq_len, 8)
        self.assertEqual(len(result.sequences), 2)
        self.assertEqual(len(result.sequences[0].token_ids), 8)
        self.assertEqual(result.sequences[0].useful_tokens, 5)
        self.assertEqual(result.sequences[1].useful_tokens, 2)
        self.assertAlmostEqual(result.utilization, (5 + 2) / (8 * 2))

    def test_preserves_document_provenance_per_token(self) -> None:
        docs = [PackDocument("doc-a", (1, 2, 3))]
        result = pack_documents(PadOnlyPolicy(), docs, PackingConfig(seq_len=4, pad_token_id=0))
        sequence = result.sequences[0]
        self.assertEqual(sequence.document_ids, ("doc-a", "doc-a", "doc-a", ""))
        self.assertEqual(sequence.span_ids, (0, 1, 2, -1))

    def test_rejects_empty_documents(self) -> None:
        with self.assertRaises(PackingError):
            validate_pack_documents([])
        with self.assertRaises(PackingError):
            pack_documents(PadOnlyPolicy(), [], PackingConfig(seq_len=4))

    def test_rejects_invalid_seq_len(self) -> None:
        docs = [PackDocument("doc-a", (1, 2))]
        with self.assertRaises(PackingError):
            pack_documents(PadOnlyPolicy(), docs, PackingConfig(seq_len=0))


if __name__ == "__main__":
    unittest.main()
