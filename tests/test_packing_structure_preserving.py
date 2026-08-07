"""Tests for structure-preserving packing policy (P2-T03)."""

from __future__ import annotations

import sys
import unittest

_ASSIGNMENT = __import__("pathlib").Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from packing import (  # noqa: E402
    ConcatAndChopPolicy,
    PackDocument,
    PackingConfig,
    StructurePreservingPolicy,
    pack_documents,
)
from packing.errors import PackingError  # noqa: E402


class TestStructurePreservingPolicy(unittest.TestCase):
    def test_emits_one_sequence_per_document(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2, 3)),
            PackDocument("doc-b", (4, 5)),
        ]
        result = pack_documents(
            StructurePreservingPolicy(),
            docs,
            PackingConfig(seq_len=6, pad_token_id=0),
        )
        self.assertEqual(result.policy, "structure_preserving")
        self.assertEqual(len(result.sequences), 2)
        self.assertEqual(result.sequences[0].token_ids, (1, 2, 3, 0, 0, 0))
        self.assertEqual(result.sequences[1].token_ids, (4, 5, 0, 0, 0, 0))

    def test_never_mixes_documents_in_one_sequence(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2)),
            PackDocument("doc-b", (3, 4)),
        ]
        structure = pack_documents(
            StructurePreservingPolicy(),
            docs,
            PackingConfig(seq_len=4, pad_token_id=0),
        )
        concat = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=4, pad_token_id=0),
        )
        self.assertEqual(len(structure.sequences), 2)
        self.assertEqual(len(concat.sequences), 1)
        useful_ids = {
            token_doc
            for sequence in structure.sequences
            for token_doc in sequence.document_ids
            if token_doc
        }
        self.assertEqual(useful_ids, {"doc-a", "doc-b"})

    def test_preserves_single_document_spans(self) -> None:
        docs = [PackDocument("doc-a", (10, 11, 12))]
        result = pack_documents(
            StructurePreservingPolicy(),
            docs,
            PackingConfig(seq_len=5, pad_token_id=0),
        )
        sequence = result.sequences[0]
        self.assertEqual(sequence.document_ids, ("doc-a", "doc-a", "doc-a", "", ""))
        self.assertEqual(sequence.span_ids, (0, 1, 2, -1, -1))

    def test_truncates_long_documents_instead_of_splitting(self) -> None:
        docs = [PackDocument("doc-a", (1, 2, 3, 4, 5))]
        result = pack_documents(
            StructurePreservingPolicy(),
            docs,
            PackingConfig(seq_len=3, pad_token_id=0),
        )
        self.assertEqual(len(result.sequences), 1)
        self.assertEqual(result.sequences[0].token_ids, (1, 2, 3))
        self.assertEqual(result.sequences[0].useful_tokens, 3)

    def test_can_reject_overlong_documents_when_truncation_disabled(self) -> None:
        docs = [PackDocument("doc-a", (1, 2, 3, 4))]
        policy = StructurePreservingPolicy(truncate_long_documents=False)
        with self.assertRaises(PackingError):
            pack_documents(policy, docs, PackingConfig(seq_len=3, pad_token_id=0))

    def test_lower_utilization_than_concat_for_same_docs(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2)),
            PackDocument("doc-b", (3, 4)),
        ]
        config = PackingConfig(seq_len=4, pad_token_id=0)
        structure = pack_documents(StructurePreservingPolicy(), docs, config)
        concat = pack_documents(ConcatAndChopPolicy(), docs, config)
        self.assertLess(structure.utilization, concat.utilization)


if __name__ == "__main__":
    unittest.main()
