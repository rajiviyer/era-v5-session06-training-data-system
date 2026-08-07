"""Tests for concat-and-chop packing policy (P2-T02)."""

from __future__ import annotations

import sys
import unittest

_ASSIGNMENT = __import__("pathlib").Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from packing import ConcatAndChopPolicy, PackDocument, PackingConfig, pack_documents  # noqa: E402


class TestConcatAndChopPolicy(unittest.TestCase):
    def test_concatenates_then_chops_across_documents(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2, 3)),
            PackDocument("doc-b", (4, 5)),
        ]
        result = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=4, pad_token_id=0),
        )
        self.assertEqual(result.policy, "concat_and_chop")
        self.assertEqual(len(result.sequences), 2)
        self.assertEqual(result.sequences[0].token_ids, (1, 2, 3, 4))
        self.assertEqual(result.sequences[0].document_ids, ("doc-a", "doc-a", "doc-a", "doc-b"))
        self.assertEqual(result.sequences[0].span_ids, (0, 1, 2, 0))
        self.assertEqual(result.sequences[1].token_ids, (5, 0, 0, 0))
        self.assertEqual(result.sequences[1].document_ids, ("doc-b", "", "", ""))
        self.assertEqual(result.sequences[1].useful_tokens, 1)

    def test_single_sequence_when_stream_shorter_than_seq_len(self) -> None:
        docs = [PackDocument("doc-a", (7, 8))]
        result = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=5, pad_token_id=0),
        )
        self.assertEqual(len(result.sequences), 1)
        self.assertEqual(result.sequences[0].token_ids, (7, 8, 0, 0, 0))
        self.assertEqual(result.sequences[0].useful_tokens, 2)
        self.assertAlmostEqual(result.utilization, 2 / 5)

    def test_exact_multiple_emits_full_windows_without_pad(self) -> None:
        docs = [PackDocument("doc-a", (1, 2, 3, 4, 5, 6))]
        result = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=3, pad_token_id=0),
        )
        self.assertEqual(len(result.sequences), 2)
        self.assertEqual(result.sequences[0].token_ids, (1, 2, 3))
        self.assertEqual(result.sequences[1].token_ids, (4, 5, 6))
        self.assertEqual(result.utilization, 1.0)

    def test_chops_mid_document_at_mechanical_boundary(self) -> None:
        docs = [PackDocument("doc-a", (10, 11, 12, 13, 14))]
        result = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=4, pad_token_id=0),
        )
        self.assertEqual(result.sequences[0].token_ids, (10, 11, 12, 13))
        self.assertEqual(result.sequences[1].token_ids, (14, 0, 0, 0))
        self.assertEqual(result.sequences[0].document_ids, ("doc-a",) * 4)
        self.assertEqual(result.sequences[1].span_ids, (4, -1, -1, -1))

    def test_is_deterministic_for_same_input(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2)),
            PackDocument("doc-b", (3, 4, 5)),
        ]
        config = PackingConfig(seq_len=3, pad_token_id=0)
        first = pack_documents(ConcatAndChopPolicy(), docs, config)
        second = pack_documents(ConcatAndChopPolicy(), docs, config)
        self.assertEqual(first.sequences, second.sequences)


if __name__ == "__main__":
    unittest.main()
