"""Tests for batch builder and masks (P2-T04–T07)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from batch import BatchBuildConfig, build_batch, encode_agentic_turns  # noqa: E402
from batch.hash import batch_content_hash, loss_mask_hash  # noqa: E402
from packing import (  # noqa: E402
    ConcatAndChopPolicy,
    PackDocument,
    PackingConfig,
    StructurePreservingPolicy,
    pack_documents,
)
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestBatchBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)

    def _pretrain_sequences(self, *, seq_len: int = 4):
        docs = [
            PackDocument("doc-a", (1, 2, 3, 0)),
            PackDocument("doc-b", (4, 5, 0, 0)),
        ]
        result = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=seq_len, pad_token_id=0),
        )
        return list(result.sequences)

    def _single_pretrain_sequence(self):
        sequences = self._pretrain_sequences()
        return [sequences[0]]

    def test_builds_causal_attention_and_position_ids(self) -> None:
        batch = build_batch(
            self._single_pretrain_sequence(),
            BatchBuildConfig(pad_token_id=0, mode="pretrain", position_id_policy="absolute"),
            packing_policy="concat_and_chop",
        )
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.seq_len, 4)
        self.assertEqual(batch.position_id_policy, "absolute")
        self.assertEqual(batch.document_ids[0], ("doc-a", "doc-a", "doc-a", "doc-a"))
        self.assertEqual(batch.span_ids[0], (0, 1, 2, 3))
        attention = batch.attention_mask[0]
        self.assertEqual(attention[2], (1, 1, 1, 0))
        self.assertEqual(batch.position_ids[0], (0, 1, 2, 0))

    def test_pretrain_loss_mask_excludes_pad(self) -> None:
        batch = build_batch(
            self._single_pretrain_sequence(),
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        # (1,2,3,0): predict 2,3 ok; predicting pad at index 3 is excluded
        self.assertEqual(batch.loss_mask[0], (1, 1, 0, 0))

    def test_agentic_loss_mask_masks_user_turns(self) -> None:
        text = (
            '{"role":"user","content":"hello"}\n'
            '{"role":"assistant","content":"hi there"}\n'
            '{"role":"tool","content":"done"}'
        )
        token_ids, eligible = encode_agentic_turns(text, self.tokenizer)
        docs = [PackDocument("doc-agentic-001", token_ids)]
        packed = pack_documents(
            StructurePreservingPolicy(),
            docs,
            PackingConfig(seq_len=len(docs[0].token_ids), pad_token_id=0),
        )
        sequence = packed.sequences[0]
        self.assertEqual(len(eligible), len(sequence.token_ids))
        batch = build_batch(
            [sequence],
            BatchBuildConfig(pad_token_id=0, mode="agentic"),
            packing_policy="structure_preserving",
            loss_eligible=[eligible],
        )
        self.assertTrue(any(batch.loss_mask[0]))
        for index in range(len(batch.loss_mask[0]) - 1):
            if not eligible[index + 1]:
                self.assertEqual(batch.loss_mask[0][index], 0)

    def test_loss_mask_hash_stable(self) -> None:
        batch = build_batch(
            self._pretrain_sequences(),
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        again = build_batch(
            self._pretrain_sequences(),
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        self.assertEqual(batch.loss_mask_hash, again.loss_mask_hash)
        self.assertEqual(batch.batch_content_hash, again.batch_content_hash)
        self.assertEqual(batch.loss_mask_hash, loss_mask_hash(batch.loss_mask))

    def test_different_loss_mask_changes_loss_mask_hash(self) -> None:
        sequences = self._pretrain_sequences()
        first = build_batch(
            sequences,
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        modified = list(sequences[0].token_ids)
        modified[3] = 99
        sequences[0] = sequences[0].__class__(
            token_ids=tuple(modified),
            document_ids=sequences[0].document_ids,
            span_ids=sequences[0].span_ids,
            useful_tokens=sequences[0].useful_tokens,
            policy=sequences[0].policy,
        )
        second = build_batch(
            sequences,
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        self.assertNotEqual(first.loss_mask_hash, second.loss_mask_hash)

    def test_batch_content_hash_includes_tensors(self) -> None:
        batch = build_batch(
            self._pretrain_sequences(),
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        expected = batch_content_hash(
            input_ids=batch.input_ids,
            loss_mask=batch.loss_mask,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
        )
        self.assertEqual(batch.batch_content_hash, expected)

    def test_reset_at_document_boundary_zeros_pad_positions(self) -> None:
        batch = build_batch(
            self._single_pretrain_sequence(),
            BatchBuildConfig(
                pad_token_id=0,
                mode="pretrain",
                position_id_policy="reset_at_document_boundary",
            ),
            packing_policy="concat_and_chop",
        )
        self.assertEqual(batch.position_ids[0], (0, 1, 2, 0))

    def test_cross_document_boundary_resets_position_ids(self) -> None:
        docs = [
            PackDocument("doc-a", (1, 2)),
            PackDocument("doc-b", (3, 4)),
        ]
        packed = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=4, pad_token_id=0),
        )
        sequence = packed.sequences[0]
        self.assertEqual(sequence.token_ids, (1, 2, 3, 4))
        self.assertEqual(sequence.document_ids, ("doc-a", "doc-a", "doc-b", "doc-b"))

        batch = build_batch(
            [sequence],
            BatchBuildConfig(
                pad_token_id=0,
                mode="pretrain",
                position_id_policy="reset_at_document_boundary",
            ),
            packing_policy="concat_and_chop",
        )
        self.assertEqual(batch.position_ids[0], (0, 1, 0, 1))


if __name__ == "__main__":
    unittest.main()
