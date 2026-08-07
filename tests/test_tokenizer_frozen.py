"""Tests for the frozen tokenizer wrapper (P1-T01, P1-T03R)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
_BPE_ARTIFACT = _ASSIGNMENT / "data" / "tokenizer" / "bpe_tokenizer.json"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tokenizer import (  # noqa: E402
    BPE_ARTIFACT_NAME,
    FrozenTokenizer,
    TokenizerFrozenError,
    ensure_bpe_tokenizer_artifact,
)


class TestFrozenTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)

    def test_loads_default_bpe_artifact(self) -> None:
        self.assertEqual(self.tokenizer.artifact_path.name, BPE_ARTIFACT_NAME)
        self.assertTrue(self.tokenizer.artifact_path.is_file())
        self.assertEqual(self.tokenizer.vocab_size, 10000)

    def test_encode_is_deterministic(self) -> None:
        text = "India monsoon farmers climate policy"
        first = self.tokenizer.encode(text)
        second = self.tokenizer.encode(text)
        self.assertEqual(first, second)
        self.assertTrue(all(isinstance(token_id, int) for token_id in first))

    def test_decode_round_trip(self) -> None:
        text = "train loss on model data"
        token_ids = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(token_ids), text)

    def test_cannot_mutate_after_load(self) -> None:
        with self.assertRaises(TokenizerFrozenError):
            self.tokenizer.artifact_path = Path("other.json")  # type: ignore[misc]

    def test_second_load_from_same_artifact_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / BPE_ARTIFACT_NAME
            shutil.copy2(_BPE_ARTIFACT, path)
            loaded = FrozenTokenizer.from_file(path)
            sample = "user assistant tool call observation"
            self.assertEqual(loaded.encode(sample), self.tokenizer.encode(sample))

    def test_encodes_indic_and_english_without_unk(self) -> None:
        unk_id = self.tokenizer.get_vocab()[self.tokenizer.unk_token]
        samples = [
            "भारत भाषा शिक्षा",
            "Climate policy helps farmers during monsoon",
            "user assistant tool call",
        ]
        for text in samples:
            token_ids = self.tokenizer.encode(text)
            self.assertTrue(token_ids)
            self.assertNotIn(unk_id, token_ids)

    def test_subword_encoding_uses_multiple_tokens(self) -> None:
        token_ids = self.tokenizer.encode("India monsoon farmers")
        self.assertGreater(len(token_ids), 1)

    def test_ensure_bpe_tokenizer_artifact_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / BPE_ARTIFACT_NAME
            first = ensure_bpe_tokenizer_artifact(path, assignment_root=_ASSIGNMENT)
            second = ensure_bpe_tokenizer_artifact(path, assignment_root=_ASSIGNMENT)
            self.assertEqual(first, second)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
