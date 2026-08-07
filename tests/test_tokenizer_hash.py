"""Tests for tokenizer hash computation and persistence (P1-T02, P1-T03R)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
_TOKENIZER_DIR = _ASSIGNMENT / "data" / "tokenizer"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tokenizer import (  # noqa: E402
    BPE_ARTIFACT_NAME,
    FrozenTokenizer,
    compute_tokenizer_hash_from_artifact,
    load_persisted_tokenizer_hash,
    persist_tokenizer_hash,
    rebuild_bpe_tokenizer_artifact,
)
from tokenizer.hash import _fingerprint  # noqa: E402


class TestTokenizerHash(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_path = _TOKENIZER_DIR / BPE_ARTIFACT_NAME
        cls.hash_path = _TOKENIZER_DIR / "tokenizer_hash.json"

    def test_tokenizer_hash_stable(self) -> None:
        first = compute_tokenizer_hash_from_artifact(self.artifact_path)
        second = compute_tokenizer_hash_from_artifact(self.artifact_path)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("tok_"))

    def test_fingerprint_includes_merges(self) -> None:
        data = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        fingerprint = _fingerprint(data)
        self.assertGreater(len(fingerprint["merges"]), 0)

    def test_persisted_hash_matches_computed(self) -> None:
        computed = compute_tokenizer_hash_from_artifact(self.artifact_path)
        persisted = load_persisted_tokenizer_hash(self.hash_path)
        self.assertEqual(computed, persisted)

    def test_frozen_tokenizer_exposes_hash(self) -> None:
        tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        self.assertEqual(tokenizer.tokenizer_hash, load_persisted_tokenizer_hash(self.hash_path))

    def test_vocab_change_changes_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / BPE_ARTIFACT_NAME
            shutil.copy2(self.artifact_path, path)
            original_hash = compute_tokenizer_hash_from_artifact(path)

            data = json.loads(path.read_text(encoding="utf-8"))
            data["model"]["vocab"]["extra_word"] = max(data["model"]["vocab"].values()) + 1
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            self.assertNotEqual(original_hash, compute_tokenizer_hash_from_artifact(path))

    def test_persist_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / BPE_ARTIFACT_NAME
            hash_path = Path(tmp) / "tokenizer_hash.json"
            shutil.copy2(self.artifact_path, artifact)
            first = persist_tokenizer_hash(artifact, hash_path)
            second = persist_tokenizer_hash(artifact, hash_path)
            self.assertEqual(first, second)

    def test_rebuild_updates_persisted_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / BPE_ARTIFACT_NAME
            rebuild_bpe_tokenizer_artifact(artifact, assignment_root=_ASSIGNMENT)
            hash_path = artifact.parent / "tokenizer_hash.json"
            self.assertEqual(
                load_persisted_tokenizer_hash(hash_path),
                compute_tokenizer_hash_from_artifact(artifact),
            )


if __name__ == "__main__":
    unittest.main()
