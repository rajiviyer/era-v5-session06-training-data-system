"""Tests for tokenizer_manifest.json generation (P1-T03, P1-T03R)."""

from __future__ import annotations

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
    TokenizerManifestError,
    build_tokenizer_manifest,
    compute_tokenizer_hash_from_artifact,
    load_tokenizer_manifest,
    rebuild_bpe_tokenizer_artifact,
    write_tokenizer_manifest,
)


class TestTokenizerManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_path = _TOKENIZER_DIR / BPE_ARTIFACT_NAME
        cls.manifest_path = _TOKENIZER_DIR / "tokenizer_manifest.json"

    def test_build_manifest_has_required_fields(self) -> None:
        manifest = build_tokenizer_manifest(self.artifact_path)
        self.assertEqual(manifest["manifest_type"], "tokenizer")
        self.assertEqual(manifest["artifact"], BPE_ARTIFACT_NAME)
        self.assertEqual(manifest["model_type"], "BPE")
        self.assertEqual(manifest["pre_tokenizer"], "Metaspace")
        self.assertTrue(manifest["frozen"])
        self.assertEqual(manifest["vocab_size"], 10000)
        self.assertGreater(manifest["merge_count"], 0)

    def test_manifest_hash_matches_artifact(self) -> None:
        manifest = build_tokenizer_manifest(self.artifact_path)
        self.assertEqual(
            manifest["tokenizer_hash"],
            compute_tokenizer_hash_from_artifact(self.artifact_path),
        )

    def test_write_and_load_committed_manifest(self) -> None:
        loaded = load_tokenizer_manifest(self.manifest_path)
        self.assertEqual(loaded["artifact"], BPE_ARTIFACT_NAME)
        self.assertEqual(loaded["model_type"], "BPE")
        self.assertGreater(loaded["merge_count"], 0)

    def test_write_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / BPE_ARTIFACT_NAME
            manifest_path = Path(tmp) / "tokenizer_manifest.json"
            shutil.copy2(self.artifact_path, artifact)
            first = write_tokenizer_manifest(artifact, manifest_path)
            second = write_tokenizer_manifest(artifact, manifest_path)
            self.assertEqual(first, second)
            self.assertEqual(load_tokenizer_manifest(manifest_path), build_tokenizer_manifest(artifact))

    def test_rebuild_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / BPE_ARTIFACT_NAME
            rebuild_bpe_tokenizer_artifact(artifact, assignment_root=_ASSIGNMENT)
            manifest_path = artifact.parent / "tokenizer_manifest.json"
            self.assertTrue(manifest_path.is_file())
            load_tokenizer_manifest(manifest_path)

    def test_rejects_invalid_manifest(self) -> None:
        from tokenizer.manifest import validate_tokenizer_manifest

        with self.assertRaises(TokenizerManifestError):
            validate_tokenizer_manifest({"manifest_type": "tokenizer"})


if __name__ == "__main__":
    unittest.main()
