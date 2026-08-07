"""Tests for shard manifests, admission gate, and build pipeline."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
_CORPUS = _ASSIGNMENT / "data" / "toy_corpus"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from corpus import load_corpus  # noqa: E402
from shards.admission import apply_admission, evaluate_admission  # noqa: E402
from shards.builder import build_shards  # noqa: E402
from shards.manifest import (  # noqa: E402
    REQUIRED_MANIFEST_KEYS,
    build_shard_manifest,
    load_shard_manifest,
    write_shard_manifest,
)
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from shards.registry import REGISTRY_FILENAME, load_registry_index  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestShardManifestAndAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, cls.documents = load_corpus(_CORPUS)

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_one_manifest(self) -> dict:
        shards = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir / "shards",
        )
        built = shards[0]
        docs_by_id = {doc["document_id"]: doc for doc in self.documents}
        shard_docs = [docs_by_id[doc_id] for doc_id in built.document_ids]
        return build_shard_manifest(built, shard_docs)

    def test_manifest_has_required_fields(self) -> None:
        manifest = apply_admission(self._build_one_manifest())
        self.assertEqual(set(manifest.keys()), REQUIRED_MANIFEST_KEYS)
        self.assertTrue(manifest["content_hash"].startswith("sha256:"))
        self.assertTrue(manifest["tokenizer_hash"].startswith("tok_"))

    def test_admission_gate_blocks_missing_tokenizer_hash(self) -> None:
        manifest = self._build_one_manifest()
        manifest["tokenizer_hash"] = ""
        admission, reasons = evaluate_admission(manifest)
        self.assertEqual(admission, "blocked")
        self.assertIn("missing_tokenizer_hash", reasons)

    def test_admission_gate_blocks_eval_overlap(self) -> None:
        manifest = self._build_one_manifest()
        manifest["eval_overlap_status"] = "overlap_detected"
        admission, reasons = evaluate_admission(manifest)
        self.assertEqual(admission, "blocked")
        self.assertIn("eval_overlap_detected", reasons)

    def test_eval_shard_blocked_in_full_pipeline(self) -> None:
        result = build_shards_with_manifests(
            self.documents,
            tokenizer=self.tokenizer,
            shards_dir=self.temp_dir / "shards",
            manifests_dir=self.temp_dir / "manifests",
        )
        eval_manifests = [
            manifest
            for manifest in result.manifests
            if "doc-eval-001" in manifest["document_ids"]
        ]
        self.assertEqual(len(eval_manifests), 1)
        self.assertEqual(eval_manifests[0]["admission"], "blocked")
        self.assertIn(eval_manifests[0]["shard_id"], result.registry["blocked_shard_ids"])
        self.assertNotIn(eval_manifests[0]["shard_id"], result.registry["admitted_shard_ids"])

    def test_admitted_shards_listed_in_registry(self) -> None:
        result = build_shards_with_manifests(
            self.documents,
            tokenizer=self.tokenizer,
            shards_dir=self.temp_dir / "shards",
            manifests_dir=self.temp_dir / "manifests",
        )
        registry_path = self.temp_dir / "manifests" / REGISTRY_FILENAME
        self.assertTrue(registry_path.is_file())
        loaded = load_registry_index(registry_path)
        self.assertEqual(loaded["admitted_shard_ids"], result.registry["admitted_shard_ids"])
        self.assertGreater(len(loaded["admitted_shard_ids"]), 0)

    def test_manifest_write_and_load_round_trip(self) -> None:
        manifest = apply_admission(self._build_one_manifest())
        manifests_dir = self.temp_dir / "manifests"
        path = write_shard_manifest(manifests_dir, manifest)
        loaded = load_shard_manifest(path)
        self.assertEqual(loaded, manifest)

    def test_tokenizer_change_invalidates_shard_binding(self) -> None:
        result = build_shards_with_manifests(
            self.documents,
            tokenizer=self.tokenizer,
            shards_dir=self.temp_dir / "first" / "shards",
            manifests_dir=self.temp_dir / "first" / "manifests",
        )
        artifact = self.tokenizer.artifact_path
        data = json.loads(artifact.read_text(encoding="utf-8"))
        modified = copy.deepcopy(data)
        modified["model"]["merges"] = modified["model"]["merges"][:-100]
        alt_path = self.temp_dir / "alt_tokenizer.json"
        alt_path.write_text(json.dumps(modified), encoding="utf-8")
        alt_tokenizer = FrozenTokenizer.from_file(alt_path)
        self.assertNotEqual(alt_tokenizer.tokenizer_hash, self.tokenizer.tokenizer_hash)

        alt_result = build_shards_with_manifests(
            self.documents,
            tokenizer=alt_tokenizer,
            shards_dir=self.temp_dir / "second" / "shards",
            manifests_dir=self.temp_dir / "second" / "manifests",
        )
        self.assertNotEqual(
            result.registry["tokenizer_hash"],
            alt_result.registry["tokenizer_hash"],
        )
        self.assertNotEqual(
            result.manifests[0]["tokenizer_hash"],
            alt_result.manifests[0]["tokenizer_hash"],
        )
        self.assertEqual(result.manifests[0]["content_hash"], alt_result.manifests[0]["content_hash"])


if __name__ == "__main__":
    unittest.main()
