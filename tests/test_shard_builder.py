"""Tests for immutable shard builder (P1-T05)."""

from __future__ import annotations

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
from shards.builder import build_shards  # noqa: E402
from shards.format import content_hash, decode_binary_shard, decode_jsonl_shard  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestShardBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, cls.documents = load_corpus(_CORPUS)

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_builds_binary_and_jsonl_shards(self) -> None:
        shards = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir,
        )
        self.assertGreaterEqual(len(shards), 8)
        formats = {shard.format for shard in shards}
        self.assertIn("binary", formats)
        self.assertIn("jsonl", formats)
        for shard in shards:
            self.assertTrue(shard.path.is_file())
            self.assertEqual(shard.content_hash, content_hash(shard.path.read_bytes()))
            self.assertGreater(shard.token_count, 0)
            self.assertEqual(shard.tokenizer_hash, self.tokenizer.tokenizer_hash)

    def test_rebuild_produces_identical_content_hash(self) -> None:
        first = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir / "first",
        )
        second = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir / "second",
        )
        self.assertEqual(
            {shard.content_hash for shard in first},
            {shard.content_hash for shard in second},
        )

    def test_modifying_shard_bytes_changes_content_hash(self) -> None:
        shards = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir,
        )
        target = next(shard for shard in shards if shard.format == "binary")
        original_hash = target.content_hash
        payload = bytearray(target.path.read_bytes())
        payload[-1] ^= 0x01
        target.path.write_bytes(payload)
        self.assertNotEqual(content_hash(bytes(payload)), original_hash)

    def test_atomic_write_leaves_no_temp_files(self) -> None:
        build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir,
        )
        temp_files = list(self.temp_dir.rglob("*.tmp"))
        self.assertEqual(temp_files, [])

    def test_pretrain_binary_round_trip(self) -> None:
        shards = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir,
        )
        binary = next(shard for shard in shards if shard.format == "binary")
        token_ids = decode_binary_shard(binary.path.read_bytes())
        self.assertEqual(len(token_ids), binary.token_count)

    def test_agentic_jsonl_round_trip(self) -> None:
        shards = build_shards(
            self.documents,
            tokenizer=self.tokenizer,
            output_dir=self.temp_dir,
        )
        agentic = next(shard for shard in shards if shard.format == "jsonl")
        records = decode_jsonl_shard(agentic.path.read_bytes())
        self.assertEqual(len(records), len(agentic.document_ids))
        total_tokens = sum(len(record["token_ids"]) for record in records)
        self.assertEqual(total_tokens, agentic.token_count)


if __name__ == "__main__":
    unittest.main()
