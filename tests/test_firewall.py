"""Tests for eval registry and training firewall (P4-T01–T05)."""

from __future__ import annotations

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

from batch import BatchBuildConfig, build_batch  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import (  # noqa: E402
    BatchCandidate,
    build_eval_registry,
    candidate_from_planned_samples,
    evaluate_firewall,
    load_eval_registry,
    log_firewall_rejection,
    write_eval_registry,
)
from packing import ConcatAndChopPolicy, PackDocument, PackingConfig, pack_documents  # noqa: E402
from runlog import RunLogWriter  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestEvalFirewall(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, cls.documents = load_corpus(_CORPUS)
        cls.documents_by_id = {doc["document_id"]: doc for doc in cls.documents}

        cls.temp_dir = Path(tempfile.mkdtemp())
        result = build_shards_with_manifests(
            cls.documents,
            tokenizer=cls.tokenizer,
            shards_dir=cls.temp_dir / "shards",
            manifests_dir=cls.temp_dir / "manifests",
        )
        cls.registry = build_eval_registry(
            cls.documents,
            manifests_dir=result.manifests_dir,
        )
        cls.eval_entry = next(entry for entry in cls.registry.entries if entry.document_id == "doc-eval-001")
        cls.eval_shard_id = cls.eval_entry.shard_id
        cls.clean_sample_id = "doc-web-004"
        cls.clean_shard_id = next(
            manifest["shard_id"]
            for manifest in result.manifests
            if cls.clean_sample_id in manifest["document_ids"]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_eval_registry_has_never_train_entry(self) -> None:
        self.assertTrue(self.eval_entry.never_train)
        self.assertEqual(self.eval_entry.benchmark_id, "mmlu_holdout_mirror")
        self.assertTrue(self.eval_entry.canary_strings)

    def test_eval_registry_round_trip(self) -> None:
        path = self.temp_dir / "eval_registry.json"
        write_eval_registry(path, self.registry)
        loaded = load_eval_registry(path)
        self.assertEqual(loaded.to_dict(), self.registry.to_dict())

    def test_firewall_blocks_never_train_shard(self) -> None:
        candidate = BatchCandidate(
            candidate_id="cand-eval-001",
            global_step=0,
            sample_ids=(self.eval_entry.document_id,),
            shard_ids=(self.eval_shard_id,),
            content_hashes=(self.eval_entry.content_hash,),
        )
        result = evaluate_firewall(candidate, self.registry, documents_by_id=self.documents_by_id)
        self.assertEqual(result.decision, "blocked")
        self.assertIn("never_train_shard_id", result.reasons)
        self.assertIn(self.eval_entry.entry_id, result.matched_entry_ids)

    def test_firewall_blocks_exact_content_hash(self) -> None:
        candidate = BatchCandidate(
            candidate_id="cand-eval-hash",
            global_step=1,
            sample_ids=("doc-web-004",),
            shard_ids=(self.clean_shard_id,),
            content_hashes=(self.eval_entry.content_hash,),
        )
        result = evaluate_firewall(candidate, self.registry)
        self.assertEqual(result.decision, "blocked")
        self.assertIn("exact_content_hash", result.reasons)

    def test_firewall_blocks_canary_string(self) -> None:
        candidate = candidate_from_planned_samples(
            candidate_id="cand-canary",
            global_step=2,
            sample_ids=[self.clean_sample_id],
            shard_ids=[self.clean_shard_id],
            documents_by_id=self.documents_by_id,
        )
        poisoned = self.documents_by_id[self.clean_sample_id].copy()
        poisoned["text"] = (
            poisoned["text"] + " MMLU holdout mirror leaked into training candidate."
        )
        docs = dict(self.documents_by_id)
        docs[self.clean_sample_id] = poisoned
        result = evaluate_firewall(candidate, self.registry, documents_by_id=docs)
        self.assertEqual(result.decision, "blocked")
        self.assertIn("canary_string_match", result.reasons)

    def test_firewall_allows_clean_candidate(self) -> None:
        candidate = candidate_from_planned_samples(
            candidate_id="cand-clean",
            global_step=3,
            sample_ids=[self.clean_sample_id],
            shard_ids=[self.clean_shard_id],
            documents_by_id=self.documents_by_id,
        )
        result = evaluate_firewall(candidate, self.registry, documents_by_id=self.documents_by_id)
        self.assertEqual(result.decision, "allowed")
        self.assertEqual(result.reasons, ())

    def test_no_eval_token_in_loss_mask(self) -> None:
        clean_doc = self.documents_by_id[self.clean_sample_id]
        token_ids = tuple(self.tokenizer.encode(clean_doc["text"])[:16])
        packed = pack_documents(
            ConcatAndChopPolicy(),
            [PackDocument(self.clean_sample_id, token_ids)],
            PackingConfig(seq_len=16, pad_token_id=0),
        )
        batch = build_batch(
            [packed.sequences[0]],
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        candidate = BatchCandidate(
            candidate_id="cand-clean-batch",
            global_step=4,
            sample_ids=(self.clean_sample_id,),
            shard_ids=(self.clean_shard_id,),
            content_hashes=(f"sha256:{clean_doc['content_sha256']}",),
            batch=batch,
        )
        result = evaluate_firewall(candidate, self.registry, documents_by_id=self.documents_by_id)
        self.assertEqual(result.decision, "allowed")
        self.assertTrue(any(sum(row) > 0 for row in batch.loss_mask))
        for row_index, document_row in enumerate(batch.document_ids):
            for token_index, document_id in enumerate(document_row):
                if document_id == "doc-eval-001":
                    self.assertEqual(batch.loss_mask[row_index][token_index], 0)

    def test_firewall_rejection_written_to_run_log(self) -> None:
        candidate = BatchCandidate(
            candidate_id="cand-logged",
            global_step=5,
            sample_ids=(self.eval_entry.document_id,),
            shard_ids=(self.eval_shard_id,),
            content_hashes=(self.eval_entry.content_hash,),
        )
        result = evaluate_firewall(candidate, self.registry)
        log_path = self.temp_dir / "run.log"
        event = log_firewall_rejection(
            RunLogWriter.open(log_path),
            result,
            run_id="s6-demo",
            branch_id="run-a",
            shard_ids=candidate.shard_ids,
            sample_ids=candidate.sample_ids,
        )
        self.assertEqual(event.event_type, "firewall_block")
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["event_type"], "firewall_block")
        self.assertEqual(payload["candidate_id"], "cand-logged")
        self.assertIn("never_train_shard_id", payload["reasons"])


if __name__ == "__main__":
    unittest.main()
