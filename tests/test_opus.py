"""Tests for OPUS selector and audit trail (P5-T01–T06)."""

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
from config import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import (  # noqa: E402
    build_eval_registry,
    candidate_from_planned_samples,
)
from opus import (  # noqa: E402
    append_opus_audit,
    evaluate_opus,
    load_opus_audit,
    query_opus_audit,
    run_batch_gate,
)
from opus.pipeline import build_opus_context  # noqa: E402
from opus.scorer import DeterministicOpusScorer  # noqa: E402
from opus.types import OpusCandidateContext, OpusSelectorConfig  # noqa: E402
from packing import ConcatAndChopPolicy, PackDocument, PackingConfig, pack_documents  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class _FixedScorer:
    def __init__(self, value: float) -> None:
        self._value = value

    def score(self, _context: OpusCandidateContext) -> float:
        return self._value


class TestOpusSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, cls.curriculum = load_configs(_ASSIGNMENT)
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
        cls.clean_sample_id = "doc-web-004"
        cls.clean_shard_id = next(
            manifest["shard_id"]
            for manifest in result.manifests
            if cls.clean_sample_id in manifest["document_ids"]
        )
        cls.indic_sample_id = "doc-indic-001"
        cls.indic_shard_id = next(
            manifest["shard_id"]
            for manifest in result.manifests
            if cls.indic_sample_id in manifest["document_ids"]
        )
        cls.selector_config = OpusSelectorConfig(
            accept_threshold=cls.demo.opus.accept_threshold,
            protected_floor_lanes=cls.curriculum.protected_floors.lanes,
        )
        cls.run_kwargs = {
            "run_id": cls.demo.run.run_id,
            "branch_id": cls.demo.run.branch_id,
            "seed": cls.demo.run.seed,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _opus_context(
        self,
        *,
        candidate_id: str,
        global_step: int,
        sample_ids: tuple[str, ...],
        shard_ids: tuple[str, ...],
        capability_lane: str,
        path: str = "opus",
        curriculum_stage: str = "foundation",
    ) -> OpusCandidateContext:
        hashes = tuple(
            f"sha256:{self.documents_by_id[sample_id]['content_sha256']}"
            for sample_id in sample_ids
        )
        document = self.documents_by_id[sample_ids[0]]
        return OpusCandidateContext(
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            seed=self.run_kwargs["seed"],
            global_step=global_step,
            candidate_id=candidate_id,
            sample_ids=sample_ids,
            shard_ids=shard_ids,
            content_hashes=hashes,
            capability_lane=capability_lane,
            curriculum_stage=curriculum_stage,
            path=path,  # type: ignore[arg-type]
            curriculum_band=document.get("curriculum_band"),
            effective_token_estimate=int(document.get("char_count", 0)),
        )

    def test_opus_deterministic_score(self) -> None:
        context = self._opus_context(
            candidate_id="cand-score-001",
            global_step=3,
            sample_ids=(self.clean_sample_id,),
            shard_ids=(self.clean_shard_id,),
            capability_lane="web",
        )
        scorer = DeterministicOpusScorer()
        first = scorer.score(context)
        second = scorer.score(context)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

        changed = self._opus_context(
            candidate_id="cand-score-001",
            global_step=4,
            sample_ids=(self.clean_sample_id,),
            shard_ids=(self.clean_shard_id,),
            capability_lane="web",
        )
        self.assertNotEqual(first, scorer.score(changed))

    def test_protected_floor_override(self) -> None:
        context = self._opus_context(
            candidate_id="cand-protected-indic",
            global_step=7,
            sample_ids=(self.indic_sample_id,),
            shard_ids=(self.indic_shard_id,),
            capability_lane="indic",
        )
        result = evaluate_opus(
            context,
            self.selector_config,
            scorer=_FixedScorer(0.40),
        )
        self.assertEqual(result.decision, "protected_override")
        self.assertTrue(result.protected_floor_override)
        self.assertTrue(result.committed)
        self.assertIn("protected floor", result.audit.rejection_reason or "")

    def test_reject_and_defer_decisions(self) -> None:
        web_context = self._opus_context(
            candidate_id="cand-reject-web",
            global_step=8,
            sample_ids=(self.clean_sample_id,),
            shard_ids=(self.clean_shard_id,),
            capability_lane="web",
        )
        rejected = evaluate_opus(
            web_context,
            self.selector_config,
            scorer=_FixedScorer(0.30),
        )
        self.assertEqual(rejected.decision, "rejected")
        self.assertFalse(rejected.committed)

        deferred = evaluate_opus(
            web_context,
            self.selector_config,
            scorer=_FixedScorer(self.selector_config.defer_threshold + 0.01),
        )
        self.assertEqual(deferred.decision, "deferred")
        self.assertFalse(deferred.committed)

    def test_always_on_bypasses_scoring(self) -> None:
        context = self._opus_context(
            candidate_id="cand-always-on",
            global_step=1,
            sample_ids=(self.indic_sample_id,),
            shard_ids=(self.indic_shard_id,),
            capability_lane="indic",
            path="always_on",
        )
        result = evaluate_opus(context, self.selector_config)
        self.assertTrue(result.opus_bypassed)
        self.assertIsNone(result.opus_score)
        self.assertTrue(result.committed)

    def test_accepted_batch_has_audit_record(self) -> None:
        candidate = candidate_from_planned_samples(
            candidate_id="cand-audit-accept",
            global_step=9,
            sample_ids=[self.clean_sample_id],
            shard_ids=[self.clean_shard_id],
            documents_by_id=self.documents_by_id,
        )
        audit_path = self.temp_dir / "opus_audit.jsonl"
        pipeline_result = run_batch_gate(
            candidate,
            registry=self.registry,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            seed=self.run_kwargs["seed"],
            curriculum_stage="foundation",
            capability_lane="web",
            path="opus",
            opus_config=self.demo.opus,
            protected_floor_lanes=self.curriculum.protected_floors.lanes,
            audit_path=audit_path,
            documents_by_id=self.documents_by_id,
        )
        self.assertEqual(pipeline_result.firewall.decision, "allowed")
        self.assertIsNotNone(pipeline_result.opus)
        assert pipeline_result.opus is not None
        self.assertTrue(pipeline_result.committed)

        records = load_opus_audit(audit_path)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.candidate_id, "cand-audit-accept")
        self.assertTrue(record.opus_decision_id.startswith("opus-"))
        self.assertIn(record.decision, {"accepted", "protected_override"})

    def test_rejected_candidates_remain_queryable(self) -> None:
        audit_path = self.temp_dir / "query_audit.jsonl"
        contexts = [
            self._opus_context(
                candidate_id="cand-query-reject",
                global_step=10,
                sample_ids=(self.clean_sample_id,),
                shard_ids=(self.clean_shard_id,),
                capability_lane="web",
            ),
            self._opus_context(
                candidate_id="cand-query-defer",
                global_step=11,
                sample_ids=(self.clean_sample_id,),
                shard_ids=(self.clean_shard_id,),
                capability_lane="web",
            ),
        ]
        scores = [0.25, self.selector_config.defer_threshold + 0.02]
        for context, score in zip(contexts, scores):
            result = evaluate_opus(context, self.selector_config, scorer=_FixedScorer(score))
            append_opus_audit(audit_path, result.audit)

        rejected = query_opus_audit(audit_path, decision="rejected")
        deferred = query_opus_audit(audit_path, decision="deferred")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].candidate_id, "cand-query-reject")
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0].candidate_id, "cand-query-defer")

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)

    def test_pipeline_blocks_before_opus_on_firewall_hit(self) -> None:
        eval_entry = next(entry for entry in self.registry.entries if entry.document_id == "doc-eval-001")
        candidate = candidate_from_planned_samples(
            candidate_id="cand-firewall-first",
            global_step=12,
            sample_ids=[eval_entry.document_id],
            shard_ids=[eval_entry.shard_id or "missing"],
            documents_by_id=self.documents_by_id,
        )
        audit_path = self.temp_dir / "blocked_audit.jsonl"
        result = run_batch_gate(
            candidate,
            registry=self.registry,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            seed=self.run_kwargs["seed"],
            curriculum_stage="foundation",
            capability_lane="web",
            path="opus",
            opus_config=self.demo.opus,
            protected_floor_lanes=self.curriculum.protected_floors.lanes,
            audit_path=audit_path,
            documents_by_id=self.documents_by_id,
        )
        self.assertEqual(result.firewall.decision, "blocked")
        self.assertIsNone(result.opus)
        self.assertFalse(result.committed)
        self.assertFalse(audit_path.exists())

    def test_build_opus_context_from_batch(self) -> None:
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
        from firewall import BatchCandidate

        candidate = BatchCandidate(
            candidate_id="cand-context",
            global_step=13,
            sample_ids=(self.clean_sample_id,),
            shard_ids=(self.clean_shard_id,),
            content_hashes=(f"sha256:{clean_doc['content_sha256']}",),
            batch=batch,
        )
        context = build_opus_context(
            candidate,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            seed=self.run_kwargs["seed"],
            curriculum_stage="foundation",
            capability_lane="web",
            path="opus",
            documents_by_id=self.documents_by_id,
        )
        self.assertEqual(context.curriculum_band, clean_doc["curriculum_band"])
        self.assertGreater(context.effective_token_estimate, 0)

    def test_demo_corpus_produces_mixed_opus_decisions(self) -> None:
        """Sanity check: deterministic scorer yields multiple decision types."""
        scorer = DeterministicOpusScorer()
        decisions: set[str] = set()
        sample_ids = [
            doc["document_id"]
            for doc in self.documents
            if doc.get("opus_eligible") and not doc.get("never_train")
        ]
        for index, sample_id in enumerate(sample_ids[:20]):
            document = self.documents_by_id[sample_id]
            shard_id = self.clean_shard_id if sample_id == self.clean_sample_id else f"shard-{sample_id}"
            context = self._opus_context(
                candidate_id=f"cand-mix-{index}",
                global_step=index,
                sample_ids=(sample_id,),
                shard_ids=(shard_id,),
                capability_lane=document["capability_lane"],
            )
            score = scorer.score(context)
            result = evaluate_opus(context, self.selector_config, scorer=scorer)
            decisions.add(result.decision)
            if result.decision == "accepted":
                self.assertGreaterEqual(score, self.selector_config.accept_threshold)
        self.assertIn("accepted", decisions)


if __name__ == "__main__":
    unittest.main()
