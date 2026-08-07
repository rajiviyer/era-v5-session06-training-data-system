"""Tests for the learning ledger (P8-T01–T04)."""

from __future__ import annotations

import json
import math
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

from batch import assemble_microbatch  # noqa: E402
from config import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import build_eval_registry  # noqa: E402
from ledger import (  # noqa: E402
    aggregate_by_shard,
    load_consumption_ledger,
    load_learning_ledger,
    verify_learning_links,
)
from ledger.errors import LedgerError  # noqa: E402
from ledger.learning import (  # noqa: E402
    LearningLedgerEvent,
    model_phase_for_step,
    validate_learning_event,
)
from ledger.learning_aggregate import classify_usefulness  # noqa: E402
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402
from trainer import (  # noqa: E402
    TinyModelConfig,
    TinyTrainer,
    TrainingContext,
    TrainingPaths,
    build_model,
    build_training_runner,
)

SEQ_LEN = 32


def _event(**overrides) -> LearningLedgerEvent:
    """A valid learning row; overrides let one field at a time go wrong."""
    fields = {
        "run_id": "run-test",
        "branch_id": "main",
        "global_step": 4,
        "ledger_offset": 7,
        "microbatch_id": "mb-00004-0",
        "sample_id": "doc-web-001",
        "shard_id": "shard-web-0000",
        "capability_lane": "web",
        "curriculum_stage": "foundation",
        "model_phase": "early",
        "loss_bearing_tokens": 12,
        "mean_loss": 2.0,
        "perplexity": round(math.exp(2.0), 6),
        "opus_score": 0.71,
        "opus_decision_id": "opus-abc123",
        "batch_content_hash": "sha256:deadbeef",
    }
    fields.update(overrides)
    return LearningLedgerEvent(**fields)


class TestLearningEventSchema(unittest.TestCase):
    """P8-T01: the row must be self-consistent before it reaches disk."""

    def test_perplexity_must_derive_from_recorded_loss(self) -> None:
        validate_learning_event(_event())
        with self.assertRaises(LedgerError):
            validate_learning_event(_event(perplexity=3.0))

    def test_rejects_empty_and_negative_fields(self) -> None:
        for bad in ({"shard_id": ""}, {"loss_bearing_tokens": 0}, {"ledger_offset": -1}):
            with self.subTest(**bad), self.assertRaises(LedgerError):
                validate_learning_event(_event(**bad))

    def test_model_phase_tracks_progress_and_anneal_stage(self) -> None:
        self.assertEqual(model_phase_for_step(0, 30, "foundation"), "early")
        self.assertEqual(model_phase_for_step(15, 30, "skill_build"), "mid")
        self.assertEqual(model_phase_for_step(29, 30, "skill_build"), "late")
        # Anneal is a curriculum fact, so it wins over the step fraction.
        self.assertEqual(model_phase_for_step(2, 30, "anneal"), "anneal")


class TestUsefulnessClassification(unittest.TestCase):
    """P8-T03: usefulness is a loss trend, and a single exposure is not a trend."""

    def test_single_exposure_is_review_not_a_guess(self) -> None:
        self.assertEqual(classify_usefulness(1, -5.0), "review")

    def test_trend_labels(self) -> None:
        self.assertEqual(classify_usefulness(3, -0.4), "useful")
        self.assertEqual(classify_usefulness(3, 0.4), "harmful")
        self.assertEqual(classify_usefulness(3, 0.001), "neutral")


class TestLearningLedgerRun(unittest.TestCase):
    """P8-T02–T04 against a real training segment, not synthetic rows."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, cls.curriculum = load_configs(_ASSIGNMENT)
        cls.schedule = compile_schedule(
            cls.curriculum,
            total_steps=cls.demo.training.total_steps,
        )
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
        pool = build_sample_pool(cls.temp_dir / "manifests", cls.documents)
        cls.run_plan = plan_run(
            cls.schedule.steps,
            pool,
            run_id=cls.demo.run.run_id,
            branch_id=cls.demo.run.branch_id,
            seed=cls.demo.run.seed,
            global_batch_size=cls.demo.training.global_batch_size,
        )

        artifacts = Path(tempfile.mkdtemp(dir=cls.temp_dir))
        cls.paths = TrainingPaths.under(artifacts)
        runner = build_training_runner(
            TrainingContext(
                demo=cls.demo,
                curriculum=cls.curriculum,
                schedule=cls.schedule,
                run_plan=cls.run_plan,
                tokenizer=cls.tokenizer,
                documents_by_id=cls.documents_by_id,
                registry=cls.registry,
            ),
            cls.paths,
        )
        # Long enough that at least one shard is exposed more than once.
        cls.summary = runner.run(stop_at_step=8)
        cls.learning = load_learning_ledger(cls.paths.learning_path)
        cls.consumption = load_consumption_ledger(cls.paths.ledger_path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_learning_ledger_links_to_consumption_step(self) -> None:
        self.assertGreater(len(self.learning), 0)

        report = verify_learning_links(self.learning, self.consumption)
        self.assertEqual(report.orphan_offsets, ())
        self.assertEqual(report.unreported_offsets, ())
        self.assertEqual(report.mismatches, ())
        self.assertTrue(report.linked)
        # Every committed batch reported loss, and sample-level rows outnumber batches.
        self.assertEqual(report.linked_offsets, self.summary.committed_microbatches)
        self.assertGreaterEqual(report.learning_rows, report.committed_batches)

        # Each row's sample really was one of the samples that batch consumed.
        by_offset = {event.ledger_offset: event for event in self.consumption}
        for event in self.learning:
            committed = by_offset[event.ledger_offset]
            self.assertIn(event.sample_id, committed.packed_sample_ids)
            self.assertEqual(event.microbatch_id, committed.microbatch_id)

    def test_gated_microbatches_produce_no_learning_rows(self) -> None:
        """A batch the model never trained on must not appear as a learning outcome."""
        gated = [o for o in self.summary.outcomes if o.status != "committed"]
        if not gated:
            self.skipTest("no firewall/OPUS rejections in this step range")
        for outcome in gated:
            self.assertEqual(outcome.learning, ())

        recorded = {event.microbatch_id for event in self.learning}
        committed_ids = {event.microbatch_id for event in self.consumption}
        self.assertEqual(recorded, committed_ids)

    def test_shard_loss_aggregate_recomputable(self) -> None:
        aggregates = aggregate_by_shard(self.learning)
        self.assertGreater(len(aggregates), 0)

        for aggregate in aggregates:
            rows = [e for e in self.learning if e.shard_id == aggregate.shard_id]
            tokens = sum(row.loss_bearing_tokens for row in rows)
            expected = sum(row.mean_loss * row.loss_bearing_tokens for row in rows) / tokens

            self.assertEqual(aggregate.loss_bearing_tokens, tokens)
            self.assertAlmostEqual(aggregate.mean_loss, expected, places=9)
            # P8-T03: perplexity is derivable from the recorded loss.
            self.assertAlmostEqual(
                aggregate.perplexity, math.exp(aggregate.mean_loss), places=4
            )
            self.assertEqual(
                aggregate.exposure_count,
                len({row.global_step for row in rows}),
            )

    def test_at_least_one_shard_shows_a_loss_trend(self) -> None:
        aggregates = aggregate_by_shard(self.learning)
        multi = [a for a in aggregates if a.exposure_count >= 2]
        self.assertTrue(multi, "no shard was exposed at more than one step")

        trended = multi[0]
        steps = [exposure.global_step for exposure in trended.exposures]
        self.assertEqual(steps, sorted(steps))
        self.assertAlmostEqual(
            trended.loss_delta,
            trended.exposures[-1].mean_loss - trended.exposures[0].mean_loss,
            places=9,
        )
        self.assertIn(trended.usefulness, {"useful", "neutral", "harmful"})

    def test_ledger_file_is_append_only_jsonl(self) -> None:
        lines = self.paths.learning_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(self.learning))

        offsets = [json.loads(line)["ledger_offset"] for line in lines]
        self.assertEqual(offsets, sorted(offsets))
        # Rows survive a round trip byte-for-byte: nothing is recomputed on read.
        self.assertEqual(
            json.loads(lines[0]),
            json.loads(json.dumps(self.learning[0].to_dict(), sort_keys=True)),
        )


class TestPerDocumentAttribution(unittest.TestCase):
    """Sample-level granularity (D5): loss splits across documents in a packed sequence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo, _ = load_configs(_ASSIGNMENT)
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, documents = load_corpus(_CORPUS)
        cls.documents_by_id = {doc["document_id"]: doc for doc in documents}
        cls.pretrain_ids = [
            doc["document_id"] for doc in documents if doc["data_type"] == "pretrain"
        ]

    def test_document_losses_partition_the_masked_tokens(self) -> None:
        sample_ids = tuple(self.pretrain_ids[:2])
        assembled = assemble_microbatch(
            sample_ids,
            ("shard-a", "shard-b"),
            documents_by_id=self.documents_by_id,
            tokenizer=self.tokenizer,
            seq_len=SEQ_LEN,
        )
        model = build_model(
            TinyModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                max_seq_len=SEQ_LEN,
                n_layers=2,
                n_heads=4,
                d_model=32,
                d_ff=64,
                dropout=0.0,
            ),
            seed=self.demo.run.seed,
        )
        trainer = TinyTrainer(model, self.demo.optimizer, gradient_accumulation_steps=1)
        result = trainer.train_microbatch(
            assembled.batch, global_step=0, microbatch_index=0
        )

        per_document = result.per_document_loss
        self.assertGreater(len(per_document), 0)
        self.assertEqual(
            len({doc.document_id for doc in per_document}), len(per_document)
        )
        for doc in per_document:
            self.assertIn(doc.document_id, sample_ids)

        # Attribution is a partition: the parts cover every loss-bearing token, and the
        # token-weighted mean of the parts is the microbatch loss.
        self.assertEqual(
            sum(doc.loss_bearing_tokens for doc in per_document),
            result.loss_bearing_tokens,
        )
        weighted = sum(doc.mean_loss * doc.loss_bearing_tokens for doc in per_document)
        self.assertAlmostEqual(weighted / result.loss_bearing_tokens, result.loss, places=4)


if __name__ == "__main__":
    unittest.main()
