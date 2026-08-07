"""Tests for the tiny training loop (P7-T01–T05)."""

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

import torch  # noqa: E402

from batch import BatchBuildConfig, assemble_microbatch, build_batch  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402
from config import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import (  # noqa: E402
    assert_no_eval_loss,
    build_eval_registry,
    candidate_from_planned_samples,
)
from ledger import load_consumption_ledger  # noqa: E402
from packing import ConcatAndChopPolicy, PackDocument, PackingConfig, pack_documents  # noqa: E402
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402
from trainer import (  # noqa: E402
    TinyModelConfig,
    TinyTrainer,
    TrainerError,
    TrainingContext,
    TrainingPaths,
    batch_to_tensors,
    build_model,
    build_training_runner,
    masked_causal_loss,
)
from trainer.loss import BatchTensors  # noqa: E402

SEQ_LEN = 32


def _tiny_config(vocab_size: int) -> TinyModelConfig:
    return TinyModelConfig(
        vocab_size=vocab_size,
        max_seq_len=SEQ_LEN,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )


class TestTinyModelConfig(unittest.TestCase):
    def test_rejects_head_dim_mismatch(self) -> None:
        with self.assertRaises(TrainerError):
            TinyModelConfig(
                vocab_size=100,
                max_seq_len=SEQ_LEN,
                n_layers=2,
                n_heads=5,
                d_model=32,
                d_ff=64,
                dropout=0.0,
            )

    def test_rejects_model_too_deep_for_cpu_demo(self) -> None:
        with self.assertRaises(TrainerError):
            TinyModelConfig(
                vocab_size=100,
                max_seq_len=SEQ_LEN,
                n_layers=8,
                n_heads=4,
                d_model=32,
                d_ff=64,
                dropout=0.0,
            )


class TestMaskedLoss(unittest.TestCase):
    """P7-T02: loss must be computed only where loss_mask == 1."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vocab_size = 64
        cls.config = _tiny_config(cls.vocab_size)

    def _padded_batch(self):
        # Real tokens then padding; pad positions must not contribute to the loss.
        token_ids = tuple(range(1, 9)) + tuple(0 for _ in range(SEQ_LEN - 8))
        docs = [PackDocument("doc-a", token_ids[:8])]
        packed = pack_documents(
            ConcatAndChopPolicy(),
            docs,
            PackingConfig(seq_len=SEQ_LEN, pad_token_id=0),
        )
        return build_batch(
            list(packed.sequences),
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )

    def test_loss_respects_mask(self) -> None:
        batch = self._padded_batch()
        tensors = batch_to_tensors(batch)
        model = build_model(self.config, seed=7)
        model.eval()

        with torch.no_grad():
            logits = model(
                tensors.input_ids,
                attention_mask=tensors.attention_mask,
                position_ids=tensors.position_ids,
            )
        masked = masked_causal_loss(logits, tensors)

        expected_tokens = sum(sum(row) for row in batch.loss_mask)
        self.assertEqual(masked.loss_bearing_tokens, expected_tokens)
        self.assertLess(masked.loss_bearing_tokens, batch.seq_len)
        self.assertTrue(torch.isfinite(masked.loss))

        # Rewriting logits at masked-out positions must not move the loss.
        torch.manual_seed(0)
        perturbed = logits.clone()
        mask = tensors.loss_mask.bool()
        perturbed[~mask] = torch.randn_like(perturbed[~mask]) * 5.0
        self.assertAlmostEqual(
            float(masked_causal_loss(perturbed, tensors).loss),
            float(masked.loss),
            places=5,
        )

        # Changing one loss-bearing position must move the loss. Shifting the whole
        # vocab row would not: softmax is invariant to a constant offset.
        moved = logits.clone()
        moved[0, 0, 0] += 25.0
        self.assertNotAlmostEqual(
            float(masked_causal_loss(moved, tensors).loss),
            float(masked.loss),
            places=3,
        )

    def test_rejects_batch_without_loss_bearing_tokens(self) -> None:
        batch = self._padded_batch()
        tensors = batch_to_tensors(batch)
        model = build_model(self.config, seed=7)
        with torch.no_grad():
            logits = model(
                tensors.input_ids,
                attention_mask=tensors.attention_mask,
                position_ids=tensors.position_ids,
            )
        empty = BatchTensors(
            input_ids=tensors.input_ids,
            loss_mask=torch.zeros_like(tensors.loss_mask),
            attention_mask=tensors.attention_mask,
            position_ids=tensors.position_ids,
        )
        with self.assertRaises(TrainerError):
            masked_causal_loss(logits, empty)


class TestTrainingRun(unittest.TestCase):
    """P7-T03–T05: real forward/backward, ledger commit, and interval checkpoints."""

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

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _context(self) -> TrainingContext:
        return TrainingContext(
            demo=self.demo,
            curriculum=self.curriculum,
            schedule=self.schedule,
            run_plan=self.run_plan,
            tokenizer=self.tokenizer,
            documents_by_id=self.documents_by_id,
            registry=self.registry,
        )

    def test_training_step_smoke(self) -> None:
        artifacts = Path(tempfile.mkdtemp(dir=self.temp_dir))
        paths = TrainingPaths.under(artifacts)
        runner = build_training_runner(self._context(), paths)

        summary = runner.run(stop_at_step=3)

        self.assertEqual(summary.start_step, 0)
        self.assertEqual(len(summary.steps), 3)
        self.assertGreater(summary.committed_microbatches, 0)
        for step in summary.steps:
            if step.microbatches == 0:
                # A step that trained on nothing reports no loss, not a zero loss.
                self.assertFalse(step.optimizer_stepped)
                self.assertIsNone(step.mean_loss)
                self.assertIsNone(step.perplexity)
                continue
            self.assertTrue(step.optimizer_stepped)
            self.assertTrue(0.0 < step.mean_loss < 100.0)
            self.assertGreater(step.loss_bearing_tokens, 0)
            self.assertGreater(step.perplexity, 1.0)

        # Every committed microbatch produced exactly one ledger row, in offset order.
        records = load_consumption_ledger(paths.ledger_path)
        self.assertEqual(len(records), summary.committed_microbatches)
        self.assertEqual(
            [record.ledger_offset for record in records],
            list(range(len(records))),
        )
        self.assertEqual(summary.final_ledger_offset, len(records) - 1)

        committed = [o for o in summary.outcomes if o.status == "committed"]
        self.assertEqual(
            [outcome.batch_content_hash for outcome in committed],
            [record.batch_content_hash for record in records],
        )

    def test_gated_microbatches_do_not_consume_ledger_offsets(self) -> None:
        artifacts = Path(tempfile.mkdtemp(dir=self.temp_dir))
        paths = TrainingPaths.under(artifacts)
        runner = build_training_runner(self._context(), paths)

        summary = runner.run(stop_at_step=6)
        skipped = [o for o in summary.outcomes if o.status != "committed"]
        if not skipped:
            self.skipTest("no firewall/OPUS rejections in this step range")

        for outcome in skipped:
            self.assertIsNone(outcome.ledger_offset)
            self.assertIsNone(outcome.training)

        records = load_consumption_ledger(paths.ledger_path)
        self.assertEqual(len(records), summary.committed_microbatches)
        self.assertLess(len(records), len(summary.outcomes))

        # Every OPUS decision is auditable, whether or not it was committed.
        audit_lines = paths.opus_audit_path.read_text(encoding="utf-8").splitlines()
        audited = {json.loads(line)["candidate_id"] for line in audit_lines if line.strip()}
        gated = {
            outcome.candidate_id
            for outcome in summary.outcomes
            if outcome.status != "firewall_blocked"
        }
        self.assertTrue(gated.issubset(audited))

    def test_checkpoint_saved_on_interval_with_ledger_binding(self) -> None:
        artifacts = Path(tempfile.mkdtemp(dir=self.temp_dir))
        paths = TrainingPaths.under(artifacts)
        runner = build_training_runner(self._context(), paths)

        interval = self.demo.training.checkpoint_interval
        summary = runner.run(stop_at_step=interval)

        self.assertEqual(summary.checkpoint_steps, (interval,))
        payload = load_checkpoint(paths.checkpoints_dir, global_step=interval)
        self.assertEqual(payload.ledger_offset, summary.final_ledger_offset)
        self.assertEqual(payload.next_global_step, interval)
        self.assertEqual(payload.branch_id, self.demo.run.branch_id)
        self.assertIsNotNone(payload.model_state)
        self.assertIsNotNone(payload.optimizer_state)

    def test_eval_document_carrying_loss_is_rejected_before_training(self) -> None:
        """Defense in depth: the loop reads the assembled mask, not just sample IDs."""
        eval_ids = [doc["document_id"] for doc in self.documents if doc.get("never_train")]
        self.assertTrue(eval_ids, "toy corpus must contain a never_train document")

        assembled = assemble_microbatch(
            (eval_ids[0],),
            ("shard-eval",),
            documents_by_id=self.documents_by_id,
            tokenizer=self.tokenizer,
            seq_len=SEQ_LEN,
        )
        leaked = candidate_from_planned_samples(
            candidate_id="cand-eval-leak",
            global_step=0,
            sample_ids=(eval_ids[0],),
            shard_ids=("shard-eval",),
            documents_by_id=self.documents_by_id,
            batch=assembled.batch,
        )
        with self.assertRaises(ValueError):
            assert_no_eval_loss(leaked, self.registry)

    def test_loss_decreases_over_repeated_batch(self) -> None:
        """Sanity check that backward + optimizer actually change the model."""
        samples = [doc["document_id"] for doc in self.documents if doc["data_type"] == "pretrain"]
        assembled = assemble_microbatch(
            tuple(samples[:2]),
            ("shard-a", "shard-a"),
            documents_by_id=self.documents_by_id,
            tokenizer=self.tokenizer,
            seq_len=SEQ_LEN,
        )
        model = build_model(_tiny_config(self.tokenizer.vocab_size), seed=self.demo.run.seed)
        trainer = TinyTrainer(model, self.demo.optimizer, gradient_accumulation_steps=1)

        losses = []
        for step in range(5):
            result = trainer.train_microbatch(
                assembled.batch,
                global_step=step,
                microbatch_index=0,
            )
            trainer.finish_step(step)
            losses.append(result.loss)

        self.assertLess(losses[-1], losses[0])


class TestAgenticAssembly(unittest.TestCase):
    """A microbatch containing agentic data must keep role-aware masking."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
        _, documents = load_corpus(_CORPUS)
        cls.documents_by_id = {doc["document_id"]: doc for doc in documents}
        cls.agentic_ids = [
            doc["document_id"] for doc in documents if doc["data_type"] == "agentic"
        ]
        cls.pretrain_ids = [
            doc["document_id"] for doc in documents if doc["data_type"] == "pretrain"
        ]

    def test_agentic_microbatch_uses_structure_preserving_masking(self) -> None:
        sample_ids = (self.agentic_ids[0], self.pretrain_ids[0])
        # Full trajectory length: a short seq_len would truncate the assistant turn.
        assembled = assemble_microbatch(
            sample_ids,
            ("shard-x", "shard-y"),
            documents_by_id=self.documents_by_id,
            tokenizer=self.tokenizer,
            seq_len=128,
        )

        self.assertEqual(assembled.packing_policy, "structure_preserving")
        self.assertEqual(assembled.mode, "agentic")
        # One sequence per document: no cross-document concatenation.
        self.assertEqual(assembled.batch.batch_size, len(sample_ids))
        # Only assistant tokens carry loss, so the agentic row is partially masked.
        agentic_row = assembled.batch.loss_mask[0]
        self.assertTrue(any(agentic_row))
        self.assertLess(sum(agentic_row), sum(1 for _ in agentic_row))

    def test_pretrain_only_microbatch_uses_concat_and_chop(self) -> None:
        assembled = assemble_microbatch(
            tuple(self.pretrain_ids[:2]),
            ("shard-y", "shard-z"),
            documents_by_id=self.documents_by_id,
            tokenizer=self.tokenizer,
            seq_len=SEQ_LEN,
        )
        self.assertEqual(assembled.packing_policy, "concat_and_chop")
        self.assertEqual(assembled.mode, "pretrain")
        self.assertGreater(assembled.utilization, 0.0)
        self.assertLessEqual(assembled.utilization, 1.0)


if __name__ == "__main__":
    unittest.main()
