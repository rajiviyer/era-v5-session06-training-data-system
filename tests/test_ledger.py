"""Tests for consumption ledger and checkpoints (P6-T01–T07)."""

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

from batch import BatchBuildConfig, build_batch  # noqa: E402
from checkpoint import (  # noqa: E402
    CheckpointError,
    build_checkpoint_payload,
    dataloader_state_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from config import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import build_eval_registry, candidate_from_planned_samples  # noqa: E402
from ledger import (  # noqa: E402
    LedgerBoundDataLoader,
    LedgerWriter,
    commit_batch,
    load_consumption_ledger,
    reconstruct_at_global_step,
)
from opus import run_batch_gate  # noqa: E402
from packing import ConcatAndChopPolicy, PackDocument, PackingConfig, pack_documents  # noqa: E402
from schedule import build_sample_pool, compile_schedule, plan_run  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


class TestConsumptionLedger(unittest.TestCase):
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
        cls.pool = build_sample_pool(cls.temp_dir / "manifests", cls.documents)
        cls.run_plan = plan_run(
            cls.schedule.steps,
            cls.pool,
            run_id=cls.demo.run.run_id,
            branch_id=cls.demo.run.branch_id,
            seed=cls.demo.run.seed,
            global_batch_size=cls.demo.training.global_batch_size,
        )
        cls.run_kwargs = {
            "run_id": cls.demo.run.run_id,
            "branch_id": cls.demo.run.branch_id,
            "seed": cls.demo.run.seed,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _build_batch_for_samples(self, sample_ids: list[str]) -> tuple[object, str]:
        sequences = []
        for sample_id in sample_ids:
            document = self.documents_by_id[sample_id]
            token_ids = tuple(self.tokenizer.encode(document["text"])[:16])
            sequences.append(PackDocument(sample_id, token_ids))
        packed = pack_documents(
            ConcatAndChopPolicy(),
            sequences,
            PackingConfig(seq_len=16, pad_token_id=0),
        )
        batch = build_batch(
            packed.sequences,
            BatchBuildConfig(pad_token_id=0, mode="pretrain"),
            packing_policy="concat_and_chop",
        )
        return batch, sample_ids[0]

    def _commit_microbatch(
        self,
        dataloader: LedgerBoundDataLoader,
        writer: LedgerWriter,
        *,
        audit_path: Path,
        max_attempts: int = 40,
    ) -> object:
        """Commit the next microbatch the gates accept.

        The firewall and OPUS reject candidates on their own schedule, so a test cannot
        assume the microbatch at any given cursor position commits. Skipping past gated
        candidates is what the training loop does too.
        """
        for _ in range(max_attempts):
            event = self._try_commit(dataloader, writer, audit_path=audit_path)
            if event is not None:
                return event
        raise AssertionError(f"no microbatch committed within {max_attempts} attempts")

    def _try_commit(
        self,
        dataloader: LedgerBoundDataLoader,
        writer: LedgerWriter,
        *,
        audit_path: Path,
    ) -> object | None:
        plan = dataloader.next_microbatch()
        sample_ids = [sample.sample_id for sample in plan.samples]
        shard_ids = [sample.shard_id for sample in plan.samples]
        candidate = candidate_from_planned_samples(
            candidate_id=plan.candidate_id,
            global_step=plan.global_step,
            sample_ids=sample_ids,
            shard_ids=shard_ids,
            documents_by_id=self.documents_by_id,
        )
        batch, _ = self._build_batch_for_samples(sample_ids)
        candidate = candidate.__class__(
            candidate_id=candidate.candidate_id,
            global_step=candidate.global_step,
            sample_ids=candidate.sample_ids,
            shard_ids=candidate.shard_ids,
            content_hashes=candidate.content_hashes,
            batch=batch,
        )
        primary_lane = plan.samples[0].capability_lane
        primary_path = plan.samples[0].path
        pipeline_result = run_batch_gate(
            candidate,
            registry=self.registry,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            seed=self.run_kwargs["seed"],
            curriculum_stage=plan.curriculum_stage,
            capability_lane=primary_lane,
            path=primary_path,  # type: ignore[arg-type]
            opus_config=self.demo.opus,
            protected_floor_lanes=self.curriculum.protected_floors.lanes,
            audit_path=audit_path,
            documents_by_id=self.documents_by_id,
        )
        if not pipeline_result.committed:
            dataloader.advance_after_skip()
            return None
        event = commit_batch(
            writer,
            pipeline_result,
            batch,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            global_step=plan.global_step,
            curriculum_stage=plan.curriculum_stage,
            mixture_lane=primary_lane,
            tokenizer_hash=self.tokenizer.tokenizer_hash,
            microbatch_index=plan.microbatch_index,
        )
        dataloader.advance_after_commit()
        return event

    def test_ledger_append_only_monotonic_offsets(self) -> None:
        ledger_path = self.temp_dir / "monotonic.jsonl"
        writer = LedgerWriter.open(ledger_path)
        dataloader = LedgerBoundDataLoader(
            self.run_plan,
            microbatch_size=self.demo.training.microbatch_size,
            global_batch_size=self.demo.training.global_batch_size,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
        )
        audit_path = self.temp_dir / "audit_monotonic.jsonl"
        offsets: list[int] = []
        for _ in range(3):
            event = self._commit_microbatch(dataloader, writer, audit_path=audit_path)
            offsets.append(event.ledger_offset)

        self.assertEqual(offsets, [0, 1, 2])
        records = load_consumption_ledger(ledger_path)
        self.assertEqual([record.ledger_offset for record in records], [0, 1, 2])

    def test_reconstruct_batch_from_ledger_step(self) -> None:
        ledger_path = self.temp_dir / "reconstruct.jsonl"
        writer = LedgerWriter.open(ledger_path)
        dataloader = LedgerBoundDataLoader(
            self.run_plan,
            microbatch_size=self.demo.training.microbatch_size,
            global_batch_size=self.demo.training.global_batch_size,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
        )
        audit_path = self.temp_dir / "audit_reconstruct.jsonl"
        committed_by_step: dict[int, list[str]] = {}
        for _ in range(4):
            event = self._commit_microbatch(dataloader, writer, audit_path=audit_path)
            committed_by_step.setdefault(event.global_step, []).append(
                event.batch_content_hash
            )

        target_step = min(committed_by_step)
        reconstructed = reconstruct_at_global_step(ledger_path, target_step)
        self.assertEqual(len(reconstructed), len(committed_by_step[target_step]))
        self.assertEqual(
            [event.batch_content_hash for event in reconstructed],
            committed_by_step[target_step],
        )
        for event in reconstructed:
            self.assertEqual(event.global_step, target_step)
            self.assertTrue(event.opus_decision_id.startswith("opus-"))

    def test_checkpoint_includes_ledger_offset_and_branch(self) -> None:
        ledger_path = self.temp_dir / "checkpoint_ledger.jsonl"
        writer = LedgerWriter.open(ledger_path)
        dataloader = LedgerBoundDataLoader(
            self.run_plan,
            microbatch_size=self.demo.training.microbatch_size,
            global_batch_size=self.demo.training.global_batch_size,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
        )
        audit_path = self.temp_dir / "audit_checkpoint.jsonl"
        self._commit_microbatch(dataloader, writer, audit_path=audit_path)
        self._commit_microbatch(dataloader, writer, audit_path=audit_path)

        payload = build_checkpoint_payload(
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            global_step=0,
            seed=self.run_kwargs["seed"],
            dataloader_state=dataloader.state(),
        )
        ckpt_dir = save_checkpoint(self.temp_dir / "checkpoints", payload)
        loaded = load_checkpoint(self.temp_dir / "checkpoints", global_step=0)

        self.assertEqual(loaded.ledger_offset, 1)
        self.assertEqual(loaded.branch_id, self.run_kwargs["branch_id"])
        self.assertEqual(loaded.run_id, self.run_kwargs["run_id"])
        self.assertTrue((ckpt_dir / "checkpoint.json").is_file())

        incomplete = payload.to_dict()
        del incomplete["ledger_offset"]
        with self.assertRaises(CheckpointError):
            from checkpoint.io import payload_from_dict

            payload_from_dict(incomplete)

    def test_dataloader_resumes_from_checkpoint_offset(self) -> None:
        ledger_path = self.temp_dir / "resume_ledger.jsonl"
        writer = LedgerWriter.open(ledger_path)
        audit_path = self.temp_dir / "audit_resume.jsonl"

        dataloader = LedgerBoundDataLoader(
            self.run_plan,
            microbatch_size=self.demo.training.microbatch_size,
            global_batch_size=self.demo.training.global_batch_size,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
        )

        for _ in range(2):
            self._commit_microbatch(dataloader, writer, audit_path=audit_path)

        # Peek at what the live loader would serve next; restore must reproduce it.
        expected_next = dataloader.next_microbatch()
        expected_offset = dataloader.ledger_offset

        payload = build_checkpoint_payload(
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
            global_step=dataloader.state().next_global_step,
            seed=self.run_kwargs["seed"],
            dataloader_state=dataloader.state(),
        )
        save_checkpoint(self.temp_dir / "checkpoints", payload)

        continued = LedgerBoundDataLoader(
            self.run_plan,
            microbatch_size=self.demo.training.microbatch_size,
            global_batch_size=self.demo.training.global_batch_size,
            run_id=self.run_kwargs["run_id"],
            branch_id=self.run_kwargs["branch_id"],
        )
        loaded = load_checkpoint(
            self.temp_dir / "checkpoints",
            global_step=payload.global_step,
        )
        continued.restore_state(dataloader_state_from_checkpoint(loaded))

        self.assertEqual(continued.ledger_offset, expected_offset)
        self.assertEqual(continued.next_ledger_offset, expected_offset + 1)
        self.assertEqual(
            continued.next_microbatch().candidate_id,
            expected_next.candidate_id,
        )
        self.assertEqual(
            [sample.sample_id for sample in continued.next_microbatch().samples],
            [sample.sample_id for sample in expected_next.samples],
        )


if __name__ == "__main__":
    unittest.main()
