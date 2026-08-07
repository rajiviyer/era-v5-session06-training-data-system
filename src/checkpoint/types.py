"""Checkpoint payload types (P6-T05)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
CHECKPOINT_DIR_PREFIX = "ckpt-"


@dataclass(frozen=True)
class CheckpointPayload:
    """Checkpoint metadata bound to ledger offset and branch ID."""

    schema_version: str
    checkpoint_id: str
    run_id: str
    branch_id: str
    global_step: int
    ledger_offset: int
    seed: int
    dataloader_version: str
    next_global_step: int
    next_microbatch_index: int
    rng_state: dict[str, Any]
    model_state: dict[str, Any] | None = None
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint.json.

        Model, optimizer, and scheduler states are deliberately excluded: they hold
        tensors, which are not JSON-serializable, and `save_checkpoint` writes them to
        `model.pt` / `optimizer.pt` / `scheduler.pt` beside this file. `tensor_files`
        records what a reader should expect to find.
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "ledger_offset": self.ledger_offset,
            "seed": self.seed,
            "dataloader_version": self.dataloader_version,
            "next_global_step": self.next_global_step,
            "next_microbatch_index": self.next_microbatch_index,
            "rng_state": self.rng_state,
            "tensor_files": self.tensor_files,
        }
        return payload

    @property
    def tensor_files(self) -> list[str]:
        """Sidecar files this checkpoint writes alongside checkpoint.json."""
        files = []
        if self.model_state is not None:
            files.append("model.pt")
        if self.optimizer_state is not None:
            files.append("optimizer.pt")
        if self.scheduler_state is not None:
            files.append("scheduler.pt")
        return files


def checkpoint_id_for_step(global_step: int) -> str:
    return f"{CHECKPOINT_DIR_PREFIX}{global_step:05d}"


def checkpoint_dir_for_step(base_dir: Path, global_step: int) -> Path:
    return base_dir / checkpoint_id_for_step(global_step)
