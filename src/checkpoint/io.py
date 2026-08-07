"""Checkpoint save/load bound to ledger offset (P6-T05–T06)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ledger.types import DataLoaderState

from .errors import CheckpointError
from .types import SCHEMA_VERSION, CheckpointPayload, checkpoint_dir_for_step, checkpoint_id_for_step


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError(
            "PyTorch is required for checkpoint tensor save/load; install torch>=2.3"
        ) from exc
    return torch


def capture_rng_state() -> dict[str, Any]:
    """Capture Python and PyTorch RNG states."""
    state: dict[str, Any] = {"python_random": random.getstate()}
    try:
        torch = _torch_module()
    except CheckpointError:
        return state
    state["torch"] = torch.get_rng_state().tolist()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state().tolist()
    return state


def _as_python_random_state(raw: Any) -> tuple[int, tuple[int, ...], Any]:
    """Coerce a persisted Python RNG state back into the shape `setstate` accepts.

    `random.getstate()` returns nested tuples, but a checkpoint round-trips through JSON,
    which has no tuple type: the state comes back as lists and `setstate` rejects it with
    "state vector must be a tuple". Handles the in-memory case unchanged.
    """
    try:
        version, internal, gauss_next = raw
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"malformed python_random state: {raw!r}") from exc
    return int(version), tuple(int(value) for value in internal), gauss_next


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by capture_rng_state."""
    python_state = state.get("python_random")
    if python_state is None:
        raise CheckpointError("rng_state must include python_random")
    random.setstate(_as_python_random_state(python_state))

    torch_state = state.get("torch")
    if torch_state is None:
        return
    torch = _torch_module()
    torch.set_rng_state(torch.as_tensor(torch_state, dtype=torch.uint8))
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.as_tensor(cuda_state, dtype=torch.uint8))


def build_checkpoint_payload(
    *,
    run_id: str,
    branch_id: str,
    global_step: int,
    seed: int,
    dataloader_state: DataLoaderState,
    model_state: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> CheckpointPayload:
    """Build a checkpoint payload from run and dataloader state."""
    checkpoint_id = checkpoint_id_for_step(global_step)
    return CheckpointPayload(
        schema_version=SCHEMA_VERSION,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        branch_id=branch_id,
        global_step=global_step,
        ledger_offset=dataloader_state.ledger_offset,
        seed=seed,
        dataloader_version=dataloader_state.dataloader_version,
        next_global_step=dataloader_state.next_global_step,
        next_microbatch_index=dataloader_state.next_microbatch_index,
        rng_state=capture_rng_state(),
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


def payload_from_dict(raw: dict[str, Any]) -> CheckpointPayload:
    """Parse checkpoint metadata from JSON."""
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported schema_version: {version}")

    required = (
        "checkpoint_id",
        "run_id",
        "branch_id",
        "global_step",
        "ledger_offset",
        "seed",
        "dataloader_version",
        "next_global_step",
        "next_microbatch_index",
        "rng_state",
    )
    for key in required:
        if key not in raw:
            raise CheckpointError(f"checkpoint missing required key: {key}")

    if raw["ledger_offset"] is None:
        raise CheckpointError("checkpoint without ledger_offset is incomplete")

    ledger_offset = int(raw["ledger_offset"])
    if ledger_offset < -1:
        raise CheckpointError("ledger_offset must be >= -1")

    rng_state = raw["rng_state"]
    if not isinstance(rng_state, dict):
        raise CheckpointError("rng_state must be an object")

    return CheckpointPayload(
        schema_version=SCHEMA_VERSION,
        checkpoint_id=str(raw["checkpoint_id"]),
        run_id=str(raw["run_id"]),
        branch_id=str(raw["branch_id"]),
        global_step=int(raw["global_step"]),
        ledger_offset=ledger_offset,
        seed=int(raw["seed"]),
        dataloader_version=str(raw["dataloader_version"]),
        next_global_step=int(raw["next_global_step"]),
        next_microbatch_index=int(raw["next_microbatch_index"]),
        rng_state=rng_state,
    )


def save_checkpoint(
    base_dir: Path,
    payload: CheckpointPayload,
    *,
    model_state: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> Path:
    """Save checkpoint metadata and optional tensor states to ckpt-{step}/."""
    from shards.io import write_json_atomic

    if payload.ledger_offset < -1:
        raise CheckpointError("checkpoint without valid ledger_offset is incomplete")

    target_dir = checkpoint_dir_for_step(base_dir, payload.global_step)
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata = payload
    if model_state is not None or optimizer_state is not None or scheduler_state is not None:
        metadata = CheckpointPayload(
            schema_version=payload.schema_version,
            checkpoint_id=payload.checkpoint_id,
            run_id=payload.run_id,
            branch_id=payload.branch_id,
            global_step=payload.global_step,
            ledger_offset=payload.ledger_offset,
            seed=payload.seed,
            dataloader_version=payload.dataloader_version,
            next_global_step=payload.next_global_step,
            next_microbatch_index=payload.next_microbatch_index,
            rng_state=payload.rng_state,
            model_state=model_state if model_state is not None else payload.model_state,
            optimizer_state=optimizer_state
            if optimizer_state is not None
            else payload.optimizer_state,
            scheduler_state=scheduler_state
            if scheduler_state is not None
            else payload.scheduler_state,
        )

    write_json_atomic(target_dir / "checkpoint.json", metadata.to_dict())

    if metadata.model_state is not None:
        torch = _torch_module()
        torch.save(metadata.model_state, target_dir / "model.pt")
    if metadata.optimizer_state is not None:
        torch = _torch_module()
        torch.save(metadata.optimizer_state, target_dir / "optimizer.pt")
    if metadata.scheduler_state is not None:
        torch = _torch_module()
        torch.save(metadata.scheduler_state, target_dir / "scheduler.pt")

    return target_dir.resolve()


def load_checkpoint(base_dir: Path, global_step: int) -> CheckpointPayload:
    """Load checkpoint metadata from ckpt-{step}/checkpoint.json."""
    target_dir = checkpoint_dir_for_step(base_dir, global_step)
    metadata_path = target_dir / "checkpoint.json"
    if not metadata_path.is_file():
        raise CheckpointError(f"checkpoint not found: {metadata_path}")

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint.json must be a JSON object")

    payload = payload_from_dict(raw)

    model_path = target_dir / "model.pt"
    optimizer_path = target_dir / "optimizer.pt"
    scheduler_path = target_dir / "scheduler.pt"

    model_state = None
    optimizer_state = None
    scheduler_state = None
    if model_path.is_file() or optimizer_path.is_file() or scheduler_path.is_file():
        torch = _torch_module()
        if model_path.is_file():
            model_state = torch.load(model_path, weights_only=False)
        if optimizer_path.is_file():
            optimizer_state = torch.load(optimizer_path, weights_only=False)
        if scheduler_path.is_file():
            scheduler_state = torch.load(scheduler_path, weights_only=False)

    return CheckpointPayload(
        schema_version=payload.schema_version,
        checkpoint_id=payload.checkpoint_id,
        run_id=payload.run_id,
        branch_id=payload.branch_id,
        global_step=payload.global_step,
        ledger_offset=payload.ledger_offset,
        seed=payload.seed,
        dataloader_version=payload.dataloader_version,
        next_global_step=payload.next_global_step,
        next_microbatch_index=payload.next_microbatch_index,
        rng_state=payload.rng_state,
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


def dataloader_state_from_checkpoint(payload: CheckpointPayload) -> DataLoaderState:
    """Restore ledger-bound dataloader state from a checkpoint payload."""
    if payload.ledger_offset < -1:
        raise CheckpointError("checkpoint without ledger_offset is incomplete")
    return DataLoaderState(
        run_id=payload.run_id,
        branch_id=payload.branch_id,
        ledger_offset=payload.ledger_offset,
        next_global_step=payload.next_global_step,
        next_microbatch_index=payload.next_microbatch_index,
        dataloader_version=payload.dataloader_version,
    )
