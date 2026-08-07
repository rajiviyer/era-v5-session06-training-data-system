"""Shard registry index for admitted and blocked shards (P1-T07)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_json_atomic

REGISTRY_FILENAME = "shard_registry.json"


def build_registry_index(
    manifests: list[dict[str, Any]],
    *,
    tokenizer_hash: str,
) -> dict[str, Any]:
    """Build a registry index partitioning shards by admission status."""
    admitted: list[str] = []
    blocked: list[str] = []
    for manifest in manifests:
        shard_id = manifest["shard_id"]
        if manifest["admission"] == "admitted":
            admitted.append(shard_id)
        else:
            blocked.append(shard_id)
    return {
        "registry_type": "shard",
        "tokenizer_hash": tokenizer_hash,
        "admitted_shard_ids": sorted(admitted),
        "blocked_shard_ids": sorted(blocked),
        "shard_count": len(manifests),
    }


def write_registry_index(manifests_dir: Path, index: dict[str, Any]) -> Path:
    """Atomically write shard_registry.json."""
    path = manifests_dir / REGISTRY_FILENAME
    write_json_atomic(path, index)
    return path


def load_registry_index(path: Path) -> dict[str, Any]:
    """Load shard registry index."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"shard registry must be a JSON object: {path}")
    return payload
