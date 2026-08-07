"""Shard manifest build, validate, and write (P1-T06)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .builder import BuiltShard
from .errors import ShardError
from .io import write_json_atomic

REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "shard_id",
        "source_ids",
        "document_ids",
        "tokenizer_hash",
        "content_hash",
        "token_count",
        "capability_lane",
        "language",
        "script",
        "license_tier",
        "cleaning_pipeline_hash",
        "dedup_status",
        "pii_screen_status",
        "eval_overlap_status",
        "parent_manifest_ids",
        "admission",
    }
)

ADMISSION_STATUSES = frozenset({"admitted", "blocked"})


def _aggregate_field(documents: list[dict[str, Any]], field: str) -> str:
    values = {doc[field] for doc in documents}
    if len(values) == 1:
        return values.pop()
    return "multi"


def _aggregate_eval_overlap(documents: list[dict[str, Any]]) -> str:
    if any(doc["eval_overlap_status"] == "overlap_detected" for doc in documents):
        return "overlap_detected"
    return "clear"


def _aggregate_license_tier(documents: list[dict[str, Any]]) -> str:
    tiers = {doc["license_tier"] for doc in documents}
    if tiers == {"safe"}:
        return "safe"
    if "blocked" in tiers:
        return "blocked"
    return "needs_review"


def build_shard_manifest(
    built: BuiltShard,
    documents: list[dict[str, Any]],
    *,
    admission: str = "blocked",
) -> dict[str, Any]:
    """Build a shard manifest from build output and source document metadata."""
    if admission not in ADMISSION_STATUSES:
        raise ShardError(f"invalid admission status: {admission}")

    ordered_docs = sorted(documents, key=lambda doc: doc["document_id"])
    expected_ids = set(built.document_ids)
    actual_ids = {doc["document_id"] for doc in ordered_docs}
    if expected_ids != actual_ids:
        raise ShardError(
            f"document_ids mismatch for {built.shard_id}: "
            f"expected {sorted(expected_ids)}, got {sorted(actual_ids)}"
        )

    return {
        "shard_id": built.shard_id,
        "source_ids": list(built.source_ids),
        "document_ids": list(built.document_ids),
        "tokenizer_hash": built.tokenizer_hash,
        "content_hash": built.content_hash,
        "token_count": built.token_count,
        "capability_lane": built.capability_lane,
        "language": _aggregate_field(ordered_docs, "language"),
        "script": _aggregate_field(ordered_docs, "script"),
        "license_tier": _aggregate_license_tier(ordered_docs),
        "cleaning_pipeline_hash": _aggregate_field(ordered_docs, "cleaning_pipeline_hash"),
        "dedup_status": _aggregate_field(ordered_docs, "dedup_status"),
        "pii_screen_status": _aggregate_field(ordered_docs, "pii_screen_status"),
        "eval_overlap_status": _aggregate_eval_overlap(ordered_docs),
        "parent_manifest_ids": [],
        "admission": admission,
    }


def validate_shard_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate manifest shape and required fields."""
    if set(manifest.keys()) != REQUIRED_MANIFEST_KEYS:
        missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
        extra = set(manifest.keys()) - REQUIRED_MANIFEST_KEYS
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if extra:
            details.append(f"unexpected keys: {sorted(extra)}")
        raise ShardError("; ".join(details))

    if manifest["admission"] not in ADMISSION_STATUSES:
        raise ShardError(f"invalid admission: {manifest['admission']}")

    token_count = manifest["token_count"]
    if not isinstance(token_count, int) or token_count <= 0:
        raise ShardError("token_count must be a positive integer")

    if not isinstance(manifest["tokenizer_hash"], str) or not manifest["tokenizer_hash"].startswith("tok_"):
        raise ShardError("tokenizer_hash must be a tok_* string")

    content_hash_value = manifest["content_hash"]
    if not isinstance(content_hash_value, str) or not content_hash_value.startswith("sha256:"):
        raise ShardError("content_hash must use sha256: prefix")

    for list_field in ("source_ids", "document_ids", "parent_manifest_ids"):
        value = manifest[list_field]
        if not isinstance(value, list):
            raise ShardError(f"{list_field} must be a list")

    return manifest


def manifest_path_for_shard(manifests_dir: Path, shard_id: str) -> Path:
    return manifests_dir / f"{shard_id}.json"


def write_shard_manifest(manifests_dir: Path, manifest: dict[str, Any]) -> Path:
    """Validate and atomically write one shard manifest."""
    validated = validate_shard_manifest(manifest)
    path = manifest_path_for_shard(manifests_dir, validated["shard_id"])
    write_json_atomic(path, validated)
    return path


def load_shard_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a shard manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShardError(f"shard manifest must be a JSON object: {path}")
    return validate_shard_manifest(payload)
