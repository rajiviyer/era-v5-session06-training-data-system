"""Eval registry schema validation (P4-T01)."""

from __future__ import annotations

from typing import Any

from .errors import FirewallError
from .types import EvalRegistry, EvalRegistryEntry

REGISTRY_TYPE = "eval"
SCHEMA_VERSION = "1.0"


def validate_eval_registry(registry: EvalRegistry) -> EvalRegistry:
    """Validate an in-memory eval registry."""
    if registry.registry_type != REGISTRY_TYPE:
        raise FirewallError(f"registry_type must be {REGISTRY_TYPE!r}")
    if registry.schema_version != SCHEMA_VERSION:
        raise FirewallError(f"unsupported schema_version: {registry.schema_version}")
    if not registry.entries:
        raise FirewallError("eval registry must contain at least one entry")

    seen_entry_ids: set[str] = set()
    never_train_count = 0
    for entry in registry.entries:
        if entry.entry_id in seen_entry_ids:
            raise FirewallError(f"duplicate entry_id: {entry.entry_id}")
        seen_entry_ids.add(entry.entry_id)
        if not entry.document_id.strip():
            raise FirewallError("document_id must be non-empty")
        if not entry.benchmark_id.strip():
            raise FirewallError("benchmark_id must be non-empty")
        if not entry.content_hash.strip():
            raise FirewallError("content_hash must be non-empty")
        if entry.never_train:
            never_train_count += 1

    if never_train_count == 0:
        raise FirewallError("eval registry must include at least one never_train entry")
    return registry


def eval_registry_from_dict(raw: dict[str, Any]) -> EvalRegistry:
    """Parse eval_registry.json into typed records."""
    try:
        entries_raw = raw["entries"]
        if not isinstance(entries_raw, list) or not entries_raw:
            raise FirewallError("entries must be a non-empty list")
        entries = tuple(
            EvalRegistryEntry(
                entry_id=str(item["entry_id"]),
                document_id=str(item["document_id"]),
                shard_id=str(item["shard_id"]) if item.get("shard_id") is not None else None,
                never_train=bool(item["never_train"]),
                benchmark_id=str(item["benchmark_id"]),
                content_hash=str(item["content_hash"]),
                canary_strings=tuple(str(value) for value in item["canary_strings"]),
            )
            for item in entries_raw
        )
        registry = EvalRegistry(
            schema_version=str(raw["schema_version"]),
            registry_type=str(raw["registry_type"]),
            entries=entries,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FirewallError(f"invalid eval registry: {exc}") from exc
    return validate_eval_registry(registry)
