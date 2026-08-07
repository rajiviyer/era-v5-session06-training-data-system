"""End-to-end shard build with manifests and admission (P1-T06–T08)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizer.frozen import FrozenTokenizer

from .admission import apply_admission
from .builder import BuiltShard, build_shards
from .manifest import build_shard_manifest, write_shard_manifest
from .registry import build_registry_index, write_registry_index


@dataclass(frozen=True)
class ShardBuildResult:
    shards: tuple[BuiltShard, ...]
    manifests: tuple[dict[str, Any], ...]
    registry: dict[str, Any]
    shards_dir: Path
    manifests_dir: Path


def _documents_for_shard(
    built: BuiltShard,
    documents_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [documents_by_id[document_id] for document_id in built.document_ids]


def build_shards_with_manifests(
    documents: list[dict[str, Any]],
    *,
    tokenizer: FrozenTokenizer,
    shards_dir: Path,
    manifests_dir: Path,
) -> ShardBuildResult:
    """Build shard files, write manifests, evaluate admission, and write registry."""
    built_shards = build_shards(documents, tokenizer=tokenizer, output_dir=shards_dir)
    documents_by_id = {doc["document_id"]: doc for doc in documents if doc.get("content_status") == "ready"}

    manifests: list[dict[str, Any]] = []
    for built in built_shards:
        shard_docs = _documents_for_shard(built, documents_by_id)
        manifest = build_shard_manifest(built, shard_docs)
        manifest = apply_admission(manifest)
        write_shard_manifest(manifests_dir, manifest)
        manifests.append(manifest)

    registry = build_registry_index(manifests, tokenizer_hash=tokenizer.tokenizer_hash)
    write_registry_index(manifests_dir, registry)

    return ShardBuildResult(
        shards=tuple(built_shards),
        manifests=tuple(manifests),
        registry=registry,
        shards_dir=shards_dir,
        manifests_dir=manifests_dir,
    )
