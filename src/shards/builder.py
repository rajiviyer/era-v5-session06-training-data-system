"""Build immutable tokenized shards from corpus documents (P1-T05)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tokenizer.frozen import FrozenTokenizer

from .format import (
    ShardFormat,
    content_hash,
    decode_binary_shard,
    decode_jsonl_shard,
    encode_binary_shard,
    encode_jsonl_shard,
    shard_id_from_content_hash,
)
from .io import write_bytes_atomic

PRETRAIN_DATA_TYPE = "pretrain"
AGENTIC_DATA_TYPE = "agentic"


@dataclass(frozen=True)
class BuiltShard:
    """Result of tokenizing and sealing one shard file."""

    shard_id: str
    path: Path
    content_hash: str
    format: ShardFormat
    token_count: int
    document_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    capability_lane: str
    tokenizer_hash: str
    data_type: str


def _group_key(document: dict[str, Any]) -> tuple[str, ...]:
    if document.get("never_train"):
        return ("never_train", document["document_id"])
    if document["data_type"] == AGENTIC_DATA_TYPE:
        return ("agentic",)
    return ("pretrain", document["capability_lane"])


def _sort_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(documents, key=lambda doc: doc["document_id"])


def _encode_pretrain_group(documents: list[dict[str, Any]], tokenizer: FrozenTokenizer) -> bytes:
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document["text"]))
    return encode_binary_shard(token_ids)


def _encode_agentic_group(documents: list[dict[str, Any]], tokenizer: FrozenTokenizer) -> bytes:
    records: list[dict[str, Any]] = []
    for document in documents:
        records.append(
            {
                "document_id": document["document_id"],
                "token_ids": tokenizer.encode(document["text"]),
            }
        )
    return encode_jsonl_shard(records)


def _token_count_for_payload(payload: bytes, shard_format: ShardFormat) -> int:
    if shard_format == "binary":
        return len(decode_binary_shard(payload))
    records = decode_jsonl_shard(payload)
    return sum(len(record["token_ids"]) for record in records)


def _filename_for_group(group: tuple[str, ...]) -> str:
    if group[0] == "never_train":
        return f"{group[1]}.bin"
    if group[0] == "agentic":
        return "agentic.jsonl"
    return f"pretrain_{group[1]}.bin"


def build_shard_group(
    documents: list[dict[str, Any]],
    *,
    tokenizer: FrozenTokenizer,
    output_dir: Path,
    group: tuple[str, ...],
) -> BuiltShard:
    """Tokenize one document group and write a sealed shard file."""
    ordered = _sort_documents(documents)
    if group[0] == "agentic":
        payload = _encode_agentic_group(ordered, tokenizer)
        shard_format: ShardFormat = "jsonl"
        data_type = AGENTIC_DATA_TYPE
    else:
        payload = _encode_pretrain_group(ordered, tokenizer)
        shard_format = "binary"
        data_type = PRETRAIN_DATA_TYPE

    hash_value = content_hash(payload)
    shard_id = shard_id_from_content_hash(hash_value)
    filename = _filename_for_group(group)
    path = output_dir / filename
    write_bytes_atomic(path, payload)

    capability_lane = ordered[0]["capability_lane"]
    source_ids = tuple(sorted({doc["source_id"] for doc in ordered}))

    return BuiltShard(
        shard_id=shard_id,
        path=path,
        content_hash=hash_value,
        format=shard_format,
        token_count=_token_count_for_payload(payload, shard_format),
        document_ids=tuple(doc["document_id"] for doc in ordered),
        source_ids=source_ids,
        capability_lane=capability_lane,
        tokenizer_hash=tokenizer.tokenizer_hash,
        data_type=data_type,
    )


def build_shards(
    documents: list[dict[str, Any]],
    *,
    tokenizer: FrozenTokenizer,
    output_dir: Path,
) -> list[BuiltShard]:
    """Build all shards for ready corpus documents."""
    ready = [doc for doc in documents if doc.get("content_status") == "ready"]
    if not ready:
        raise ValueError("no ready documents to shard")

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for document in ready:
        groups.setdefault(_group_key(document), []).append(document)

    built: list[BuiltShard] = []
    for group in sorted(groups):
        built.append(
            build_shard_group(
                groups[group],
                tokenizer=tokenizer,
                output_dir=output_dir,
                group=group,
            )
        )
    return built
