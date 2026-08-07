"""Shard on-disk formats (D2: binary pretrain, JSONL agentic)."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Literal

from .errors import ShardError

BINARY_MAGIC = b"S6BIN"
BINARY_VERSION = 1
BINARY_HEADER = struct.Struct("<5sII")  # magic, version, token_count

CONTENT_HASH_PREFIX = "sha256:"


def content_hash(payload: bytes) -> str:
    """Return the canonical content hash for sealed shard bytes."""
    digest = hashlib.sha256(payload).hexdigest()
    return f"{CONTENT_HASH_PREFIX}{digest}"


def shard_id_from_content_hash(content_hash_value: str) -> str:
    """Derive a stable shard id from a content hash string."""
    if not content_hash_value.startswith(CONTENT_HASH_PREFIX):
        raise ShardError(f"invalid content hash: {content_hash_value}")
    digest = content_hash_value.removeprefix(CONTENT_HASH_PREFIX)
    return f"shard_{digest[:12]}"


def encode_binary_shard(token_ids: list[int]) -> bytes:
    """Pack token IDs into the pretrain binary shard format."""
    if not token_ids:
        raise ShardError("binary shard requires at least one token")
    for token_id in token_ids:
        if not isinstance(token_id, int) or token_id < 0:
            raise ShardError(f"invalid token id: {token_id!r}")
        if token_id >= 2**32:
            raise ShardError(f"token id out of uint32 range: {token_id}")
    header = BINARY_HEADER.pack(BINARY_MAGIC, BINARY_VERSION, len(token_ids))
    body = struct.pack(f"<{len(token_ids)}I", *token_ids)
    return header + body


def decode_binary_shard(payload: bytes) -> list[int]:
    """Decode a pretrain binary shard payload."""
    if len(payload) < BINARY_HEADER.size:
        raise ShardError("binary shard payload too short")
    magic, version, token_count = BINARY_HEADER.unpack_from(payload)
    if magic != BINARY_MAGIC:
        raise ShardError("binary shard has invalid magic header")
    if version != BINARY_VERSION:
        raise ShardError(f"unsupported binary shard version: {version}")
    expected_size = BINARY_HEADER.size + token_count * 4
    if len(payload) != expected_size:
        raise ShardError("binary shard payload size does not match header")
    if token_count == 0:
        raise ShardError("binary shard requires at least one token")
    token_ids = list(struct.unpack_from(f"<{token_count}I", payload, BINARY_HEADER.size))
    return token_ids


def encode_jsonl_shard(records: list[dict[str, Any]]) -> bytes:
    """Encode agentic shard records as deterministic JSONL bytes."""
    if not records:
        raise ShardError("jsonl shard requires at least one record")
    lines: list[str] = []
    for record in records:
        if "document_id" not in record or "token_ids" not in record:
            raise ShardError("jsonl shard record requires document_id and token_ids")
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def decode_jsonl_shard(payload: bytes) -> list[dict[str, Any]]:
    """Decode agentic JSONL shard bytes."""
    text = payload.decode("utf-8")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShardError(f"jsonl shard line {line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ShardError(f"jsonl shard line {line_number}: expected object")
        records.append(record)
    if not records:
        raise ShardError("jsonl shard requires at least one record")
    return records


ShardFormat = Literal["binary", "jsonl"]
