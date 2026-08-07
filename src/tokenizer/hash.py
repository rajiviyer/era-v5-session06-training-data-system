"""Tokenizer fingerprint and hash persistence (P1-T02)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HASH_PREFIX = "tok_"
HASH_HEX_LEN = 12
HASH_SIDECAR = "tokenizer_hash.json"


def _fingerprint(data: dict[str, Any]) -> dict[str, Any]:
    """Canonical vocab + merges + special-token payload for hashing."""
    model = data.get("model") or {}
    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []

    special_tokens: set[str] = set()
    for field in ("unk_token", "bos_token", "eos_token", "pad_token"):
        value = model.get(field)
        if isinstance(value, str) and value:
            special_tokens.add(value)
    for entry in data.get("added_tokens") or []:
        if isinstance(entry, dict) and entry.get("special"):
            content = entry.get("content")
            if isinstance(content, str) and content:
                special_tokens.add(content)

    return {
        "merges": merges,
        "special_tokens": sorted(special_tokens),
        "vocab": sorted(vocab.items(), key=lambda item: item[0]),
    }


def compute_tokenizer_hash_from_artifact(artifact_path: Path) -> str:
    """Return a stable `tok_<hex>` hash for the tokenizer artifact."""
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    canonical = json.dumps(_fingerprint(data), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest[:HASH_HEX_LEN]}"


def persist_tokenizer_hash(
    artifact_path: Path,
    hash_path: Path | None = None,
) -> str:
    """Compute and write the tokenizer hash sidecar JSON."""
    resolved_artifact = artifact_path.resolve()
    tokenizer_hash = compute_tokenizer_hash_from_artifact(resolved_artifact)
    resolved_hash_path = (
        hash_path.resolve() if hash_path is not None else resolved_artifact.parent / HASH_SIDECAR
    )
    payload = {
        "artifact": resolved_artifact.name,
        "tokenizer_hash": tokenizer_hash,
    }
    resolved_hash_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_hash_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tokenizer_hash


def load_persisted_tokenizer_hash(hash_path: Path) -> str:
    """Load a previously persisted tokenizer hash."""
    payload = json.loads(hash_path.read_text(encoding="utf-8"))
    tokenizer_hash = payload.get("tokenizer_hash")
    if not isinstance(tokenizer_hash, str) or not tokenizer_hash.startswith(HASH_PREFIX):
        raise ValueError(f"invalid tokenizer_hash in {hash_path}")
    return tokenizer_hash
