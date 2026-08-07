"""Build and validate tokenizer_manifest.json (P1-T03)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import TokenizerManifestError
from .hash import compute_tokenizer_hash_from_artifact

MANIFEST_FILENAME = "tokenizer_manifest.json"

REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "artifact",
        "frozen",
        "manifest_type",
        "merge_count",
        "model_type",
        "pre_tokenizer",
        "special_tokens",
        "tokenizer_hash",
        "vocab_size",
    }
)


def build_tokenizer_manifest(artifact_path: Path) -> dict[str, Any]:
    """Build a tokenizer manifest from a Hugging Face tokenizers JSON artifact."""
    resolved = artifact_path.resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    model = data.get("model") or {}
    if not isinstance(model, dict):
        raise TokenizerManifestError(f"tokenizer model section must be an object: {resolved}")

    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []
    pre_tokenizer = data.get("pre_tokenizer") or {}

    special_tokens = sorted(
        token
        for token in vocab
        if isinstance(token, str) and token.startswith("<") and token.endswith(">")
    )
    if not special_tokens:
        raise TokenizerManifestError(f"no special tokens found in vocab: {resolved}")

    return {
        "manifest_type": "tokenizer",
        "artifact": resolved.name,
        "tokenizer_hash": compute_tokenizer_hash_from_artifact(resolved),
        "model_type": model.get("type", "unknown"),
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": special_tokens,
        "pre_tokenizer": pre_tokenizer.get("type"),
        "frozen": True,
    }


def validate_tokenizer_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate manifest shape and required fields."""
    if set(manifest.keys()) != REQUIRED_MANIFEST_KEYS:
        missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
        extra = set(manifest.keys()) - REQUIRED_MANIFEST_KEYS
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if extra:
            details.append(f"unexpected keys: {sorted(extra)}")
        raise TokenizerManifestError("; ".join(details))

    if manifest["manifest_type"] != "tokenizer":
        raise TokenizerManifestError("manifest_type must be 'tokenizer'")
    if not isinstance(manifest["tokenizer_hash"], str) or not manifest["tokenizer_hash"].startswith("tok_"):
        raise TokenizerManifestError("tokenizer_hash must be a tok_* string")
    if not isinstance(manifest["vocab_size"], int) or manifest["vocab_size"] <= 0:
        raise TokenizerManifestError("vocab_size must be a positive integer")
    if not isinstance(manifest["merge_count"], int) or manifest["merge_count"] < 0:
        raise TokenizerManifestError("merge_count must be a non-negative integer")
    if manifest["frozen"] is not True:
        raise TokenizerManifestError("frozen must be true")
    if not isinstance(manifest["special_tokens"], list) or not manifest["special_tokens"]:
        raise TokenizerManifestError("special_tokens must be a non-empty list")
    return manifest


def write_tokenizer_manifest(
    artifact_path: Path,
    manifest_path: Path | None = None,
) -> Path:
    """Write tokenizer_manifest.json and return its path."""
    resolved_artifact = artifact_path.resolve()
    manifest = validate_tokenizer_manifest(build_tokenizer_manifest(resolved_artifact))
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else resolved_artifact.parent / MANIFEST_FILENAME
    )
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_manifest


def load_tokenizer_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate a tokenizer manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TokenizerManifestError(f"tokenizer manifest must be a JSON object: {manifest_path}")
    return validate_tokenizer_manifest(payload)
