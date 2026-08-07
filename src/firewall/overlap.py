"""Eval overlap checks (P4-T03)."""

from __future__ import annotations

from typing import Any

from .types import BatchCandidate, EvalRegistry, EvalRegistryEntry


def check_overlaps(
    candidate: BatchCandidate,
    registry: EvalRegistry,
    *,
    documents_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return firewall reasons and matched eval entry IDs."""
    reasons: list[str] = []
    matched: list[str] = []

    never_train_shards = {
        entry.shard_id for entry in registry.entries if entry.never_train and entry.shard_id
    }
    never_train_documents = {entry.document_id for entry in registry.entries if entry.never_train}
    blocked_hashes = {entry.content_hash for entry in registry.entries if entry.never_train}
    blocked_hashes |= {_bare_hash(value) for value in list(blocked_hashes)}

    for shard_id in candidate.shard_ids:
        if shard_id in never_train_shards:
            reasons.append("never_train_shard_id")
            matched.extend(
                entry.entry_id
                for entry in registry.entries
                if entry.shard_id == shard_id and entry.never_train
            )

    for sample_id in candidate.sample_ids:
        if sample_id in never_train_documents:
            reasons.append("never_train_document_id")
            matched.extend(
                entry.entry_id
                for entry in registry.entries
                if entry.document_id == sample_id and entry.never_train
            )

    for content_hash in candidate.content_hashes:
        normalized = content_hash if content_hash.startswith("sha256:") else f"sha256:{content_hash}"
        bare = _bare_hash(normalized)
        if normalized in blocked_hashes or bare in blocked_hashes:
            reasons.append("exact_content_hash")
            matched.extend(
                entry.entry_id
                for entry in registry.entries
                if _bare_hash(entry.content_hash) == bare
            )

    if documents_by_id is not None:
        for sample_id in candidate.sample_ids:
            document = documents_by_id.get(sample_id)
            if document is None:
                continue
            text = str(document.get("text", ""))
            for entry in registry.entries:
                if _contains_canary(text, entry):
                    reasons.append("canary_string_match")
                    matched.append(entry.entry_id)
                if _benchmark_overlap_stub(text, entry):
                    reasons.append("benchmark_overlap_stub")
                    matched.append(entry.entry_id)

    return _dedupe(reasons), _dedupe(matched)


def _contains_canary(text: str, entry: EvalRegistryEntry) -> bool:
    lowered = text.lower()
    return any(canary.lower() in lowered for canary in entry.canary_strings if canary)


def _benchmark_overlap_stub(text: str, entry: EvalRegistryEntry) -> bool:
    if not entry.benchmark_id.startswith("mmlu"):
        return False
    lowered = text.lower()
    return "mmlu" in lowered and "holdout" in lowered


def _bare_hash(value: str) -> str:
    return value.removeprefix("sha256:")


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
