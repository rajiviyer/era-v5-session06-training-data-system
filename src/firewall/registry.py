"""Build and persist eval_registry.json (P4-T01, P4-T02)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from batch.types import Batch
from shards.manifest import load_shard_manifest
from shards.registry import REGISTRY_FILENAME as SHARD_REGISTRY_FILENAME

from .errors import FirewallError
from .schema import REGISTRY_TYPE, SCHEMA_VERSION, validate_eval_registry
from .types import BatchCandidate, EvalRegistry, EvalRegistryEntry

REGISTRY_FILENAME = "eval_registry.json"

DEFAULT_CANARIES: dict[str, tuple[str, ...]] = {
    "doc-eval-001": (
        "MMLU holdout mirror",
        "Which planet is known as the Red Planet?",
    ),
}

DEFAULT_BENCHMARK_IDS: dict[str, str] = {
    "doc-eval-001": "mmlu_holdout_mirror",
}


def build_eval_registry(
    documents: list[dict[str, Any]],
    *,
    manifests_dir: Path | None = None,
) -> EvalRegistry:
    """Build eval registry entries from never_train corpus rows and shard manifests."""
    never_train_docs = [doc for doc in documents if doc.get("never_train")]
    if not never_train_docs:
        raise FirewallError("corpus must include at least one never_train document")

    shard_id_by_document = _shard_ids_by_document(manifests_dir)

    entries: list[EvalRegistryEntry] = []
    for document in sorted(never_train_docs, key=lambda row: row["document_id"]):
        document_id = document["document_id"]
        entries.append(
            EvalRegistryEntry(
                entry_id=f"eval-{document_id}",
                document_id=document_id,
                shard_id=shard_id_by_document.get(document_id),
                never_train=True,
                benchmark_id=DEFAULT_BENCHMARK_IDS.get(document_id, f"benchmark_{document_id}"),
                content_hash=_normalize_content_hash(document["content_sha256"]),
                canary_strings=DEFAULT_CANARIES.get(document_id, (_title_canary(document),)),
            )
        )

    return validate_eval_registry(
        EvalRegistry(
            schema_version=SCHEMA_VERSION,
            registry_type=REGISTRY_TYPE,
            entries=tuple(entries),
        )
    )


def _shard_ids_by_document(manifests_dir: Path | None) -> dict[str, str]:
    if manifests_dir is None or not manifests_dir.is_dir():
        return {}

    # `shard_*.json` rather than `*.json`: manifests/ also holds the registry index and
    # the tokenizer manifest, neither of which is a shard manifest.
    mapping: dict[str, str] = {}
    for path in sorted(manifests_dir.glob("shard_*.json")):
        if path.name == SHARD_REGISTRY_FILENAME:
            continue
        manifest = load_shard_manifest(path)
        for document_id in manifest["document_ids"]:
            mapping[document_id] = manifest["shard_id"]
    return mapping


def _title_canary(document: dict[str, Any]) -> str:
    title = str(document.get("title", "")).strip()
    if title:
        return title
    text = str(document.get("text", ""))
    return text[:48].strip()


def _normalize_content_hash(value: str) -> str:
    if value.startswith("sha256:"):
        return value
    return f"sha256:{value}"


def write_eval_registry(path: Path, registry: EvalRegistry) -> Path:
    """Atomically write eval_registry.json."""
    from shards.io import write_json_atomic

    validated = validate_eval_registry(registry)
    write_json_atomic(path.resolve(), validated.to_dict())
    return path.resolve()


def load_eval_registry(path: Path) -> EvalRegistry:
    """Load eval_registry.json."""
    if not path.is_file():
        raise FirewallError(f"eval registry not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FirewallError("eval registry must be a JSON object")
    from .schema import eval_registry_from_dict

    return eval_registry_from_dict(raw)


def candidate_from_planned_samples(
    *,
    candidate_id: str,
    global_step: int,
    sample_ids: list[str] | tuple[str, ...],
    shard_ids: list[str] | tuple[str, ...],
    documents_by_id: dict[str, dict[str, Any]],
    batch: Batch | None = None,
) -> BatchCandidate:
    """Build a firewall candidate from planner output.

    Pass `batch` once the microbatch is assembled so `assert_no_eval_loss` can inspect
    the real loss mask instead of trusting sample IDs alone.
    """
    hashes: list[str] = []
    for sample_id in sample_ids:
        document = documents_by_id.get(sample_id)
        if document is None:
            raise FirewallError(f"unknown sample_id in candidate: {sample_id}")
        hashes.append(_normalize_content_hash(document["content_sha256"]))
    return BatchCandidate(
        candidate_id=candidate_id,
        global_step=global_step,
        sample_ids=tuple(sample_ids),
        shard_ids=tuple(shard_ids),
        content_hashes=tuple(hashes),
        batch=batch,
    )
