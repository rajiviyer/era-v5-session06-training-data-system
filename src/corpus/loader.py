"""Load toy corpus provenance and document metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import validate_document_record

PROVENANCE_KEYS = frozenset(
    {
        "id",
        "title",
        "source_url",
        "local_path",
        "format",
        "license",
        "track",
        "transforms",
        "cleaning_pipeline_hash",
    }
)


class CorpusError(ValueError):
    """Raised when corpus files are missing or invalid."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CorpusError(f"corpus file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise CorpusError(f"{path}:{line_number}: row must be a JSON object")
            records.append(record)
    return records


def validate_provenance_record(record: dict[str, Any]) -> dict[str, Any]:
    if set(record.keys()) != PROVENANCE_KEYS:
        raise CorpusError(f"provenance record has unexpected keys: {sorted(record.keys())}")
    source_id = record["id"]
    if not isinstance(source_id, str) or not source_id.strip():
        raise CorpusError("provenance id must be a non-empty string")
    transforms = record["transforms"]
    if not isinstance(transforms, list):
        raise CorpusError(f"provenance {source_id}: transforms must be a list")
    return record


def load_provenance(path: Path) -> dict[str, dict[str, Any]]:
    """Load provenance.jsonl indexed by source id."""
    registry: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        validated = validate_provenance_record(record)
        source_id = validated["id"]
        if source_id in registry:
            raise CorpusError(f"duplicate provenance id: {source_id}")
        registry[source_id] = validated
    if not registry:
        raise CorpusError(f"provenance registry is empty: {path}")
    return registry


def load_documents(
    path: Path,
    *,
    provenance: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate documents.jsonl."""
    provenance_ids = frozenset(provenance.keys()) if provenance is not None else None
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in _read_jsonl(path):
        validated = validate_document_record(record, provenance_ids=provenance_ids)
        document_id = validated["document_id"]
        if document_id in seen_ids:
            raise CorpusError(f"duplicate document_id: {document_id}")
        seen_ids.add(document_id)
        documents.append(validated)
    if not documents:
        raise CorpusError(f"document registry is empty: {path}")
    return documents


def load_corpus(corpus_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load provenance and documents from a toy corpus directory."""
    root = corpus_dir.resolve()
    provenance = load_provenance(root / "provenance.jsonl")
    documents = load_documents(root / "documents.jsonl", provenance=provenance)
    return provenance, documents
