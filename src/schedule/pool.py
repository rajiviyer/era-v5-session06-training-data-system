"""Admitted shard sample pool for planning (P3-T04–T06)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shards.manifest import load_shard_manifest, manifest_path_for_shard
from shards.registry import REGISTRY_FILENAME, load_registry_index

from .errors import ScheduleError


@dataclass(frozen=True)
class SampleCandidate:
    """One plannable document in an admitted shard."""

    sample_id: str
    shard_id: str
    capability_lane: str
    language: str
    curriculum_band: str
    indic_tier: str | None
    reasoning_trace_band: str | None
    always_on_eligible: bool
    opus_eligible: bool
    anneal_eligible: bool


def build_sample_pool(
    manifests_dir: Path,
    documents: list[dict[str, Any]],
) -> tuple[SampleCandidate, ...]:
    """Expand admitted shard manifests into document-level planning candidates."""
    registry_path = manifests_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        raise ScheduleError(f"shard registry not found: {registry_path}")

    registry = load_registry_index(registry_path)
    admitted_ids = registry.get("admitted_shard_ids")
    if not isinstance(admitted_ids, list):
        raise ScheduleError("admitted_shard_ids must be a list")

    documents_by_id = {doc["document_id"]: doc for doc in documents}
    candidates: list[SampleCandidate] = []

    for shard_id in sorted(admitted_ids):
        manifest = load_shard_manifest(manifest_path_for_shard(manifests_dir, shard_id))
        for document_id in manifest["document_ids"]:
            document = documents_by_id.get(document_id)
            if document is None:
                raise ScheduleError(
                    f"manifest {shard_id} references unknown document_id {document_id}"
                )
            if document.get("never_train"):
                continue
            candidates.append(_candidate_from_document(document, shard_id=shard_id))

    if not candidates:
        raise ScheduleError("sample pool is empty after loading admitted shards")

    return tuple(sorted(candidates, key=lambda candidate: candidate.sample_id))


def _candidate_from_document(document: dict[str, Any], *, shard_id: str) -> SampleCandidate:
    return SampleCandidate(
        sample_id=document["document_id"],
        shard_id=shard_id,
        capability_lane=document["capability_lane"],
        language=document["language"],
        curriculum_band=document["curriculum_band"],
        indic_tier=document["indic_tier"],
        reasoning_trace_band=document["reasoning_trace_band"],
        always_on_eligible=bool(document["always_on_eligible"]),
        opus_eligible=bool(document["opus_eligible"]),
        anneal_eligible=bool(document["anneal_eligible"]),
    )
