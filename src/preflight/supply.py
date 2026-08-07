"""Dataset supply and admission pre-flight reports (PX-T01)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus import load_corpus
from schedule.io import load_schedule_json
from schedule.types import LANE_KEYS, StepSchedule
from shards.admission import evaluate_admission
from shards.io import write_json_atomic
from shards.manifest import load_shard_manifest
from shards.registry import REGISTRY_FILENAME, load_registry_index

SUPPLY_REPORT_FILENAME = "dataset_supply.json"
ADMISSION_AUDIT_FILENAME = "admission_audit.json"


@dataclass(frozen=True)
class PreflightInputs:
    """Artifacts the dry-run reads; no training run required."""

    assignment_root: Path
    artifacts_root: Path
    schedule_path: Path
    manifests_dir: Path
    corpus_dir: Path


def _average_lane_quotas(steps: tuple[StepSchedule, ...]) -> dict[str, float]:
    totals = {lane: 0.0 for lane in LANE_KEYS}
    if not steps:
        return totals
    for step in steps:
        for lane, quota in step.opus_lane_quotas.items():
            totals[lane] = totals.get(lane, 0.0) + quota
    count = len(steps)
    return {lane: total / count for lane, total in totals.items()}


def _lane_supply_from_manifests(manifests: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    supply: dict[str, dict[str, int]] = {
        lane: {"admitted_shards": 0, "admitted_tokens": 0} for lane in LANE_KEYS
    }
    for manifest in manifests:
        if manifest.get("admission") != "admitted":
            continue
        lane = str(manifest["capability_lane"])
        if lane not in supply:
            supply[lane] = {"admitted_shards": 0, "admitted_tokens": 0}
        supply[lane]["admitted_shards"] += 1
        supply[lane]["admitted_tokens"] += int(manifest["token_count"])
    return supply


def build_dataset_supply(inputs: PreflightInputs) -> dict[str, Any]:
    """Compare compiled schedule lane demand to admitted shard supply per stage."""
    schedule = load_schedule_json(inputs.schedule_path)
    manifests = _load_manifests(inputs.manifests_dir)
    lane_supply = _lane_supply_from_manifests(manifests)

    stages: list[dict[str, Any]] = []
    under_supplied_lanes: list[str] = []

    for boundary in schedule.phase_boundaries:
        stage_steps = tuple(
            step for step in schedule.steps if boundary.step_start <= step.step < boundary.step_end
        )
        avg_quotas = _average_lane_quotas(stage_steps)
        lane_rows: list[dict[str, Any]] = []

        for lane in sorted(LANE_KEYS):
            quota = avg_quotas.get(lane, 0.0)
            supply = lane_supply.get(lane, {"admitted_shards": 0, "admitted_tokens": 0})
            admitted_shards = supply["admitted_shards"]
            admitted_tokens = supply["admitted_tokens"]
            under_supplied = quota > 1e-6 and admitted_shards == 0
            if under_supplied:
                under_supplied_lanes.append(f"{boundary.name}:{lane}")

            lane_rows.append(
                {
                    "lane": lane,
                    "average_opus_quota": round(quota, 6),
                    "admitted_shard_count": admitted_shards,
                    "admitted_token_count": admitted_tokens,
                    "under_supplied": under_supplied,
                }
            )

        stages.append(
            {
                "stage": boundary.name,
                "step_start": boundary.step_start,
                "step_end": boundary.step_end,
                "lanes": lane_rows,
            }
        )

    return {
        "schema_version": "1.0",
        "schedule_path": _relative_or_absolute(inputs.schedule_path, inputs.assignment_root),
        "total_steps": schedule.total_steps,
        "stages": stages,
        "under_supplied_lanes": sorted(under_supplied_lanes),
        "has_supply_warnings": bool(under_supplied_lanes),
        "compiler_warnings": list(schedule.warnings),
    }


def build_admission_audit(inputs: PreflightInputs) -> dict[str, Any]:
    """Audit every shard manifest with the admission gate."""
    manifests = _load_manifests(inputs.manifests_dir)
    registry_path = inputs.manifests_dir / REGISTRY_FILENAME
    registry = load_registry_index(registry_path) if registry_path.is_file() else {}

    admitted_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for manifest in sorted(manifests, key=lambda item: item["shard_id"]):
        status, reasons = evaluate_admission(manifest)
        row = {
            "shard_id": manifest["shard_id"],
            "capability_lane": manifest["capability_lane"],
            "token_count": manifest["token_count"],
            "manifest_admission": manifest.get("admission"),
            "evaluated_admission": status,
            "block_reasons": reasons,
        }
        if status == "admitted":
            admitted_rows.append(row)
        else:
            blocked_rows.append(row)

    return {
        "schema_version": "1.0",
        "registry_path": _relative_or_absolute(registry_path, inputs.assignment_root),
        "tokenizer_hash": registry.get("tokenizer_hash"),
        "admitted_count": len(admitted_rows),
        "blocked_count": len(blocked_rows),
        "admitted": admitted_rows,
        "blocked": blocked_rows,
    }


def write_preflight_reports(
    inputs: PreflightInputs,
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write dataset supply and admission audit JSON reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    supply_path = reports_dir / SUPPLY_REPORT_FILENAME
    admission_path = reports_dir / ADMISSION_AUDIT_FILENAME
    write_json_atomic(supply_path, build_dataset_supply(inputs))
    write_json_atomic(admission_path, build_admission_audit(inputs))
    return supply_path, admission_path


def _load_manifests(manifests_dir: Path) -> list[dict[str, Any]]:
    if not manifests_dir.is_dir():
        raise FileNotFoundError(f"manifests directory not found: {manifests_dir}")
    manifests: list[dict[str, Any]] = []
    for path in sorted(manifests_dir.glob("shard_*.json")):
        if path.name == REGISTRY_FILENAME:
            continue
        manifests.append(load_shard_manifest(path))
    if not manifests:
        raise FileNotFoundError(f"no shard manifests found under {manifests_dir}")
    return manifests


def _relative_or_absolute(path: Path, assignment_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(assignment_root.resolve()))
    except ValueError:
        return str(path.resolve())


def corpus_lane_counts(corpus_dir: Path) -> dict[str, int]:
    """Count ready documents by capability lane in the toy corpus."""
    _, documents = load_corpus(corpus_dir)
    counts: dict[str, int] = defaultdict(int)
    for document in documents:
        if document.get("content_status") == "ready":
            counts[str(document["capability_lane"])] += 1
    return dict(sorted(counts.items()))
