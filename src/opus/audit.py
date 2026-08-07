"""Append-only OPUS audit log and query helpers (P5-T04–T05)."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import OpusError
from .types import AUDIT_FILENAME, OpusAuditRecord, OpusDecision


def append_opus_audit(path: Path, record: OpusAuditRecord) -> None:
    """Append one OPUS audit record to opus_audit.jsonl."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def load_opus_audit(path: Path) -> tuple[OpusAuditRecord, ...]:
    """Load all OPUS audit records from disk."""
    target = path.resolve()
    if not target.is_file():
        return ()

    records: list[OpusAuditRecord] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpusError(f"invalid JSON in {target} line {line_number}") from exc
        records.append(audit_record_from_dict(payload))
    return tuple(records)


def query_opus_audit(
    path: Path,
    *,
    decision: OpusDecision | None = None,
    candidate_id: str | None = None,
    global_step: int | None = None,
) -> tuple[OpusAuditRecord, ...]:
    """Query rejected, deferred, or other OPUS decisions without deleting rows."""
    records = load_opus_audit(path)
    filtered: list[OpusAuditRecord] = []
    for record in records:
        if decision is not None and record.decision != decision:
            continue
        if candidate_id is not None and record.candidate_id != candidate_id:
            continue
        if global_step is not None and record.global_step != global_step:
            continue
        filtered.append(record)
    return tuple(filtered)


def audit_record_from_dict(payload: dict[str, object]) -> OpusAuditRecord:
    """Parse one audit record from JSON."""
    required = (
        "opus_decision_id",
        "candidate_id",
        "global_step",
        "run_id",
        "branch_id",
        "sample_ids",
        "shard_ids",
        "capability_lane",
        "curriculum_stage",
        "path",
        "decision",
        "protected_floor_override",
        "opus_bypassed",
        "effective_token_estimate",
    )
    for key in required:
        if key not in payload:
            raise OpusError(f"opus audit record missing key: {key}")

    opus_score_raw = payload.get("opus_score")
    opus_score = None if opus_score_raw is None else float(opus_score_raw)
    rejection_reason = payload.get("rejection_reason")
    if rejection_reason is not None:
        rejection_reason = str(rejection_reason)

    return OpusAuditRecord(
        opus_decision_id=str(payload["opus_decision_id"]),
        candidate_id=str(payload["candidate_id"]),
        global_step=int(payload["global_step"]),
        run_id=str(payload["run_id"]),
        branch_id=str(payload["branch_id"]),
        sample_ids=tuple(str(item) for item in payload["sample_ids"]),
        shard_ids=tuple(str(item) for item in payload["shard_ids"]),
        capability_lane=str(payload["capability_lane"]),
        curriculum_stage=str(payload["curriculum_stage"]),
        path=str(payload["path"]),  # type: ignore[arg-type]
        opus_score=opus_score,
        decision=str(payload["decision"]),  # type: ignore[arg-type]
        rejection_reason=rejection_reason,
        protected_floor_override=bool(payload["protected_floor_override"]),
        opus_bypassed=bool(payload["opus_bypassed"]),
        effective_token_estimate=int(payload["effective_token_estimate"]),
    )
