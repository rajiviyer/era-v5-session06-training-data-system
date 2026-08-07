"""Shard admission gate (P1-T07)."""

from __future__ import annotations

from typing import Any

from .errors import ShardError

BLOCK_REASONS = (
    "missing_tokenizer_hash",
    "missing_cleaning_lineage",
    "unsafe_license_tier",
    "eval_overlap_detected",
    "failed_dedup",
    "failed_pii_screen",
)


def evaluate_admission(manifest: dict[str, Any]) -> tuple[str, list[str]]:
    """Return admission status and blocking reasons for a shard manifest."""
    reasons: list[str] = []

    tokenizer_hash = manifest.get("tokenizer_hash", "")
    if not isinstance(tokenizer_hash, str) or not tokenizer_hash.startswith("tok_"):
        reasons.append("missing_tokenizer_hash")

    cleaning_hash = manifest.get("cleaning_pipeline_hash", "")
    if not isinstance(cleaning_hash, str) or not cleaning_hash.strip() or cleaning_hash == "multi":
        reasons.append("missing_cleaning_lineage")

    license_tier = manifest.get("license_tier")
    if license_tier != "safe":
        reasons.append("unsafe_license_tier")

    if manifest.get("eval_overlap_status") != "clear":
        reasons.append("eval_overlap_detected")

    if manifest.get("dedup_status") != "passed":
        reasons.append("failed_dedup")

    if manifest.get("pii_screen_status") != "screened":
        reasons.append("failed_pii_screen")

    for reason in reasons:
        if reason not in BLOCK_REASONS:
            raise ShardError(f"unknown block reason emitted: {reason}")

    if reasons:
        return "blocked", reasons
    return "admitted", []


def apply_admission(manifest: dict[str, Any]) -> dict[str, Any]:
    """Set manifest admission field from gate evaluation."""
    admission, _reasons = evaluate_admission(manifest)
    updated = dict(manifest)
    updated["admission"] = admission
    return updated
