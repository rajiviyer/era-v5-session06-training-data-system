"""Document metadata schema and validation for the toy corpus."""

from __future__ import annotations

import hashlib
from typing import Any

CAPABILITY_LANES = frozenset(
    {"web", "code", "indic", "stem", "reasoning", "long_context", "agentic"}
)
DATA_TYPES = frozenset({"pretrain", "agentic", "sft"})
PACKING_POLICIES = frozenset(
    {"concat_and_chop", "structure_preserving", "pad_only"}
)
CONTENT_STATUSES = frozenset({"pending_p1", "ready"})
DEDUP_STATUSES = frozenset({"passed", "duplicate"})
PII_STATUSES = frozenset({"screened", "failed"})
EVAL_OVERLAP_STATUSES = frozenset({"clear", "overlap_detected"})
LICENSE_TIERS = frozenset({"safe", "needs_review", "blocked"})
INDIC_TIERS = frozenset({"A", "B", "C", "D"})
INDIC_LANGUAGE_TIERS = frozenset({"T1", "T2", "T3"})
T1_INDIC_LANGUAGES = frozenset({"hi", "bn", "ta", "te", "mr"})
T2_INDIC_LANGUAGES = frozenset({"gu", "kn", "ml", "pa", "or", "as", "ur"})
REASONING_BANDS = frozenset({"short", "medium", "long", "ultra"})
CURRICULUM_BANDS = frozenset({"B0", "B1", "B2", "B3", "B4", "B5"})

LANGUAGE_SCRIPTS: dict[str, str] = {
    "en": "Latin",
    "en-IN": "Latin",
    "hi": "Devanagari",
    "mr": "Devanagari",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
}

EMPTY_TEXT_SHA256 = hashlib.sha256(b"").hexdigest()

DOCUMENT_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "document_id",
        "source_id",
        "title",
        "text",
        "content_status",
        "capability_lane",
        "data_type",
        "packing_policy",
        "language",
        "script",
        "license_tier",
        "cleaning_pipeline_hash",
        "dedup_status",
        "pii_screen_status",
        "eval_overlap_status",
        "never_train",
        "curriculum_band",
        "indic_tier",
        "indic_language_tier",
        "reasoning_trace_band",
        "always_on_eligible",
        "opus_eligible",
        "anneal_eligible",
        "epoch_budget_max",
        "content_sha256",
        "char_count",
        "local_path",
    }
)


class CorpusSchemaError(ValueError):
    """Raised when a corpus record fails metadata validation."""


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_document_record(
    record: dict[str, Any],
    *,
    provenance_ids: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Validate one documents.jsonl row. Returns the record if valid."""
    if set(record.keys()) != DOCUMENT_METADATA_KEYS:
        missing = DOCUMENT_METADATA_KEYS - set(record.keys())
        extra = set(record.keys()) - DOCUMENT_METADATA_KEYS
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if extra:
            details.append(f"unexpected keys: {sorted(extra)}")
        raise CorpusSchemaError("; ".join(details))

    document_id = _non_empty_str(record["document_id"], label="document_id")
    source_id = _non_empty_str(record["source_id"], label="source_id")
    if provenance_ids is not None and source_id not in provenance_ids:
        raise CorpusSchemaError(f"unknown source_id '{source_id}' for {document_id}")

    text = record["text"]
    if not isinstance(text, str):
        raise CorpusSchemaError("text must be a string")

    content_status = record["content_status"]
    if content_status not in CONTENT_STATUSES:
        raise CorpusSchemaError(f"invalid content_status: {content_status}")

    if content_status == "pending_p1" and text != "":
        raise CorpusSchemaError(
            f"{document_id}: pending_p1 documents must have empty text until P1"
        )
    if content_status == "ready" and not text.strip():
        raise CorpusSchemaError(f"{document_id}: ready documents must have non-empty text")

    capability_lane = record["capability_lane"]
    if capability_lane not in CAPABILITY_LANES:
        raise CorpusSchemaError(f"invalid capability_lane: {capability_lane}")

    data_type = record["data_type"]
    if data_type not in DATA_TYPES:
        raise CorpusSchemaError(f"invalid data_type: {data_type}")

    packing_policy = record["packing_policy"]
    if packing_policy not in PACKING_POLICIES:
        raise CorpusSchemaError(f"invalid packing_policy: {packing_policy}")

    if data_type == "agentic" and packing_policy != "structure_preserving":
        raise CorpusSchemaError(
            f"{document_id}: agentic documents must use structure_preserving packing"
        )

    if record["dedup_status"] not in DEDUP_STATUSES:
        raise CorpusSchemaError(f"invalid dedup_status: {record['dedup_status']}")
    if record["pii_screen_status"] not in PII_STATUSES:
        raise CorpusSchemaError(f"invalid pii_screen_status: {record['pii_screen_status']}")
    if record["eval_overlap_status"] not in EVAL_OVERLAP_STATUSES:
        raise CorpusSchemaError(
            f"invalid eval_overlap_status: {record['eval_overlap_status']}"
        )
    if record["license_tier"] not in LICENSE_TIERS:
        raise CorpusSchemaError(f"invalid license_tier: {record['license_tier']}")

    indic_tier = record["indic_tier"]
    if indic_tier is not None and indic_tier not in INDIC_TIERS:
        raise CorpusSchemaError(f"invalid indic_tier: {indic_tier}")

    indic_language_tier = record["indic_language_tier"]
    if indic_language_tier is not None and indic_language_tier not in INDIC_LANGUAGE_TIERS:
        raise CorpusSchemaError(f"invalid indic_language_tier: {indic_language_tier}")

    language = _non_empty_str(record["language"], label="language")
    script = _non_empty_str(record["script"], label="script")
    expected_script = LANGUAGE_SCRIPTS.get(language)
    if expected_script is not None and script != expected_script:
        raise CorpusSchemaError(
            f"{document_id}: language '{language}' expects script '{expected_script}', got '{script}'"
        )

    if capability_lane == "indic":
        if indic_tier is None:
            raise CorpusSchemaError(f"{document_id}: indic lane requires indic_tier")
        if indic_language_tier is None:
            raise CorpusSchemaError(f"{document_id}: indic lane requires indic_language_tier")
        if language in T1_INDIC_LANGUAGES and indic_language_tier != "T1":
            raise CorpusSchemaError(
                f"{document_id}: language '{language}' must use indic_language_tier T1"
            )
        if language in T2_INDIC_LANGUAGES and indic_language_tier != "T2":
            raise CorpusSchemaError(
                f"{document_id}: language '{language}' must use indic_language_tier T2"
            )
    else:
        if indic_tier is not None:
            raise CorpusSchemaError(
                f"{document_id}: non-indic lane must set indic_tier null"
            )
        if indic_language_tier is not None:
            raise CorpusSchemaError(
                f"{document_id}: non-indic lane must set indic_language_tier null"
            )

    reasoning_band = record["reasoning_trace_band"]
    if reasoning_band is not None and reasoning_band not in REASONING_BANDS:
        raise CorpusSchemaError(f"invalid reasoning_trace_band: {reasoning_band}")

    if record["curriculum_band"] not in CURRICULUM_BANDS:
        raise CorpusSchemaError(f"invalid curriculum_band: {record['curriculum_band']}")

    for flag in ("never_train", "always_on_eligible", "opus_eligible", "anneal_eligible"):
        if not isinstance(record[flag], bool):
            raise CorpusSchemaError(f"{flag} must be a boolean")

    epoch_budget = record["epoch_budget_max"]
    if not isinstance(epoch_budget, int) or epoch_budget < 0:
        raise CorpusSchemaError("epoch_budget_max must be a non-negative integer")

    char_count = record["char_count"]
    if not isinstance(char_count, int) or char_count < 0:
        raise CorpusSchemaError("char_count must be a non-negative integer")
    if char_count != len(text):
        raise CorpusSchemaError(
            f"{document_id}: char_count ({char_count}) must match len(text) ({len(text)})"
        )

    expected_hash = content_sha256(text)
    if record["content_sha256"] != expected_hash:
        raise CorpusSchemaError(
            f"{document_id}: content_sha256 does not match text ({expected_hash})"
        )

    if record["never_train"]:
        if record["opus_eligible"]:
            raise CorpusSchemaError(f"{document_id}: never_train rows cannot be opus_eligible")
        if record["always_on_eligible"]:
            raise CorpusSchemaError(
                f"{document_id}: never_train rows cannot be always_on_eligible"
            )
        if record["eval_overlap_status"] != "overlap_detected":
            raise CorpusSchemaError(
                f"{document_id}: never_train rows must have eval_overlap_status=overlap_detected"
            )

    if record["license_tier"] == "blocked" and record["opus_eligible"]:
        raise CorpusSchemaError(f"{document_id}: blocked license cannot be opus_eligible")

    return record


def _non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusSchemaError(f"{label} must be a non-empty string")
    return value.strip()
