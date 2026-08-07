"""Evidence bundle: pass/fail per requirement, computed from generated artifacts."""

from .collector import collect_evidence, write_evidence_json
from .report import render_evidence_markdown, write_evidence_markdown
from .types import (
    EVIDENCE_JSON_FILENAME,
    EVIDENCE_MD_FILENAME,
    REQUIREMENT_KEYS,
    EvidenceBundle,
    EvidenceError,
    RequirementResult,
)

__all__ = [
    "EVIDENCE_JSON_FILENAME",
    "EVIDENCE_MD_FILENAME",
    "REQUIREMENT_KEYS",
    "EvidenceBundle",
    "EvidenceError",
    "RequirementResult",
    "collect_evidence",
    "render_evidence_markdown",
    "write_evidence_json",
    "write_evidence_markdown",
]
