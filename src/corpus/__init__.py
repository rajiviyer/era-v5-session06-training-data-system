"""Toy corpus metadata schema, validation, and loading."""

from .loader import (
    CorpusError,
    load_corpus,
    load_documents,
    load_provenance,
)
from .schema import (
    CAPABILITY_LANES,
    CONTENT_STATUSES,
    DATA_TYPES,
    DOCUMENT_METADATA_KEYS,
    EMPTY_TEXT_SHA256,
    INDIC_LANGUAGE_TIERS,
    PACKING_POLICIES,
    T1_INDIC_LANGUAGES,
    T2_INDIC_LANGUAGES,
    validate_document_record,
)

__all__ = [
    "CAPABILITY_LANES",
    "CONTENT_STATUSES",
    "CorpusError",
    "DATA_TYPES",
    "DOCUMENT_METADATA_KEYS",
    "EMPTY_TEXT_SHA256",
    "INDIC_LANGUAGE_TIERS",
    "PACKING_POLICIES",
    "T1_INDIC_LANGUAGES",
    "T2_INDIC_LANGUAGES",
    "load_corpus",
    "load_documents",
    "load_provenance",
    "validate_document_record",
]
