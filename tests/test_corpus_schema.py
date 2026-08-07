"""Tests for toy corpus metadata schema and loading."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
_CORPUS = _ASSIGNMENT / "data" / "toy_corpus"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from corpus import (  # noqa: E402
    CAPABILITY_LANES,
    T1_INDIC_LANGUAGES,
    T2_INDIC_LANGUAGES,
    load_corpus,
    validate_document_record,
)
from corpus.schema import CorpusSchemaError, content_sha256  # noqa: E402


class TestToyCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provenance, cls.documents = load_corpus(_CORPUS)

    def test_loads_provenance_and_documents(self) -> None:
        self.assertGreaterEqual(len(self.provenance), 5)
        self.assertGreaterEqual(len(self.documents), 50)
        self.assertLessEqual(len(self.documents), 100)

    def test_all_documents_ready_with_text(self) -> None:
        for doc in self.documents:
            self.assertEqual(doc["content_status"], "ready")
            self.assertTrue(doc["text"].strip())
            self.assertEqual(doc["content_sha256"], content_sha256(doc["text"]))
            self.assertGreater(doc["char_count"], 0)

    def test_covers_all_capability_lanes(self) -> None:
        lanes = {doc["capability_lane"] for doc in self.documents}
        for required in CAPABILITY_LANES:
            self.assertIn(required, lanes)

    def test_includes_never_train_eval_row(self) -> None:
        eval_docs = [doc for doc in self.documents if doc["never_train"]]
        self.assertEqual(len(eval_docs), 1)
        self.assertEqual(eval_docs[0]["document_id"], "doc-eval-001")
        self.assertEqual(eval_docs[0]["eval_overlap_status"], "overlap_detected")

    def test_agentic_uses_structure_preserving_packing(self) -> None:
        agentic = [doc for doc in self.documents if doc["capability_lane"] == "agentic"]
        self.assertGreaterEqual(len(agentic), 2)
        for doc in agentic:
            self.assertEqual(doc["packing_policy"], "structure_preserving")

    def test_covers_t1_and_t2_indic_languages(self) -> None:
        indic = [doc for doc in self.documents if doc["capability_lane"] == "indic"]
        languages = {doc["language"] for doc in indic}
        t1_present = languages & T1_INDIC_LANGUAGES
        t2_present = languages & T2_INDIC_LANGUAGES
        self.assertTrue(T1_INDIC_LANGUAGES.issubset(t1_present))
        self.assertTrue({"gu", "kn", "ml"}.issubset(t2_present))
        for doc in indic:
            self.assertIn(doc["indic_language_tier"], {"T1", "T2"})
            if doc["language"] in T1_INDIC_LANGUAGES:
                self.assertEqual(doc["indic_language_tier"], "T1")
            if doc["language"] in T2_INDIC_LANGUAGES:
                self.assertEqual(doc["indic_language_tier"], "T2")

    def test_indic_document_count(self) -> None:
        indic = [doc for doc in self.documents if doc["capability_lane"] == "indic"]
        self.assertGreaterEqual(len(indic), 12)

    def test_rejects_ready_document_with_empty_text(self) -> None:
        doc = copy.deepcopy(self.documents[0])
        doc["text"] = ""
        doc["char_count"] = 0
        doc["content_sha256"] = content_sha256("")
        with self.assertRaises(CorpusSchemaError):
            validate_document_record(doc)


if __name__ == "__main__":
    unittest.main()
