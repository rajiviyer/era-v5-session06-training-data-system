# Toy Corpus (Session 6)

Committed toy corpus for the Session 6 demo. **Document text is filled (P1-T04 complete).**

## Files

| File | Purpose |
|------|---------|
| `corpus_schema.json` | Machine-readable field definitions and enums |
| `provenance.jsonl` | Source registry (license, format, fetch metadata) |
| `documents.jsonl` | 55 documents with natural text and routing metadata |

## Regenerate documents

```powershell
python session06/assignment/scripts/build_corpus_documents.py
```

```bash
python session06/assignment/scripts/build_corpus_documents.py
```

## Design choice (D1)

This assignment uses a **committed toy JSONL corpus** under `data/toy_corpus/`, not Session 4 cleaned output. Session 4 field names are preserved where useful; Session 6 adds mixture-routing tags from Session 5.

## Indic language coverage (Session 5 aligned)

The corpus includes **Tier 1 and Tier 2** Indic languages for routing and tokenizer stress tests:

| Tier | Languages in corpus | Scripts |
|------|---------------------|---------|
| **T1** | Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Marathi (`mr`) | Devanagari, Bengali, Tamil, Telugu |
| **T2** | Gujarati (`gu`), Kannada (`kn`), Malayalam (`ml`) | Gujarati, Kannada, Malayalam |

Quality tiers (`indic_tier`: A/B) are separate from language tiers (`indic_language_tier`: T1/T2/T3).

Text is curated synthetic prose per lane (web, indic native script, code, STEM, reasoning, long context, agentic). Session 5 floor percentages (e.g. T1 ≥10% each at 10T scale) are **not** enforced at medium corpus size.

**Tokenizer:** Final pipeline uses **Session 2 BPE** (decision D7). See [MENTOR.md](../../MENTOR.md) and task P1-T03R in [TASKS.md](../../TASKS.md).

## Document lifecycle

```text
provenance.jsonl  →  documents.jsonl (metadata skeleton, P0)
                    →  documents.jsonl (text filled, P1-T04)
                    →  tokenized shards + manifests (P1-T05+)
```

## Required metadata (summary)

Every row in `documents.jsonl` must include the fields listed in `corpus_schema.json`. Key routing fields:

- `capability_lane`: web, code, indic, stem, reasoning, long_context, agentic
- `indic_language_tier`: T1, T2, T3 (Indic lane only; null otherwise)
- `indic_tier`: A, B, C, D (quality tier for Indic lane)
- `language` / `script`: must match (e.g. `bn` + Bengali)
- `always_on_eligible`, `opus_eligible`, `anneal_eligible`, `never_train`
- `content_status`: `ready` (non-empty text with matching `content_sha256`)

## Validation

```powershell
python -m pytest session06/assignment/tests/test_corpus_schema.py -v
```

```bash
python -m pytest session06/assignment/tests/test_corpus_schema.py -v
```

Loader: `src/corpus/` (`load_provenance`, `load_documents`, `validate_document_record`).
