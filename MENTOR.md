# Session 6 — Mentor and Architecture Guide

**Purpose:** Define how this assignment should be built and learned. Students and AI coding assistants should follow this document when direction is unclear.

**Read order for new contributors:**

1. [ASSIGNMENT.md](ASSIGNMENT.md) — what graders require  
2. [SCOPE.md](SCOPE.md) — architecture and invariants  
3. **This doc** — scale, realism, and how to work  
4. [TASKS.md](TASKS.md) — phased task tracker  
5. [REVIEW.md](REVIEW.md) — complexity guardrails and pre-commit checks  

---

## 1. Learning goal

Session 6 is not “make tests pass.” It is **learn how frontier training data systems behave at medium scale**:

- Frozen subword tokenizers with merge tables and hashes  
- Immutable shards and manifests with admission gates  
- Mixture schedules, firewalls, OPUS, ledgers, checkpoints  
- Crash resume, replay, and fork with verifiable evidence  

You should finish able to explain **why** each artifact exists, not only **where** the code lives.

---

## 2. Scale philosophy: medium scale, real contracts

| Dimension | Target | Avoid |
|-----------|--------|-------|
| **Corpus** | 50–200 documents, natural text, 3+ lanes | Empty metadata shells forever; vocab-whitelisted sentences |
| **Tokenizer** | Session 2 BPE (~10k, Metaspace), frozen + hashed | WordLevel toy vocab (~63 words) as the final design |
| **Model** | Tiny causal LM (2–4 layers), CPU-friendly | Frontier-scale model or fake loss |
| **Training run** | ~50 steps with crash/resume demo | Manual steps or hardcoded evidence |
| **Infrastructure** | Single process, generated artifacts | Distributed stack, databases, message queues |

**Rule:** Reduce **data volume and compute**, not **system realism**. Manifests, merges, subword tokenization, and ledger binding should look like production patterns in miniature.

---

## 3. Locked architectural decisions

These are decided. Do not re-debate unless the student explicitly asks to change direction.

| ID | Decision | Choice |
|----|----------|--------|
| D1 | Corpus source | Committed JSONL under `data/toy_corpus/` |
| **D2** | **Shard storage format** | **Binary `.bin` for pretrain lanes; JSONL for agentic** |
| D3 | Curriculum stages | 3 stages (foundation, skill_build, anneal) |
| **D7** | **Frozen tokenizer** | **Session 2 BPE** (`session02/artifacts/tokenizer.json`), copied to `data/tokenizer/bpe_tokenizer.json` |
| **D6** | **Exceed / visualization** | **CLI-only** (`dry_run_dataset.py`, `data_card`, `verify_artifacts.py`); no webapp |
| D7 note | WordLevel scaffold | Early P1 WordLevel code was plumbing only; **replace before P1-T04/P1-T05** |

| **D6** | **Visualization / exceed** | **CLI-only reports** (dry-run, data card, verify); no webapp (GitHub submission) |

Open decisions (agents **must ask** before implementing): **D4, D5** — see [TASKS.md](TASKS.md).

---

## 4. For students

### Your job

- Implement tasks in [TASKS.md](TASKS.md) order unless the mentor doc says otherwise.  
- Run tests after each phase.  
- Read generated manifests and ledgers, not only code.  
- Ask **why** when a shortcut feels too simple (WordLevel, hardcoded hashes, fake OPUS).

### The mentor’s job (human instructor or AI assistant)

- **Propose the default path** when architecture is ambiguous.  
- **Challenge overly basic choices** that break realism (WordLevel for final shards, empty corpus text, simulated recovery).  
- **Ask clarifying questions** only when a decision is genuinely open (D2, D4, etc.) or you need a preference between valid options.  
- **Explain tradeoffs** in plain language before large changes.  
- **Not defer** “what tokenizer should we use?” back to you if D7 already answers it.

### When you should push back on the assistant

- It implements without tying work to a TASKS.md item.  
- It hardcodes evidence or skips ledger/checkpoint binding.  
- It keeps WordLevel after P1-T03R is pending.  
- It adds frameworks or abstractions [REVIEW.md](REVIEW.md) warns against.

---

## 5. For AI coding assistants

When working in `session06/assignment/`, act as a **technical architect and mentor**, not a passive order-taker.

### Default behavior

1. **Lead with recommendation** — state what you would do and why, then implement (unless the user asked for advice only).  
2. **Follow locked decisions** in §3 and [TASKS.md](TASKS.md) open-decision table.  
3. **Ask questions** when:  
   - A row in the open-decision table is unresolved **and** the task requires it.  
   - Two valid approaches differ in learning value or grading risk.  
   - The user’s message contradicts a locked decision (confirm before overriding).  
4. **Do not ask** when the docs already decide (e.g. BPE over WordLevel, evidence must be generated).  
5. **Flag misalignment** proactively: “WordLevel is too basic for shard building; next step is P1-T03R.”  
6. **Keep scope minimal** per [REVIEW.md](REVIEW.md) — realism ≠ over-engineering.

### Phrasing to prefer

- “Recommended next step is … because …”  
- “Before P1-T05, we should complete P1-T03R so merges appear in `tokenizer_hash`.”

### Phrasing to avoid

- “Which tokenizer do you want?” (when D7 is set)  
- “WordLevel is simpler; we can keep it” (contradicts architecture)  
- “Tests pass, so we’re done” (without checking realism or acceptance criteria)

---

## 6. Session continuity (ERA V5 arc)

| Session | You learned | Session 6 uses it |
|---------|-------------|-------------------|
| S1 | Transformer, masks, next-token loss | Batch builder, loss masks |
| S2 | **BPE, merges, fertility, freeze** | **Frozen BPE tokenizer + hash** |
| S3 | Provenance, lanes, license | Shard manifests, admission |
| S4 | Cleaning, dedup, PII, contamination fields | Manifest lineage fields |
| S5 | Curriculum, OPUS, Always-ON floors | Mixture compiler, audit |

Dropping BPE in Session 6 breaks the narrative. The data execution system must exercise **merge-aware** tokenization.

---

## 7. Immediate architectural backlog

Session 6 is complete (P0–P12 + PX). Tag `session06-complete` on `main` when ready to ship.

---

## 8. Definition of good progress

| Phase | You should be able to explain… |
|-------|--------------------------------|
| P0 | What configs control crash/resume and curriculum stages |
| P1 | Why `tokenizer_hash` includes merges; what admission blocks |
| P2 | Difference between concat-and-chop vs structure-preserving packing |
| P6–P9 | Why checkpoint binds `ledger_offset` + `branch_id` |
| P11 | How `evidence.json` is produced from real artifacts |

---

*Last updated: 2026-08-07 · P11 complete (structured run.log, evidence bundle); backlog refreshed. Update when architectural decisions or mentor protocol change.*
