# Session 6: Scope and Understanding

**Purpose:** Base document for all Session 6 assignment work. Defines what we are building, why it matters, how course concepts map to deliverables, and what "done" means.

**Sources:**
- [ASSIGNMENT.md](ASSIGNMENT.md) (grading contract)
- [Session 6 course material](../course_material/Session_6_Building_the_Training_Dataset.md)
- Session 5 mixture plan: [session05/assignment/README.md](../../session05/assignment/README.md)
- Session 4 cleaning pipeline: [session04/assignment/README.md](../../session04/assignment/README.md)

**Status:** Implementation in progress (medium scale, realistic contracts)

---

## 1. North Star

Session 5 produced a **mixture and curriculum recipe**. Session 6 must turn that recipe into an **executable, auditable training data stream**.

The assignment is not about training a frontier model at scale. It is about proving that the **Training Data Execution System** is:

| Property | Meaning |
|----------|---------|
| **Correct** | Loss masks, attention masks, position IDs, and packing policies match data type and training mode |
| **Reproducible** | Same inputs, tokenizer, and ledger state produce the same batches |
| **Auditable** | Every consumed batch, OPUS decision, and learning signal is recorded with hashes and lineage |
| **Recoverable** | Crash resume, historical replay, and intentional fork are well-defined and verifiable |
| **Efficient** | Packing utilization and useful loss-bearing tokens/sec are measured, not assumed |

**Realism constraint:** Use **medium scale** (small corpus, small model, short run) but **production-shaped contracts**: subword BPE with merges, manifest discipline, append-only ledgers, generated evidence. Do not simplify the tokenizer or data contracts in ways that would not appear in a real training stack.

> The assignment is complete only when the system can prove what it consumed, why it consumed it, what the model learned from it, and how the run can be reconstructed.

---

## 2. What We Are Building

A **medium-scale but complete** Training Data Execution System for ERA V5.

```text
Session 4 cleaned corpus
        ↓
  Tokenize + shard (immutable, manifest-backed)
        ↓
  Mixture timeline compiler (Session 5 recipe → per-step quotas)
        ↓
  Eval firewall (block never-train / test shards)
        ↓
  OPUS selector (accept / reject / defer + protected-floor override)
        ↓
  Batch builder (pack, mask, position IDs)
        ↓
  Training loop (tiny model, real forward + loss)
        ↓
  Consumption ledger + learning ledger + checkpoints
        ↓
  Crash / replay / fork demonstration + evidence bundle
```

**Scale constraint:** Use a **committed medium corpus** (50–200 documents), **Session 2 BPE tokenizer**, and a **small model**. Reduce volume and compute, not contract realism. The goal is end-to-end correctness and evidence, not 10T throughput.

**Non-negotiable demonstration:** The final run must:

1. **Deliberately crash**, resume from checkpoint, and prove the **next batch is exactly** the expected batch (no skip, no repeat).
2. **Replay an earlier interval** and prove reconstructed batch IDs, token spans, and hashes match the original run.
3. **Fork from an earlier checkpoint** with an explicit new branch ID.

---

## 3. Upstream Contracts (Sessions 1–5)

The dataloader is downstream of every prior design choice. Session 6 must honor these obligations:

| Session | Contract | Session 6 obligation |
|---------|----------|------------------------|
| **S1** Transformer | Next-token loss on intended tokens only; bounded context; attention within window | Correct `loss_mask`, `attention_mask`, `position_ids`; EOS boundaries preserved |
| **S2** Tokenizer | Same raw text → same token IDs; frozen vocab, **merges**, and special tokens | `tokenizer_hash` on every shard; BPE/subword tokenization; shards invalid if tokenizer changes |
| **S3** Sourcing | Provenance, license, capability tags, held-out status travel with data | Shard manifests carry source IDs, license tier, capability lane, language/script |
| **S4** Cleaning | Only cleaned, deduped, PII-screened, contamination-scanned data enters training | `cleaning_pipeline_hash`, dedup status, PII status, contamination status on manifests |
| **S5** Mixture | Curriculum stages, protected floors, OPUS, anneal reserve, lane weights | Executable schedule; Always-ON floor; OPUS audit trail; anneal holdback enforced |

Session 4 output (`corpus.jsonl` + `manifest.jsonl`) is the natural upstream input for tokenized shards in this assignment.

---

## 4. Core Vocabulary

Shared terms from the course (used consistently in code, logs, and evidence):

| Term | Definition |
|------|------------|
| **Token** | Integer ID from the frozen tokenizer |
| **Sequence** | Fixed-length window of tokens (e.g. 256–4096 for toy run) |
| **Sample** | One training example (pretrain: one window; SFT/agentic: multi-field with masks) |
| **Microbatch** | Small batch processed by one worker/GPU before gradient accumulation |
| **Global batch** | All samples contributing to one optimizer update |
| **Training step** | One optimizer update |
| **Checkpoint step** | Step where model + optimizer + scheduler + RNG + **dataloader state + ledger offset** are saved |
| **Shard** | Immutable tokenized storage unit with manifest |
| **Ledger offset** | Append-only index into the consumption ledger; binds checkpoint to data position |
| **Branch ID** | Identifies a data stream lineage after resume, replay, or fork |

**Key distinction:** The GPU computes over every position, but the model **learns only** from positions where `loss_mask = 1`.

---

## 5. Subsystems and Requirements Matrix

Each subsystem must produce **reproducible evidence**. Hardcoded or simulated results fail inspection.

| # | Subsystem | Assignment requirement | Primary artifacts |
|---|-----------|------------------------|-------------------|
| A | **Shard registry** | Immutable tokenized shards with manifests | `shards/*.bin`, `manifests/shard_*.json` |
| B | **Tokenizer freeze** | Frozen tokenizer and content hashes | `tokenizer/` dir, `tokenizer_hash` in manifests |
| C | **Packing engine** | Packing policies for different data types | Packing config, utilization report |
| D | **Batch builder** | Correct loss, attention, position IDs | Per-batch `loss_mask_hash`, batch records in ledger |
| E | **Mixture compiler** | Curriculum stages, lane weights, protected floors | `schedule.json`, stage transitions in log |
| F | **Eval firewall** | Evaluation/validation never enter loss-bearing batches | Eval registry, firewall rejection events |
| G | **OPUS selector** | Accept, reject, defer, protected-floor override | `opus_audit.jsonl` |
| H | **Consumption ledger** | Append-only record of what was actually consumed | `ledgers/consumption.jsonl` |
| I | **Learning ledger** | Token/sample loss linked back to shards | `ledgers/learning.jsonl`, perplexity aggregates |
| J | **Checkpoint binding** | Checkpoints tied to ledger offsets | `checkpoints/ckpt-*/` + `ledger_offset` |
| K | **Recovery modes** | Resume, replay, fork without batch drift | `run.log` crash/resume/replay events |
| L | **Throughput metrics** | Packing utilization, useful tokens/sec | `reports/throughput.json` |

---

## 6. Subsystem Detail

### 6.1 Immutable tokenized shards and manifests

**Intuition:** Tokenization is expensive and must be frozen. A shard is a sealed object; any change creates a new shard with new hashes.

**Manifest fields (minimum):**

```json
{
  "shard_id": "...",
  "source_ids": ["..."],
  "document_ids": ["..."],
  "tokenizer_hash": "...",
  "content_hash": "sha256:...",
  "token_count": 0,
  "capability_lane": "web | code | indic | stem | reasoning | long_context | agentic",
  "language": "...",
  "script": "...",
  "license_tier": "...",
  "cleaning_pipeline_hash": "...",
  "dedup_status": "passed",
  "pii_screen_status": "screened",
  "eval_overlap_status": "clear",
  "parent_manifest_ids": [],
  "admission": "admitted | blocked"
}
```

**Admission gate:** Block shards missing tokenizer hash, cleaning lineage, eval overlap clearance, or safe license tier.

**Storage (D2, locked):** Indexed binary arrays for pretrain lanes (`S6BIN` header + uint32 token IDs). JSONL records for agentic (`document_id`, `token_ids` per line). Manifest discipline is required regardless of format.

### 6.2 Packing policies

Packing fills fixed-length sequences with useful tokens. Policy choice depends on data type:

| Policy | Best for | Risk |
|--------|----------|------|
| Pad-only | Structured SFT/agentic | Low utilization |
| Concat-and-chop | Plain web pretrain | Mechanical cut points |
| Greedy pack | Mixed lengths | Order-dependent |
| Best-fit pack | Many short docs | Better utilization |
| Structure-preserving | SFT, agentic, reasoning | Higher mask complexity |
| Long-context pack | 32K+ windows | Expensive waste if under-filled |

**Assignment must demonstrate** at least two data types with appropriate policies (e.g. web pretrain + agentic structure-preserving).

**Metrics:** `packing_utilization = useful_tokens / (seq_len × num_sequences)`.

### 6.3 Batch builder (masks and positions)

Every batch carries training meaning, not just token IDs:

| Field | Role |
|-------|------|
| `input_ids` | Token sequence |
| `loss_mask` | 1 = gradient-bearing, 0 = context-only or pad |
| `attention_mask` | Causal / block / document-boundary policy |
| `position_ids` | RoPE or absolute position policy (record policy in ledger) |
| `document_ids` / `span_ids` | Provenance within packed window |
| `loss_mask_hash` | Fingerprint of which positions train |

**Modes to support (at least two in demo):**

- **Pretrain:** next-token loss on non-pad, non-EOS-context tokens
- **SFT or agentic:** loss on assistant/tool-call tokens only; user and tool output masked

### 6.4 Mixture timeline compiler

Converts Session 5 recipe into **per-step quotas**:

- Current curriculum stage and token budget for that stage
- Lane weights (web, code, indic, stem, reasoning, long_context, agentic)
- Protected floors (Always-ON fraction, e.g. 11% bypassing OPUS)
- Anneal reserve (shards tagged `anneal_eligible` excluded until anneal stage)
- Warmup bands at stage transitions (soft blend, not hard switch)
- Supply checks: flag lanes that need repeat, synthetic fill, or schedule change

**Toy schedule:** 2–3 stages over a small token budget is enough if transitions and floor enforcement are visible in logs.

### 6.5 OPUS selector and audit trail

OPUS scores candidate batches and decides accept / reject / defer. Protected floors can **override** rejection for indic, agentic, or reasoning lanes.

**Per-candidate audit record:**

```json
{
  "candidate_id": "...",
  "shard_ids": ["..."],
  "capability_lane": "...",
  "curriculum_stage": "...",
  "opus_score": 0.0,
  "decision": "accepted | rejected | deferred | protected_override",
  "rejection_reason": "...",
  "protected_floor_override": false,
  "effective_token_estimate": 0
}
```

Rejected clean data must remain visible (not silently dropped). Deferred batches may enter a later stage.

**Toy OPUS:** A deterministic scoring function (e.g. hash-based or simple feature score) is fine if decisions are logged and reproducible. Do not hardcode accept/reject lists.

### 6.6 Eval firewall

Test and validation shards live in the same registry discipline as training shards, but with `never_train=true`.

**Checks before a batch enters training:**

- Direct test shard overlap
- MinHash / exact benchmark overlap
- Canary string matches
- Benchmark-derived explanation overlap

**Firewall event:** Block batch, write rejection to log and OPUS/ledger, never assign loss-bearing tokens from eval data.

**Demo must include:** At least one candidate blocked by firewall; evidence that no eval token appears in any `loss_mask=1` position.

### 6.7 Consumption ledger

Append-only record of **what actually happened** (not just what was planned).

**Per consumed microbatch / global step:**

```json
{
  "run_id": "...",
  "branch_id": "...",
  "global_step": 0,
  "ledger_offset": 0,
  "checkpoint_id": "...",
  "microbatch_id": "...",
  "packed_sample_ids": ["..."],
  "shard_ids": ["..."],
  "token_span_ids": ["..."],
  "loss_mask_hash": "...",
  "attention_policy": "...",
  "position_policy": "...",
  "mixture_lane": "...",
  "curriculum_stage": "...",
  "tokenizer_hash": "...",
  "dataloader_version": "...",
  "opus_decision_id": "...",
  "batch_content_hash": "..."
}
```

The ledger is the run's memory: reconstruct any step, resume from offset, audit suspicious behavior.

### 6.8 Learning ledger

Links consumption back to **learning outcomes**:

- Per-shard or per-sample average loss / perplexity
- High-perplexity token clusters
- Loss delta before/after exposure window
- Model phase (early, mid, late, anneal)
- OPUS score vs actual loss improvement
- Usefulness classification: useful | neutral | harmful | review

**Levels for toy run:** Sample-level or aggregated shard-level traces are sufficient. Full token-level traces optional but valuable for demo.

### 6.9 Checkpoints and recovery modes

A checkpoint without a data position is **incomplete**. Each checkpoint saves:

- Model weights (tiny model)
- Optimizer and scheduler state
- RNG state
- **Ledger offset**
- **Branch ID**
- Global step

| Mode | Behavior |
|------|----------|
| **Resume** | Continue same run from latest checkpoint + ledger offset (crash recovery) |
| **Replay** | Restore older checkpoint; feed **identical** historical stream from that offset |
| **Fork** | Restore checkpoint; new `branch_id`; ledger records divergence point |
| **Audit** | Reconstruct shards/batches for a step or token range |

**Critical invariant:** After resume, `batch[N]` must match pre-crash `batch[N]` in IDs, spans, and hashes.

---

## 7. End-to-End Data Flow

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Session 4       │     │ Tokenize +       │     │ Shard registry  │
│ corpus.jsonl    │────▶│ manifest build   │────▶│ (immutable)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                            │
┌─────────────────┐     ┌──────────────────┐                  ▼
│ Session 5       │     │ Mixture timeline │     ┌─────────────────┐
│ curriculum YAML │────▶│ compiler         │────▶│ Sample planner  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                            │
                     ┌──────────────────┐                   ▼
                     │ Eval firewall    │◀────── Candidate batch
                     └────────┬─────────┘
                              │ pass
                              ▼
                     ┌──────────────────┐
                     │ OPUS selector    │
                     └────────┬─────────┘
                              │ accepted
                              ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │ Packing + masks  │────▶│ Training step   │
                     └──────────────────┘     │ (forward+loss)  │
                                              └────────┬────────┘
                                                       │
              ┌────────────────────────────────────────┼────────────────────┐
              ▼                    ▼                   ▼                    ▼
     Consumption ledger    Learning ledger    Checkpoint save      Throughput report
```

---

## 8. Demonstration Script (Target Behavior)

One command runs the full demo without manual steps, e.g.:

```powershell
python session06/assignment/scripts/run_demo.py
```

```bash
python session06/assignment/scripts/run_demo.py
```

**Suggested demo phases (automated):**

| Phase | Action | Evidence produced |
|-------|--------|-------------------|
| 1 | Build shards from toy corpus + manifests | `manifests/`, `shards/` |
| 2 | Compile mixture schedule | `schedule.json` |
| 3 | Train N steps; log every batch | `ledgers/consumption.jsonl`, `run.log` |
| 4 | Save checkpoint at step K | `checkpoints/ckpt-K/` |
| 5 | Continue to step M | Learning ledger entries |
| 6 | **Simulate crash** at step M | `run.log`: `crash_at_step=M` |
| 7 | **Resume** from ckpt-K | Next batch hash matches pre-crash expectation |
| 8 | **Replay** steps K..M | Batch IDs, spans, hashes match original |
| 9 | **Fork** from ckpt-K with new branch | New branch_id, divergence logged |
| 10 | Firewall + OPUS edge cases | Blocked eval batch, protected override |
| 11 | Emit evidence bundle | `evidence.json`, `evidence.md` |

Exact command and paths may differ; behavior may not.

---

## 9. Submission Artifacts Structure

Regenerated by the demo command into `submission_artifacts/` (or equivalent):

```text
submission_artifacts/
  run.log                          # Complete event sequence
  evidence.json                    # Machine-readable pass/fail per requirement
  evidence.md                      # Human-readable summary
  manifests/
    shard_*.json
    tokenizer_manifest.json
  shards/
    *.bin or *.jsonl
  schedule.json
  eval_registry.json
  ledgers/
    consumption.jsonl
    learning.jsonl
    opus_audit.jsonl
  checkpoints/
    ckpt-*/
  reports/
    packing_utilization.json
    throughput.json
    replay_verification.json
    resume_verification.json
```

### 9.1 run.log events (minimum)

Log clearly:

- Run start (run_id, branch_id, config hashes)
- Shard admission / rejection
- Stage transitions
- OPUS decisions (accept, reject, defer, protected override)
- Firewall blocks
- Batch committed (step, ledger_offset, batch_content_hash)
- Checkpoint saved (step, ledger_offset)
- Crash simulated
- Resume / replay / fork initiated
- Verification results (pass/fail with expected vs actual hashes)
- Run complete

### 9.2 evidence.json schema (conceptual)

```json
{
  "requirements": {
    "immutable_shards_with_manifests": { "passed": true, "evidence_path": "..." },
    "frozen_tokenizer_hashes": { "passed": true, "evidence_path": "..." },
    "packing_policies": { "passed": true, "evidence_path": "..." },
    "correct_masks": { "passed": true, "evidence_path": "..." },
    "curriculum_and_floors": { "passed": true, "evidence_path": "..." },
    "eval_firewall": { "passed": true, "evidence_path": "..." },
    "opus_audit_trail": { "passed": true, "evidence_path": "..." },
    "consumption_ledger": { "passed": true, "evidence_path": "..." },
    "learning_ledger": { "passed": true, "evidence_path": "..." },
    "checkpoint_ledger_binding": { "passed": true, "evidence_path": "..." },
    "crash_resume_no_skip_repeat": { "passed": true, "evidence_path": "..." },
    "replay_hash_match": { "passed": true, "evidence_path": "..." },
    "fork_new_branch": { "passed": true, "evidence_path": "..." },
    "packing_and_throughput": { "passed": true, "evidence_path": "..." }
  },
  "generated_at": "...",
  "demo_command": "...",
  "git_commit": "..."
}
```

Evidence must be **generated by the implementation** from actual run outputs. Static/hardcoded evidence fails Step 3 inspection.

---

## 10. Evaluation Model (1,000 points)

Three-step grading:

| Step | What happens | Failure examples |
|------|--------------|------------------|
| **1. Execute** | Run submitted command; regenerate artifacts | Command fails, missing outputs |
| **2. Verify evidence** | Cross-check logs, evidence, manifests, ledgers | Resume repeats/skips batch; replay hash mismatch; eval in loss batch |
| **3. Inspect code** | Confirm evidence is produced by real logic | Hardcoded hashes, simulated OPUS, fake ledger |

**Principle:** A subsystem gets credit only when reproducible evidence supports it.

---

## 11. Core Invariants (Test These)

These should become automated tests:

1. **Tokenizer immutability:** Changing tokenizer hash invalidates shard without re-tokenization.
2. **Loss mask correctness:** Sum of `loss_mask` matches reported useful tokens; eval tokens always masked.
3. **Loss mask hash stability:** Same batch composition → same `loss_mask_hash`.
4. **Ledger append-only:** Offsets monotonic; no in-place edits.
5. **Checkpoint completeness:** Every checkpoint records `ledger_offset` and `branch_id`.
6. **Resume exactness:** Post-resume batch sequence equals pre-crash sequence from same offset.
7. **Replay exactness:** Reconstructed batches match original `batch_content_hash` for replayed range.
8. **Fork divergence:** New branch_id; ledger records parent offset and fork point.
9. **Firewall:** No `never_train` shard ID in any consumption ledger entry with loss-bearing tokens.
10. **OPUS audit completeness:** Every accepted batch has a matching audit record.
11. **Protected floor:** Always-ON fraction met per step (within tolerance for toy scale).
12. **Packing utilization:** Reported utilization matches recomputation from batch contents.

---

## 12. Suggested Implementation Phases

Use this order to de-risk the assignment:

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| **P0** | Project layout, config schema, toy corpus | — |
| **P1** | Tokenizer freeze + shard builder + manifests | P0, Session 4 corpus optional |
| **P2** | Packing engine + batch builder (masks, positions) | P1 |
| **P3** | Mixture compiler + sample planner | P1, Session 5 schedule subset |
| **P4** | Eval registry + firewall | P1 |
| **P5** | OPUS selector + audit log | P3, P4 |
| **P6** | Consumption ledger + checkpoint binding | P2, P5 |
| **P7** | Tiny training loop (real forward/backward) | P6 |
| **P8** | Learning ledger + loss/perplexity tracking | P7 |
| **P9** | Crash / resume / replay / fork | P6, P7 |
| **P10** | Throughput + packing reports | P2, P7 |
| **P11** | Demo orchestrator + evidence generator | P9, P10 |
| **P12** | Tests for invariants + README | All |

---

## 13. Proposed Repository Layout

```text
session06/assignment/
  SCOPE.md                    ← this document
  ASSIGNMENT.md               ← official brief
  README.md                   ← final submission README (architecture + run instructions)
  configs/
    demo.yaml                 ← toy run config (seq len, stages, floors, steps)
    curriculum.yaml           ← subset of Session 5 schedule
  src/
    tokenizer/                ← freeze + hash
    shards/                   ← build, manifest, registry
    packing/                  ← policies per data type
    batch/                    ← masks, position ids, hashing
    schedule/                 ← mixture timeline compiler
    firewall/                 ← eval registry + checks
    opus/                     ← selector + audit
    ledger/                   ← consumption + learning
    checkpoint/               ← save/load with ledger offset
    trainer/                  ← tiny model training loop
    recovery/                 ← resume, replay, fork
    evidence/                 ← evidence.json/md generator
    metrics/                  ← throughput, utilization
  scripts/
    run_demo.py               ← one-command full demonstration
    build_shards.py           ← optional standalone shard build
  tests/
    test_*.py                 ← invariant tests
  data/
    toy_corpus/               ← small committed corpus
  submission_artifacts/       ← gitignored; generated by demo
```

---

## 14. Design Decisions (Medium Scale, Realistic Contracts)

Decisions to lock early so implementation stays focused.

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Corpus size | 50–200 documents across 3+ lanes, **natural text** | Enough for multi-stage, OPUS, firewall demos; not vocab-whitelisted |
| Sequence length | 256–512 tokens | Fast CPU runs; packing still visible |
| Model | Small causal LM (e.g. 2–4 layer, tiny dim) | Real loss; CPU-friendly |
| OPUS scoring | Deterministic function of shard metadata + step | Reproducible without training a proxy model |
| Distributed training | Single-process fake "ranks" optional | Ledger fields for rank/microbatch still populated |
| **Tokenizer** | **Session 2 BPE** (~10k vocab, Metaspace), frozen under `data/tokenizer/` | Subword merges, multilingual Indic support; matches S2→S6 arc. WordLevel was early scaffold only. |
| Checkpoint interval | Every 10–20 steps | Enough points for crash/resume demo |

---

## 15. Open Questions (Resolve During Implementation)

Track decisions here as work proceeds:

| # | Question | Options | Status |
|---|----------|---------|--------|
| Q1 | Reuse Session 4 cleaned output or ship minimal toy corpus? | S4 corpus vs inline JSONL | **Resolved: committed toy JSONL (D1)** |
| Q2 | Binary shards vs JSONL token records? | Binary for pretrain; JSONL for agentic | **Resolved (D2)** |
| Q3 | How many curriculum stages in demo? | 2 vs 3 | **Resolved: 3 stages (D3)** |
| Q4 | Real PyTorch training vs minimal autograd loop? | PyTorch preferred per project stack | Open (see D4) |
| Q5 | Token-level vs sample-level learning ledger? | Sample-level minimum; token-level stretch | Open (see D5) |
| Q6 | Include webapp visualization? | CLI-only exceed reports (dry-run, data card, verify) | **Resolved (D6): no webapp** |
| Q7 | WordLevel vs BPE tokenizer? | WordLevel scaffold vs Session 2 BPE | **Resolved: Session 2 BPE (D7)** |

---

## 16. References and Prior Art

From ASSIGNMENT.md and course material:

| Reference | Relevance |
|-----------|-----------|
| [Megatron GPT dataset](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py) | Document, sample, shuffle indices; planned sample lookup |
| [Mosaic StreamingDataset](https://github.com/mosaicml/streaming) | Mid-epoch resume; deterministic ordering |
| [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | Curation and dedup at scale (context) |
| [WebDataset](https://github.com/webdataset/webdataset) | Multi-file samples in tar shards |
| LakeFS / Iceberg / Delta | Versioned data and audit log analogies |

Session 5 implementation sketch: [session05/assignment/notes-implementation-sketch.md](../../session05/assignment/notes-implementation-sketch.md)

---

## 17. Success Criteria Checklist

Before submission, all must be true:

- [x] One command runs full demo end-to-end
- [x] `submission_artifacts/` regenerates cleanly from scratch
- [x] `evidence.json` shows all requirements passed with valid evidence paths
- [x] Crash → resume produces **identical** next batch (verified in `resume_verification.json`)
- [x] Replay of interval matches original batch hashes
- [x] Fork creates new branch with logged divergence
- [x] Eval/test shard blocked; never appears in loss-bearing batch
- [x] OPUS accept/reject/defer/override all appear in audit log
- [x] Consumption ledger reconstructs any logged step
- [x] Learning ledger links at least one shard to loss metrics
- [x] Packing utilization and useful tokens/sec reported and reproducible
- [x] Automated tests pass for core invariants
- [x] README explains architecture and design decisions
- [x] No hardcoded evidence or simulated behavior

---

## 18. Relationship to Future Sessions

Session 6 output feeds forward:

| Downstream | Uses Session 6 |
|------------|----------------|
| Distributed training (Session 7+) | Ledger offsets, rank-aware consumption, checkpoint binding |
| Evaluation suite | Firewall guarantees eval integrity |
| V6 corpus planning | Learning ledger usefulness signals, OPUS rejection analysis |
| Inference / release | Reproducible training provenance |

The two-way learning ledger closes the loop: **V5 training teaches V6 what to collect, protect, repeat, defer, or reject.**
