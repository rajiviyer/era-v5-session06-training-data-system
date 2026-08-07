# Session 6 Assignment — Task Tracker

**Purpose:** Trackable work breakdown for the Training Data Execution System.  
**Baseline:** [SCOPE.md](SCOPE.md) · **Grading contract:** [ASSIGNMENT.md](ASSIGNMENT.md) · **Mentor guide:** [MENTOR.md](MENTOR.md)

**How to use this doc**

- Read [MENTOR.md](MENTOR.md) first for scale, realism, and how AI assistants should guide you.
- Update **Status** as work progresses: `not_started` → `in_progress` → `done` (or `blocked`).
- Do not mark a task `done` until its **Acceptance criteria** are met and linked tests pass (where applicable).
- Respect **Depends on** order; recovery tasks (P9) should start as soon as P6 ledger design is stable.
- **Locked decisions (D1–D3, D6, D7):** do not re-open without explicit intent. **Open decisions (D4–D5):** assistants should ask before implementing.

**Status legend**

| Status | Meaning |
|--------|---------|
| `not_started` | Not begun |
| `in_progress` | Active work |
| `blocked` | Waiting on dependency or decision |
| `done` | Acceptance criteria verified |

---

## Phase summary

| Phase | Name | Tasks | Done | Status |
|-------|------|-------|------|--------|
| P0 | Scaffold and config | 6 | 6 | `done` |
| P1 | Tokenizer, shards, manifests | 9 | 9 | `done` |
| P2 | Packing and batch builder | 7 | 7 | `done` |
| P3 | Mixture compiler and planner | 6 | 6 | `done` |
| P4 | Eval registry and firewall | 5 | 5 | `done` |
| P5 | OPUS selector and audit | 6 | 6 | `done` |
| P6 | Consumption ledger and checkpoints | 7 | 7 | `done` |
| P7 | Tiny training loop | 5 | 5 | `done` |
| P8 | Learning ledger | 4 | 4 | `done` |
| P9 | Crash, resume, replay, fork | 8 | 8 | `done` |
| P10 | Throughput and packing metrics | 4 | 4 | `done` |
| P11 | Demo orchestrator and evidence | 6 | 6 | `done` |
| P12 | Tests, README, submission | 5 | 5 | `done` |
| PX | CLI exceed (stretch) | 3 | 3 | `done` |
| **Total** | | **81** | **81** | |

*Update the Done column as tasks complete.*

---

## Handoff notes (2026-08-07, ship pass)

- **Completed:** P0–P12 and PX. Session 6 is fully done: demo, evidence, tests, pre-flight CLI, and
  SCOPE.md §17 checklist signed off (P12-T05).
- **Tests:** 220 passing (`pytest session06/assignment/tests -v`), including 6 preflight tests.
- **Config:** `config/loader.py` refactored to Pydantic (~95 lines); validation lives in
  `config/schemas.py`.
- **PX scripts:** `dry_run_dataset.py` (supply + admission + data card),
  `verify_artifacts.py` (CI-style invariant runner over `submission_artifacts/`).
- **Branching — resolved (2026-08-07): commit directly on `main`.** Tag `session06-complete` when
  the session ships.
- **Do not re-open:** D1, D2, D3, D6, D7, D8 (locked)

### Demo run observed (11 phases, `configs/demo.yaml`, ~10 s)

| Phase | Result |
|-------|--------|
| 1 Build shards | 8 shards, 7 admitted, 1 blocked |
| 2 Compile schedule | 50 steps, 3 stages, 0 supply warnings |
| 3 Train and crash | crashed at step 25 microbatch 1, 26 committed batches |
| 4 Resume | `ckpt-00020`, attempt 0 → 1, 26 retained, 34 re-committed |
| 5 Verify resume | 7 compared, 7 matched, 0 skipped, 0 repeated |
| 6 Replay | steps 20–25, 8 batches re-derived, 8 matched on planner, hashes, spans |
| 7 Fork | branch `run-a-fork-1` from `ckpt-00020`, 7 batches, diverges at step 20 |
| 8 Metrics | utilization 0.598 (concat 0.627, structure 0.435); ~2,200 loss-bearing tok/s, ~2,700 raw tok/s |
| 9 Gate activity | OPUS 50/27/13/10, 11 firewall blocks, 120 learning rows link to 60 batches |
| 10 Audit `run.log` | 195 events; all 14 SCOPE §9.1 event types present |
| 11 Evidence bundle | 14/14 requirements passed |

### Artifacts added during P11 (were missing from SCOPE §9)

| Artifact | Written by |
|----------|-----------|
| `manifests/tokenizer_manifest.json` | Phase 1, from the committed BPE artifact |
| `eval_registry.json` | `_build_context`, alongside the registry the firewall uses |
| `evidence.json` / `evidence.md` | Phase 11 collector |

### Full-run behavior observed at P7 (50 steps, `configs/demo.yaml`)

| Signal | Value |
|--------|-------|
| Model | 1.71M params (2 layers, d_model 128), ~8 s for 50 CPU steps |
| Loss | 9.24 → ~7.9 (decreasing, finite every step) |
| Microbatches | 62 committed, 24 OPUS-rejected, 14 OPUS-deferred |
| OPUS decisions | accepted, rejected, deferred, protected_override all present |
| Packing | 55 `concat_and_chop`, 7 `structure_preserving`; mean utilization 0.60 |
| Checkpoints | steps 10, 20, 30, 40, 50 with `model.pt` + `optimizer.pt` |

### Changes to earlier phases made during P7

| Change | Why |
|--------|-----|
| `configs/demo.yaml` gained an `optimizer:` block (+ `OptimizerConfig`) | P7-T03 needs LR, betas, weight decay, and grad clip; curriculum `lr_multiplier` scales it per stage |
| `LedgerBoundDataLoader.advance_after_skip()` | Gated microbatches must move the plan cursor without consuming a `ledger_offset` |
| `plan_step` draws without replacement within a step | The planner could put the same document in one microbatch twice, which packing rejects and which would double that document's effective epoch count |
| `CheckpointPayload.to_dict()` no longer embeds tensor states | Tensors are not JSON-serializable; they already go to `model.pt` / `optimizer.pt` sidecars. Only surfaced once P7 saved real weights |
| `candidate_from_planned_samples(batch=...)` | Lets the firewall's `assert_no_eval_loss` inspect the real loss mask; the loop now calls it before every training step |
| `encode_agentic_turns` replaces `loss_eligible_from_agentic_text` | Token IDs and role eligibility must come from the same per-line encoding to stay aligned |

---

## Open decisions (resolve before or during implementation)

| ID | Question | Options | Owner | Status |
|----|----------|---------|-------|--------|
| D1 | Corpus source | Toy JSONL (committed) vs Session 4 cleaned output | — | `resolved: committed toy JSONL` |
| D2 | Shard storage format | Binary arrays (pretrain) + JSONL (agentic) | — | `resolved: binary pretrain + JSONL agentic` |
| D3 | Curriculum stages in demo | 2 vs 3 stages | — | `resolved: 3 stages (foundation, skill_build, anneal)` |
| D4 | Training backend | PyTorch (preferred) vs minimal autograd | — | `resolved: PyTorch` |
| D5 | Learning ledger granularity | Sample-level (min) vs token-level (stretch) | — | `resolved: sample-level` |
| D6 | Visualization / exceed UX | Webapp vs CLI-only reports | — | `resolved: CLI-only exceed (no webapp)` |
| D7 | Frozen tokenizer | WordLevel scaffold vs Session 2 BPE (~10k) | — | `resolved: Session 2 BPE` |
| D8 | Re-committing offsets after resume | `attempt` field vs per-attempt files vs resume at tail | — | `resolved: attempt field, uniqueness (attempt, ledger_offset)` |

---

## P0 — Scaffold and config

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P0-T01 | Create folder layout per SCOPE.md §13 (`src/`, `scripts/`, `tests/`, `configs/`, `data/`) | — | `done` | |
| P0-T02 | Add `.gitignore` for `submission_artifacts/`, checkpoints, `__pycache__` | P0-T01 | `done` | |
| P0-T03 | Define `configs/demo.yaml` (seeds, seq_len, steps, crash/resume steps, checkpoint interval) | P0-T01 | `done` | |
| P0-T04 | Define `configs/curriculum.yaml` (2–3 stages, lane weights, Always-ON 11%, anneal reserve subset) | P0-T01, D3 | `done` | |
| P0-T05 | Add config loader module with validation (required keys, types, paths) | P0-T03, P0-T04 | `done` | |
| P0-T06 | Commit toy corpus skeleton under `data/toy_corpus/` (metadata schema only; content in P1) | P0-T01, D1 | `done` | |

### P0-T03 acceptance criteria

- [x] Fixed global seed documented in config
- [x] `seq_len`, `total_steps`, `checkpoint_interval`, `crash_at_step`, `resume_from_checkpoint` defined
- [x] Output path for `submission_artifacts/` configurable

### P0-T04 acceptance criteria

- [x] At least 2 capability lanes with non-zero weights per stage
- [x] `always_on_fraction: 0.11` documented
- [x] Stage token boundaries or step boundaries explicit

---

## P1 — Tokenizer, shards, manifests

**Maps to:** Subsystems A, B · Assignment items: immutable shards, frozen tokenizer, content hashes

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P1-T01 | Implement tokenizer wrapper with freeze semantics (load once, no runtime mutation) | P0-T05 | `done` | |
| P1-T02 | Compute and persist `tokenizer_hash` (vocab + merges + special tokens) | P1-T01 | `done` | |
| P1-T03 | Write `tokenizer_manifest.json` generator | P1-T02 | `done` | |
| P1-T03R | **Replace WordLevel with Session 2 BPE artifact; regenerate hash, manifest, tests** | P1-T03, D7 | `done` | |
| P1-T04 | Build corpus documents (50–100 docs, natural text, 3+ lanes minimum) | P0-T06, D1, P1-T03R | `done` | `scripts/build_corpus_documents.py` |
| P1-T05 | Implement shard builder: raw doc → token IDs → immutable shard file | P1-T01, P1-T04, D2 | `done` | `src/shards/` |
| P1-T06 | Implement shard manifest writer (all required fields per SCOPE.md §6.1) | P1-T05 | `done` | `src/shards/manifest.py` |
| P1-T07 | Implement admission gate (block missing hash, license, cleaning lineage, eval clearance) | P1-T06 | `done` | `src/shards/admission.py`, `registry.py` |
| P1-T08 | Add `scripts/build_shards.py` standalone entry point | P1-T07 | `done` | `scripts/build_shards.py` |

### P1-T03R acceptance criteria

- [x] `data/tokenizer/bpe_tokenizer.json` committed (from Session 2 BPE artifact)
- [x] `tokenizer_hash` includes non-empty merges; `merge_count > 0` in manifest
- [x] `model_type: BPE` in `tokenizer_manifest.json`
- [x] WordLevel-only tests updated or removed; BPE encode/decode tests pass
- [x] `FrozenTokenizer.load_default()` loads BPE artifact

### P1-T04 acceptance criteria

- [x] 50–100 documents with **natural language text** (not limited to a fixed word whitelist)
- [x] At least web, indic, and agentic lanes represented
- [x] `content_status: ready`; valid `content_sha256` per row

### P1-T05 acceptance criteria

- [x] Shard file write is atomic (temp + rename)
- [x] Re-running builder produces identical `content_hash`
- [x] Modifying shard bytes changes `content_hash` (new shard ID required)

### P1-T07 acceptance criteria

- [x] Missing `tokenizer_hash` → `admission: blocked`
- [x] `eval_overlap_status != clear` → blocked from training registry
- [x] Admitted shards listed in shard registry index

### P1 tests (link to P12)

- [x] `test_tokenizer_hash_stable`
- [x] `test_shard_content_hash_immutable`
- [x] `test_admission_gate_blocks_incomplete_manifest`

---

## P2 — Packing and batch builder

**Maps to:** Subsystems C, D · Assignment items: packing policies, loss/attention/position IDs

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P2-T01 | Define packing policy interface (`pack(docs, seq_len) → sequences`) | P1-T05 | `done` | `src/packing/` |
| P2-T02 | Implement concat-and-chop policy (web/pretrain lane) | P2-T01 | `done` | `src/packing/concat_and_chop.py` |
| P2-T03 | Implement structure-preserving policy (agentic/SFT lane) | P2-T01 | `done` | `src/packing/structure_preserving.py` |
| P2-T04 | Implement batch builder: `input_ids`, `loss_mask`, `attention_mask`, `position_ids` | P2-T02, P2-T03 | `done` | `src/batch/` |
| P2-T05 | Implement pretrain loss mask rule (next-token; exclude pad) | P2-T04 | `done` | `src/batch/masks.py` |
| P2-T06 | Implement agentic loss mask rule (assistant/tool-call only) | P2-T04 | `done` | `src/batch/masks.py`, `agentic.py` |
| P2-T07 | Compute `loss_mask_hash` and `batch_content_hash` per batch | P2-T04 | `done` | `src/batch/hash.py` |

### P2-T04 acceptance criteria

- [x] Causal attention mask for all modes
- [x] Document/span IDs preserved in batch metadata
- [x] Position ID policy recorded (absolute or reset-at-EOS) for ledger

### P2-T07 acceptance criteria

- [x] Identical batch tensors → identical hashes
- [x] Different loss mask → different `loss_mask_hash`

### P2 tests

- [x] `test_pretrain_loss_mask_excludes_pad`
- [x] `test_agentic_loss_mask_masks_user_turns`
- [x] `test_loss_mask_hash_stable`
- [x] `test_two_packing_policies_produce_different_utilization`

---

## P3 — Mixture compiler and sample planner

**Maps to:** Subsystem E · Assignment items: curriculum stages, lane weights, protected floors

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P3-T01 | Parse `curriculum.yaml` into stage records | P0-T04 | `done` | `src/schedule/stages.py` |
| P3-T02 | Implement mixture timeline compiler (stage → per-step lane quotas) | P3-T01 | `done` | `src/schedule/compiler.py` |
| P3-T03 | Emit `schedule.json` with compiled quotas and stage boundaries | P3-T02 | `done` | `src/schedule/io.py`, `scripts/compile_schedule.py` |
| P3-T04 | Implement Always-ON floor sampler (11% bypasses OPUS path) | P3-T02 | `done` | `src/schedule/always_on.py` |
| P3-T05 | Implement anneal reserve filter (`anneal_eligible` shards excluded until anneal stage) | P3-T02, P1-T06 | `done` | `src/schedule/filters.py` |
| P3-T06 | Implement deterministic sample planner (seed + step → shard/sample IDs) | P3-T02, P1-T07 | `done` | `src/schedule/planner.py`, `pool.py` |

### P3-T02 acceptance criteria

- [x] Stage transition visible in output (different lane mix across stages)
- [x] Warmup/blend band at transitions (no hard instant switch)
- [x] Supply shortfall flagged in compiler warnings (log or report)

### P3-T06 acceptance criteria

- [x] Same `(run_id, branch_id, step, seed)` → same planned sample IDs
- [x] Planner respects lane quotas from compiled schedule

### P3 tests

- [x] `test_schedule_compiler_stage_boundaries`
- [x] `test_planner_deterministic`
- [x] `test_always_on_fraction_met`

---

## P4 — Eval registry and firewall

**Maps to:** Subsystem F · Assignment items: evaluation and validation firewalls

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P4-T01 | Define eval registry schema (`never_train`, benchmark IDs, content hashes, canaries) | P1-T06 | `done` | `src/firewall/schema.py`, `types.py` |
| P4-T02 | Create at least one test/eval shard in registry with `never_train=true` | P4-T01, P1-T04 | `done` | `src/firewall/registry.py`, `scripts/build_eval_registry.py` |
| P4-T03 | Implement overlap checks (exact hash, canary string, benchmark overlap stub) | P4-T01 | `done` | `src/firewall/overlap.py` |
| P4-T04 | Implement firewall gate on candidate batches (block before loss assignment) | P4-T03, P2-T04 | `done` | `src/firewall/gate.py` |
| P4-T05 | Write firewall rejection events to `run.log` (not silently dropped) | P4-T04 | `done` | `src/firewall/log.py` |

### P4-T04 acceptance criteria

- [x] Candidate containing eval shard ID → blocked
- [x] Blocked batch never reaches consumption ledger with loss-bearing eval tokens
- [x] At least one firewall block demonstrated in demo run (11 blocks; `canary_string_match` on `doc-web-contaminated-001`)

### P4 tests

- [x] `test_firewall_blocks_never_train_shard`
- [x] `test_no_eval_token_in_loss_mask`

---

## P5 — OPUS selector and audit

**Maps to:** Subsystem G · Assignment items: OPUS accept/reject/defer, protected-floor override

**Status:** Code complete (P5-T01–T06). P5-T04 **demo** criterion (accepted, rejected, deferred, protected_override all visible in generated `opus_audit.jsonl`) deferred to **P11** Phase 10 / evidence collector.

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P5-T01 | Define OPUS scoring interface (deterministic, reproducible) | P3-T06 | `done` | `src/opus/scorer.py` |
| P5-T02 | Implement accept / reject / defer decision logic with configurable threshold | P5-T01 | `done` | `src/opus/selector.py` |
| P5-T03 | Implement protected-floor override (indic, agentic, reasoning lanes) | P5-T02, P3-T04 | `done` | `src/opus/selector.py` |
| P5-T04 | Append OPUS audit records to `opus_audit.jsonl` | P5-T02 | `done` | `src/opus/audit.py` |
| P5-T05 | Ensure rejected/deferred candidates remain queryable (not deleted) | P5-T04 | `done` | `src/opus/audit.py` |
| P5-T06 | Wire OPUS into batch pipeline (after firewall, before commit) | P4-T04, P5-T04 | `done` | `src/opus/pipeline.py` |

### P5-T04 acceptance criteria

- [x] Every accepted training batch has matching audit record with `opus_decision_id` (unit tests; full run in P11)
- [x] Demo includes at least one: accepted, rejected, deferred, protected_override (met by `scripts/run_demo.py`; asserted by `test_opus_audit_shows_every_decision_type`)
- [x] Scores reproducible from shard metadata + step (no random hidden state)

### P5 tests

- [x] `test_opus_deterministic_score`
- [x] `test_protected_floor_override`
- [x] `test_accepted_batch_has_audit_record`

---

## P6 — Consumption ledger and checkpoints

**Maps to:** Subsystems H, J · Assignment items: consumption ledger, checkpoints tied to ledger offsets

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P6-T01 | Define consumption ledger event schema (SCOPE.md §6.7) | P2-T07, P5-T04 | `done` | `src/ledger/types.py`, `schema.py` |
| P6-T02 | Implement append-only ledger writer with monotonic `ledger_offset` | P6-T01 | `done` | `src/ledger/writer.py` |
| P6-T03 | Log every committed batch (global step, hashes, shard/span IDs, stage, OPUS ID) | P6-T02 | `done` | `src/ledger/commit.py` |
| P6-T04 | Implement ledger reader / reconstructor for arbitrary step | P6-T02 | `done` | `src/ledger/reader.py` |
| P6-T05 | Define checkpoint payload (model, optimizer, scheduler, RNG, **ledger_offset**, **branch_id**, step) | P6-T02 | `done` | `src/checkpoint/types.py` |
| P6-T06 | Implement checkpoint save/load to `checkpoints/ckpt-{step}/` | P6-T05 | `done` | `src/checkpoint/io.py` |
| P6-T07 | Bind dataloader state to ledger offset (next batch = offset + 1, not fresh random) | P6-T04, P6-T06 | `done` | `src/ledger/dataloader.py` |

### P6-T07 acceptance criteria

- [x] After load, dataloader continues from saved `ledger_offset`
- [x] Checkpoint without `ledger_offset` rejected as incomplete
- [x] `branch_id` persisted and restored

### P6 tests

- [x] `test_ledger_append_only_monotonic_offsets`
- [x] `test_checkpoint_includes_ledger_offset_and_branch`
- [x] `test_reconstruct_batch_from_ledger_step`

---

## P7 — Tiny training loop

**Maps to:** Assignment item: real training consumption (with small model)

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P7-T01 | Define tiny causal LM config (2–4 layers, small dim, CPU-safe) | P0-T05, D4 | `done` | `src/trainer/model.py`, `configs/demo.yaml` |
| P7-T02 | Implement forward pass with masked loss (respect `loss_mask`) | P2-T05, P7-T01 | `done` | `src/trainer/loss.py` |
| P7-T03 | Implement training step (backward, optimizer update) | P7-T02 | `done` | `src/trainer/step.py` |
| P7-T04 | Integrate training step with batch pipeline and ledger commit | P6-T03, P7-T03 | `done` | `src/trainer/loop.py`, `src/batch/assemble.py` |
| P7-T05 | Save checkpoint on configured interval during training | P6-T06, P7-T04 | `done` | `src/trainer/loop.py` |

### P7-T02 acceptance criteria

- [x] Loss computed only on `loss_mask == 1` positions
- [x] Loss value finite and logged per step (`StepResult.mean_loss`; non-finite raises `TrainerError`)

### P7 design notes

- **One microbatch = one `Batch` = one consumption ledger row.** `assemble_microbatch`
  therefore packs a microbatch structure-preserving whenever it contains any agentic
  document: a `Batch` cannot mix packing policies, and concatenating a tool-call
  trajectory into an unrelated document would destroy the role boundaries the agentic
  loss mask depends on.
- **Gated microbatches advance the plan cursor, not `ledger_offset`.** The offset
  counts only batches the model learned from, so a checkpoint's `ledger_offset` is a
  true count of consumed batches. Both cursors are checkpointed independently, so
  resume stays exact.
- **Checkpoints are named for the step training resumes at** (`ckpt-00020` = restart at
  step 20), which is how `recovery.resume_from_checkpoint_step` reads.
- **Gradient accumulation scales by the planned count**, not the accepted count, so a
  rejected microbatch contributes nothing rather than reweighting its peers. A step
  where everything was rejected returns `optimizer_stepped=False, microbatches=0` and
  `mean_loss=None`: a step that trained on nothing has no loss, and `0.0` would read as
  a perfect score to any P8 or P10 aggregate.
- **`assert_no_eval_loss` runs before the training step.** The firewall and OPUS match
  on sample IDs and content hashes; this reads the assembled loss mask itself, so a
  never-train document reaching loss position fails loudly instead of silently
  training. Covered by `test_eval_document_carrying_loss_is_rejected_before_training`.

### P7 tests

- [x] `test_training_step_smoke` (CPU, 3 steps, ledger rows match committed microbatches)
- [x] `test_loss_respects_mask`
- [x] `test_gated_microbatches_do_not_consume_ledger_offsets`
- [x] `test_checkpoint_saved_on_interval_with_ledger_binding`
- [x] `test_loss_decreases_over_repeated_batch`
- [x] `test_agentic_microbatch_uses_structure_preserving_masking`
- [x] `test_eval_document_carrying_loss_is_rejected_before_training`

---

## P8 — Learning ledger

**Maps to:** Subsystem I · Assignment items: token/sample loss tracking, learning ledger

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P8-T01 | Define learning ledger event schema (shard/sample, loss, perplexity, stage, step) | P7-T04 | `done` | `src/ledger/learning.py` |
| P8-T02 | Append learning events to `learning.jsonl` after each training step | P8-T01 | `done` | `src/ledger/learning.py`, `src/trainer/loop.py` |
| P8-T03 | Aggregate per-shard loss / perplexity across steps | P8-T02, D5 | `done` | `src/ledger/learning_aggregate.py` |
| P8-T04 | Link learning entries back to consumption ledger (`global_step`, `shard_ids`) | P8-T02, P6-T03 | `done` | `src/ledger/learning_aggregate.py` |

### P8-T03 acceptance criteria

- [x] At least one shard shows loss trend across multiple steps
- [x] Perplexity derivable from recorded loss

### P8 design notes

- **One row per (committed microbatch, document), not per sequence.** D5 says
  sample-level. Concat-and-chop packs several documents into one sequence, so a
  per-sequence row would average two shards together and make per-shard aggregates
  fiction. `per_document_losses` splits the masked loss exactly.
- **Loss at position `i` belongs to `document_ids[i + 1]`**, the document that owns the
  predicted token. The prediction that straddles a packing boundary is scored against
  the next document's first token; crediting the previous document would give shard A
  tokens only shard B can explain.
- **Identity fields are copied from the consumption event**, never recomputed. A
  learning row cannot claim a step, offset, or batch hash the run did not commit, which
  is what makes `verify_learning_links` a real check instead of a tautology.
- **Losses are rounded before recording** (6 decimals), so an aggregate recomputed from
  `learning.jsonl` matches one computed from the in-memory events exactly.
- **Usefulness is assigned at aggregate time, not per row.** It is a loss trend across
  exposures, and a shard exposed once is `review` rather than a guess.

### Full-run behavior observed at P8 (50 steps, `configs/demo.yaml`)

| Signal | Value |
|--------|-------|
| Learning rows | 124 sample-level rows for 62 committed microbatches |
| Link report | `linked=True`; 0 orphans, 0 unreported offsets, 0 mismatches |
| Shards traced | 7 (web, indic, code, reasoning, agentic, stem, long_context) |
| Loss trend | web shard: 29 exposures, 9.23 → 7.93 (`loss_delta -1.30`) |
| Usefulness | 6 `useful`, 1 `review` (long_context, exposed once) |
| Model phases | early, mid, late, and anneal all present |

### Changes to earlier phases made during P8

| Change | Why |
|--------|-----|
| `MaskedLoss` now carries `token_loss` instead of `per_sequence_loss` / `per_sequence_tokens` | Sample-level attribution needs per-token detail grouped by document; per-sequence values were coarser than D5 requires and had no remaining consumer |
| `MicrobatchResult.per_document_loss` (new `DocumentLoss` type) | The learning ledger writes one row per document, so the trainer must return the split, not the sequence average |
| `TrainingPaths.learning_path` | `ledgers/learning.jsonl` alongside the consumption ledger and OPUS audit |
| `MicrobatchOutcome.learning` | Lets tests and the P11 evidence collector see that gated microbatches produced no learning rows |

### P8 tests

- [x] `test_learning_ledger_links_to_consumption_step`
- [x] `test_shard_loss_aggregate_recomputable`
- [x] `test_at_least_one_shard_shows_a_loss_trend`
- [x] `test_gated_microbatches_produce_no_learning_rows`
- [x] `test_ledger_file_is_append_only_jsonl`
- [x] `test_document_losses_partition_the_masked_tokens`
- [x] `test_perplexity_must_derive_from_recorded_loss`

---

## P9 — Crash, resume, replay, fork

**Maps to:** Subsystem K · Assignment items: crash recovery, replay, fork (highest grading risk)

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P9-T01 | Implement crash simulation (stop run at configured step, leave partial state) | P7-T05 | `done` | `src/recovery/crash.py` |
| P9-T02 | Implement **resume** from checkpoint + ledger offset | P6-T07, P9-T01 | `done` | `src/recovery/resume.py` |
| P9-T03 | Verify post-resume next batch matches pre-crash expected batch | P9-T02 | `done` | `src/recovery/verify.py` |
| P9-T04 | Write `reports/resume_verification.json` (expected vs actual hashes) | P9-T03 | `done` | `src/recovery/verify.py` |
| P9-T05 | Implement **replay** of step range from ledger | P6-T04 | `done` | `src/recovery/replay.py` |
| P9-T06 | Verify replayed batch IDs, spans, and hashes match original run | P9-T05 | `done` | `src/recovery/replay.py` |
| P9-T07 | Write `reports/replay_verification.json` | P9-T06 | `done` | `src/recovery/replay.py` |
| P9-T08 | Implement **fork** from checkpoint with new `branch_id` and divergence log | P6-T06 | `done` | `src/recovery/fork.py` |

### P9-T03 acceptance criteria

- [x] No skipped batches after resume
- [x] No repeated batches after resume
- [x] `batch_content_hash` at step N post-resume equals pre-crash record

### P9-T06 acceptance criteria

- [x] Replay range [K, M] produces identical `batch_content_hash` sequence as original
- [x] Token span IDs match ledger reconstruction

### P9-T08 acceptance criteria

- [x] New `branch_id` assigned at fork point
- [x] Ledger records parent branch, fork offset, and divergence step (`ledgers/forks.jsonl`)
- [x] Forked stream differs from replay stream after fork point

### P9-T01 design notes

- **The crash raises out of the training loop.** `TrainingRunner.run()` never returns, in-memory
  model and optimizer state are lost, and accumulated gradients are discarded unapplied. The caller
  catches `SimulatedCrash` and inspects disk, which is all a real crash leaves.
- **The abort fires at a microbatch boundary**, before any work for that microbatch, so the ledger
  tail is always a whole number of committed batches. `crash_after_microbatches=1` (default) kills the
  run mid-step; `0` kills it on a step boundary.
- **The dataloader cursor is the trigger.** `next_microbatch_index` already counts every attempted
  microbatch of the step, committed or gated, so no separate counter can drift from it.
- **`run.log` gets one `simulated_crash` line, and recovery must never read it.** It is evidence for
  the reader; resume has to work from the checkpoint and the ledger alone, or it is simulation.

### Crash state observed with `configs/demo.yaml` (crash_at_step 25, resume from 20)

| Signal | Value |
|--------|-------|
| Abort point | step 25, microbatch 1, `ledger_offset` 33 |
| Ledger tail | 34 consumption rows, 68 learning rows, last committed row at step 24 |
| Checkpoints on disk | `ckpt-00010`, `ckpt-00020`; none at or after the crash step |
| Resume checkpoint | `next_global_step=20`, `ledger_offset=26` |
| **Rows ahead of the resume checkpoint** | **7 (steps 20–24)** |

That last row is the P9-T02 problem: the ledger already holds the batches resume is about to
re-consume. See the open question below.

### D8 — ledger attempts (resolved: option A)

Resume restores a checkpoint that sits **behind** the ledger tail, so the rows in between have to
be re-committed without deleting anything. Resolution: **`attempt` on both ledger events;
uniqueness is `(attempt, ledger_offset)`.**

- The crashed attempt is retained and becomes the **expected** record; the resumed attempt is the
  **actual** one. P9-T03 compares them, and neither side is written by the verifier.
- `load_consumption_ledger` ordering is now: attempts never decrease; offsets increment by one
  *within* an attempt; a new attempt may not start past the previous tail (that would be a skipped
  batch).
- Lookups (`get_event_at_offset`, `reconstruct_at_global_step`) answer from the **newest** attempt
  holding the position, so "reconstruct step N" means what the model is actually carrying.
- `effective_events()` drops superseded learning rows before aggregation. A crashed attempt's loss
  describes weight updates that were rolled back; counting them would inflate exposure counts and
  the loss trend with learning the model does not have.
- Readers default a missing `attempt` to 0, which is precisely what a pre-D8 row was. No migration.

### P9-T02–T04 design notes

- **Resume restores four things together:** model/optimizer weights, RNG state, the data cursor
  (`next_global_step`, `next_microbatch_index`), and the ledger position opened as a new attempt at
  `ledger_offset + 1`. `resume_from_checkpoint` cross-checks the last two against each other, since
  a disagreement would either overwrite a row or leave a hole.
- **Recovery never reads the `simulated_crash` line.** Resume works from checkpoint plus ledger
  alone. If it needed the crash record it would be simulated recovery.
- **Resume refuses a checkpoint with no tensor state**, which would silently restart from random
  weights, and a checkpoint whose `run_id`/`branch_id` does not match the plan.
- **`verify_resume` distinguishes skipped from not-reached.** A batch the resumed run stopped short
  of is not a skip; only a gap *inside* the range it covered is.

### Fix to an earlier phase found by P9-T02

| Change | Why |
|--------|-----|
| `checkpoint/io.py` `restore_rng_state` coerces the Python RNG state back to tuples | `random.getstate()` returns nested tuples, but a checkpoint round-trips through JSON, which has no tuple type. `random.setstate` rejected the reloaded lists with "state vector must be a tuple". P6 never round-tripped RNG state through disk, so no test had reached the code path |

### Crash → resume observed with `configs/demo.yaml`

| Signal | Value |
|--------|-------|
| Crash | step 25, microbatch 1, `ledger_offset` 33 (34 rows) |
| Resume | `ckpt-00020`, `ledger_offset` 26, attempt 0 → 1 |
| Re-consumed | steps 20–24, 7 microbatches, ledger offsets 27–33 |
| Ledger after | 41 rows: 34 in attempt 0 (retained), 7 in attempt 1 |
| Verification | 7 compared, 7 matched, 0 skipped, 0 repeated, offsets contiguous, **passed** |
| Learning ledger | 82 rows across both attempts; link report clean |

### P9-T05–T08 design notes

- **Replay does not train and appends nothing.** It answers "can the run reproduce recorded
  history?" via two independent checks per microbatch: the planner is re-run and must produce the
  same sample IDs, and the batch is rebuilt (tokenize, pack, mask) and must produce the same hashes
  and token spans. Reading a hash back and comparing it to itself would prove nothing, so neither
  check does that.
- **Replay uses the effective stream.** Across a crash boundary the newest attempt per microbatch
  wins, because that is the batch the model actually carries.
- **A fork gets its own lineage on disk:** `branches/<branch_id>/`, offsets from 0. A branch is a
  different history, not a later part of the same one, so the parent's append-only ordering rules
  stay simple. The link lives in `ledgers/forks.jsonl` on the parent.
- **Forks diverge for free.** The planner is seeded on `branch_id`, so a new branch draws a
  different stream from the same pool and schedule. No policy change is needed to separate them.
- **Fork divergence is compared over the union of steps, not the intersection.** The branches gate
  differently, so one can commit where the other committed nothing. That is a real difference in
  what was consumed, and intersecting can leave nothing to compare at all.

### P4-T04 firewall demo: how the block is produced

`data/toy_corpus` gained `doc-web-contaminated-001`, a web row whose metadata says clean
(`eval_overlap_status: clear`, `never_train: false`) but whose text mirrors an MMLU holdout item.
It is admitted, enters the sample pool, and gets planned like any other document; only the
firewall's canary check on the actual text catches it.

Marking it `never_train` instead would have excluded it at the pool and proved nothing: the demo
would show the metadata filter working, not the firewall. The run produces 11 blocks with reason
`canary_string_match`, and the document never appears in `consumption.jsonl`.

### Fixes to earlier phases found by P9-T05–T08

| Change | Why |
|--------|-----|
| `TrainingRunner` rejects a fresh runner over a non-empty ledger | The dataloader would start at step 0 while the writer continued appending, silently re-training steps 0..N into the same history. Continuing a run is what resume and fork are for |
| `scripts/build_corpus_documents.py` is re-runnable | It reads `documents.jsonl` as its skeleton, so a second run hit "duplicate document_id" and could not add a row. EXTRA_DOCS is now the source of truth for those rows |
| `tests/test_ledger.py` skips past gated microbatches | Four P6 tests assumed the microbatch at a given cursor always commits. Adding one corpus document shifted the planner stream and broke them; the real loop skips, so the tests now do too |

### P9 tests (critical)

- [x] `test_crash_stops_at_configured_step` (P9-T01)
- [x] `test_crash_leaves_partial_step_in_ledger` (P9-T01)
- [x] `test_newest_checkpoint_is_behind_the_ledger_tail` (P9-T01)
- [x] `test_resume_no_skip_no_repeat`
- [x] `test_post_resume_batches_match_pre_crash_hashes` (P9-T03)
- [x] `test_crashed_rows_are_retained_not_overwritten` (D8 append-only)
- [x] `test_aggregates_exclude_rolled_back_attempts` (D8 supersession)
- [x] `test_writes_resume_verification_report` (P9-T04)
- [x] `test_verification_fails_when_a_batch_diverges` (the report can fail, not only pass)
- [x] `test_replay_hash_match`
- [x] `test_fork_new_branch_id`
- [x] `test_replay_detects_a_tampered_ledger` (replay recomputes, it does not read back)
- [x] `test_forked_stream_diverges_from_the_parent`
- [x] `test_fork_event_records_parent_and_offset`
- [x] `test_fresh_runner_refuses_to_retrain_over_an_existing_ledger`
- [x] `test_firewall_blocks_a_contaminated_document_in_the_run` (P4-T04)

---

## P10 — Throughput and packing metrics

**Maps to:** Subsystem L · Assignment items: packing utilization, useful loss-bearing tokens/sec

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P10-T01 | Compute packing utilization per batch and run aggregate | P2-T07, P7-T04 | `done` | `src/metrics/packing.py` |
| P10-T02 | Write `reports/packing_utilization.json` | P10-T01 | `done` | `src/metrics/packing.py` |
| P10-T03 | Measure useful loss-bearing tokens/sec and raw tokens/sec | P7-T04 | `done` | `src/metrics/timing.py`, `throughput.py` |
| P10-T04 | Write `reports/throughput.json` (recomputable from ledger + timings) | P10-T03 | `done` | `src/metrics/throughput.py` |

### P10-T02 acceptance criteria

- [x] Utilization formula documented: `useful_tokens / (seq_len × num_sequences)` (emitted in the report as `formula`)
- [x] Report values match independent recomputation from batch files/ledger

### P10 design notes

- **Timings live outside the consumption ledger.** The ledger records what was consumed, which is
  reproducible; a duration is a property of the machine and changes every run. Putting it in the
  ledger would add a non-reproducible field to the record that resume and replay verify against.
  Durations go to `reports/step_timings.jsonl` instead.
- **That separation is what makes throughput checkable.** Token counts come from
  `consumption.jsonl` (batches rebuilt and re-counted), seconds come from the timings file, and the
  reported rate is the join of two independently generated artifacts. Neither file records a rate,
  so the numbers cannot be stale or invented.
- **Nothing is read back from a recorded metric.** Every batch is rebuilt through the same
  `rebuild_batch` path replay uses, then re-counted. A packing claim that cannot be reconstructed
  earns no credit (ASSIGNMENT.md), so the report is derived, never cached.
- **Every attempt counts, unlike the learning aggregates.** A crashed attempt's batches were built,
  padded, and timed for real, so they belong in a performance measure even though their weight
  updates were rolled back. The learning ledger makes the opposite choice for the opposite reason.
- **The step clock spans gating.** A microbatch the firewall or OPUS discarded still cost wall time;
  excluding it would overstate throughput.
- **Run utilization is token-weighted**, not the mean of per-batch ratios, so a two-sequence batch
  does not count as much as an eight-sequence one.
- **Steps missing a timing row are reported, not assumed instantaneous**, which would inflate the
  rate. `steps_without_timings` surfaces them.

### Metrics observed in the demo run

| Signal | Value |
|--------|-------|
| Utilization | 0.598 over 60 batches |
| By policy | `concat_and_chop` 0.627, `structure_preserving` 0.435 (one sequence per document pads more) |
| Throughput | 2,209 loss-bearing tokens/s; 2,670 raw tokens/s over 3.84 s of measured step time |

The gap between the two rates is the point: roughly 17% of the non-pad tokens the model saw were
masked out of the loss and moved no gradient.

### P10 tests

- [x] `test_packing_utilization_recomputable`
- [x] `test_throughput_metrics_from_ledger`
- [x] `test_run_utilization_is_token_weighted`
- [x] `test_missing_timings_are_reported_not_treated_as_free`
- [x] `test_metrics_reports_are_recomputable_from_the_artifacts` (demo output)

---

## P11 — Demo orchestrator and evidence bundle

**Maps to:** Submission requirements · One-command demo · evidence.json/md

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P11-T01 | Implement structured `run.log` writer (all event types per SCOPE.md §9.1) | P6-T03 | `done` | `src/runlog/` |
| P11-T02 | Implement `scripts/run_demo.py` orchestrating full demo sequence | — | `done` | `scripts/run_demo.py`, `src/demo/pipeline.py` |
| P11-T03 | Demo phase: build shards → compile schedule → train → crash → resume → replay → fork | P11-T02 | `done` | 11 phases |
| P11-T04 | Implement evidence collector (read artifacts, run checks, no hardcoding) | P11-T03 | `done` | `src/evidence/checks.py`, `collector.py` |
| P11-T05 | Generate `evidence.json` with pass/fail per requirement + evidence paths | P11-T04 | `done` | `src/evidence/collector.py` |
| P11-T06 | Generate `evidence.md` human-readable summary from `evidence.json` | P11-T05 | `done` | `src/evidence/report.py` |

### P11-T01 design notes

- **One writer per log file.** `seq` comes from an in-memory counter, so two writers open on one
  path would hand out the same numbers and the ordering claims in the log (crash before resume,
  commit before checkpoint) would stop meaning anything. The demo creates one `RunLogWriter` and
  passes it into the trainer and the recovery functions; a fork writes to its own branch log.
  `load_run_log` enforces strictly increasing `seq`, so the mistake fails loudly rather than
  quietly.
- **`EVENT_TYPES` is the SCOPE.md §9.1 list, and `emit` rejects anything else.** A misspelled event
  type would otherwise produce a line no reader looks for, and the log would appear complete while a
  required event was missing.
- **Only a stage *change* is a transition.** A resumed runner starts with no previous stage;
  logging its first step as a transition would invent a curriculum change the schedule never had.
- **Firewall and crash logging were rerouted through the writer.** They previously appended to
  `run.log` directly, which left their lines outside the sequence.
- **The demo audits its own log (phase 10).** It reads the finished file back and checks it against
  `EVENT_TYPES`. Whether the events were emitted is a property of the artifact, not of the code that
  was supposed to emit them.

### P11-T04–T06 design notes

- **Every check reads artifacts and recomputes.** Shard hashes are recomputed from the bytes on
  disk, the tokenizer hash from the committed BPE artifact, resume is re-verified from the ledger
  using only the *parameters* the report records, and every rate and ratio is redone from per-row
  data. Where a report's verdict is used, it is compared against a recomputation, never trusted.
- **`tests/test_evidence.py` proves the checks can fail.** Eleven tests corrupt one artifact each (a
  shard byte, a tokenizer hash, a schedule step, a ledger row, the OPUS audit, each verification
  report) and require the matching requirement to flip to failed. A hardcoded `passed: true` would
  survive all of them.
- **`evidence_path` is always a file; `evidence_paths` may include directories.** A path a grader
  cannot open is not evidence, so the collector fails the bundle when the primary path is not a
  readable file.
- **`evidence.json` keys are left unsorted**, so requirements appear in SCOPE.md §9.2 order.
- **`evidence.md` is rendered from the bundle only**, so the two files cannot disagree.
- **Phase 11 runs last and can fail a passing run.** The artifacts are the submission, not the code
  path that wrote them.

### P11-T02 acceptance criteria

- [x] Single command runs unattended from clean state
- [x] Regenerates full `submission_artifacts/` directory
- [x] Exit code non-zero if any verification fails

### P11-T02 note: dependency inverted on purpose

TASKS.md originally had P11-T02 depending on P9-T08 and P10-T04. That ordering front-loads the
lowest-risk work: grading Step 1 runs the command, so until it exists a correct implementation
scores nothing on execution. The orchestrator was therefore built against the phases that already
worked, and the remaining phases slot into `src/demo/pipeline.py` as they land.

`PENDING_PHASES` lists what is still missing, and `run_demo.py` prints it. A partial demo that says
so is honest; one that quietly omits replay and fork is not.

### Demo run observed (`scripts/run_demo.py`, 7.9 s)

| Phase | Result |
|-------|--------|
| 1 Build shards | 8 shards, 7 admitted, 1 blocked by the admission gate |
| 2 Compile schedule | 50 steps across foundation, skill_build, anneal; 0 supply warnings |
| 3 Train and crash | crashed at step 25 microbatch 1; 34 committed batches |
| 4 Resume | `ckpt-00020`, attempt 0 → 1; 34 rows retained, 35 re-committed |
| 5 Verify resume | 7 compared, 7 matched, 0 skipped, 0 repeated |
| 6 Gate activity | OPUS accepted 58 / rejected 26 / deferred 16 / protected_override 11; **0 firewall blocks** |

Artifacts: 39 files (5 checkpoints with tensors, 3 ledgers, 9 manifests, `schedule.json`, `run.log`,
`reports/resume_verification.json`).

**P5-T04's deferred demo criterion is now met**: all four OPUS decision types appear in generated
`opus_audit.jsonl`, asserted by `test_opus_audit_shows_every_decision_type`.

**P4-T04's deferred criterion is still open**: the demo run produces 0 firewall blocks, so no eval
candidate is being planned into a batch. Needs a planned candidate that the firewall rejects.

### P11-T05 acceptance criteria

- [x] All 14 requirement keys from SCOPE.md §9.2 present (`REQUIREMENT_KEYS`; collector raises if the checks do not cover it exactly)
- [x] Each `evidence_path` points to existing file (asserted by `test_every_evidence_path_points_at_a_real_artifact`)
- [x] `passed` values computed from verification outputs, not static literals (`TestEvidenceDetectsTampering`)
- [x] OPUS audit completeness includes P5-T04 demo criterion: `opus_audit.jsonl` shows accepted, rejected, deferred, and protected_override

### P11 demo sequence checklist

- [x] Phase 1: Build shards + manifests
- [x] Phase 2: Compile schedule
- [x] Phase 3: Train N steps with ledger logging
- [x] Phase 4: Checkpoint at step K
- [x] Phase 5: Continue to step M
- [x] Phase 6: Simulate crash at M
- [x] Phase 7: Resume from K; verify next batch
- [x] Phase 8: Replay K..M; verify hashes
- [x] Phase 9: Fork from K; verify new branch
- [x] Phase 10: Firewall + OPUS edge cases exercised (all four OPUS decisions; 11 firewall blocks)
- [x] Phase 11: Emit evidence bundle

### P11 tests

- [x] `test_every_line_is_one_json_event_with_the_envelope`
- [x] `test_unknown_event_type_is_rejected`
- [x] `test_reopening_continues_the_sequence`
- [x] `test_duplicate_sequence_numbers_are_an_error`
- [x] `test_missing_event_types_reports_the_scope_vocabulary_gap`
- [x] `test_run_log_covers_every_event_type_scope_requires` (demo output)
- [x] `test_run_log_batch_commits_match_the_consumption_ledger`
- [x] `test_run_log_verification_events_agree_with_the_reports`
- [x] `test_all_fourteen_requirement_keys_are_present_and_passing`
- [x] `test_every_evidence_path_points_at_a_real_artifact`
- [x] `TestEvidenceDetectsTampering` (11 corruption tests, one per requirement family)

---

## P12 — Tests, README, submission

**Maps to:** ASSIGNMENT.md submission list · Core invariants (SCOPE.md §11)

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| P12-T01 | Implement all invariant tests listed in P1–P10 test sections | P1–P10 | `done` | audited; see invariant map below |
| P12-T02 | Add integration test: mini end-to-end run (5 steps, ledger + checkpoint) | P11-T02 | `done` | `tests/test_demo_pipeline.py::TestDemoEndToEnd` |
| P12-T03 | Write `README.md` (architecture, design decisions, one-command run, test instructions) | P11-T02 | `done` | `README.md` |
| P12-T04 | Verify clean regeneration: delete artifacts → run demo → all checks pass | P11-T02 | `done` | directory deleted entirely, demo exits 0 |
| P12-T05 | Final submission checklist review (SCOPE.md §17) | P12-T01–P12-T04 | `done` | reviewed 2026-08-07 |

### P12-T01 invariant map (SCOPE.md §11 → test)

| # | Invariant | Test |
|---|-----------|------|
| 1 | Tokenizer immutability | `test_tokenizer_change_invalidates_shard_binding` |
| 2 | Loss mask correctness; eval tokens masked | `test_pretrain_loss_mask_excludes_pad`, `test_no_eval_token_in_loss_mask` |
| 3 | `loss_mask_hash` stability | `test_loss_mask_hash_stable`, `test_different_loss_mask_changes_loss_mask_hash` |
| 4 | Ledger append-only, monotonic offsets | `test_ledger_append_only_monotonic_offsets`, `test_ledger_file_is_append_only_jsonl` |
| 5 | Checkpoint completeness | `test_checkpoint_includes_ledger_offset_and_branch` |
| 6 | Resume exactness | `test_resume_no_skip_no_repeat`, `test_post_resume_batches_match_pre_crash_hashes` |
| 7 | Replay exactness | `test_replay_hash_match`, `test_replay_detects_a_tampered_ledger` |
| 8 | Fork divergence | `test_fork_new_branch_id`, `test_forked_stream_diverges_from_the_parent`, `test_fork_event_records_parent_and_offset` |
| 9 | Firewall blocks never_train | `test_firewall_blocks_never_train_shard`, `test_firewall_blocks_a_contaminated_document_in_the_run` |
| 10 | OPUS audit completeness | `test_accepted_batch_has_audit_record`, `test_opus_audit_shows_every_decision_type` |
| 11 | Protected floor / Always-ON | `test_always_on_fraction_met`, `test_protected_floor_override` |
| 12 | Packing utilization recomputable | `test_packing_utilization_recomputable`, `test_run_utilization_is_token_weighted` |

### P12-T02 note

The integration test runs the *full* demo (50 steps, crash, resume, replay, fork, metrics,
evidence) into a temp directory rather than a 5-step miniature. A 5-step run cannot reach a
checkpoint interval, a stage transition, or a crash point, so it would not exercise what the
integration test exists to cover.

### P12-T03 acceptance criteria

- [x] Architecture diagram or flow included (Mermaid flowchart of the data path)
- [x] Documents why toy scale choices were made ("reduce data and compute, not system realism")
- [x] Exact demo command documented
- [x] How to run tests documented

### P12-T04 evidence

`submission_artifacts/` was deleted outright (not emptied) and `run_demo.py` regenerated the full
tree with exit code 0 and 14/14 requirements passed. `_clean_artifacts_dir` also empties the
directory on every normal run, keeping `.gitkeep`; `test_clean_run_removes_stale_artifacts_but_keeps_gitkeep`
asserts stale files do not survive.

---

## Requirement → task traceability

Quick map from ASSIGNMENT.md bullets to task IDs:

Each row is also a key in `evidence.json`, checked against generated artifacts by
`src/evidence/checks.py`.

| Assignment requirement | Primary tasks |
|------------------------|---------------|
| Immutable tokenized shards with manifests | P1-T05, P1-T06, P1-T07 |
| Frozen tokenizer and content hashes | P1-T01, P1-T02, P1-T03 |
| Packing policies for different data types | P2-T02, P2-T03 |
| Correct loss, attention, position IDs | P2-T04, P2-T05, P2-T06 |
| Curriculum stages, lane weights, protected floors | P3-T02, P3-T04, P3-T05 |
| Evaluation and validation firewalls | P4-T01–P4-T05 |
| OPUS accept/reject/defer/override | P5-T02–P5-T06 |
| Training consumption ledger | P6-T01–P6-T04 |
| Learning ledger + loss tracking | P8-T01–P8-T04 |
| Checkpoints tied to ledger offsets | P6-T05–P6-T07 |
| Crash recovery (no skip/repeat) | P9-T01–P9-T04 |
| Replay historical stream | P9-T05–P9-T07 |
| Fork from earlier checkpoint | P9-T08 |
| Packing utilization + useful tokens/sec | P10-T01–P10-T04 |
| One-command demo | P11-T02 |
| Automated tests | P12-T01, P12-T02 |
| Execution log + evidence bundle | P11-T01, P11-T05, P11-T06 |

---

## PX — CLI exceed (stretch, reusable on real dataset)

**Purpose:** Pre-flight audit and documentation beyond minimum grading. GitHub-repo friendly: scripts + JSON/Markdown reports only (decision D6).

**Maps to:** Production dry-run before large runs; portable to Session 4 cleaned output later.

| ID | Task | Depends on | Status | Assignee |
|----|------|------------|--------|----------|
| PX-T01 | `scripts/dry_run_dataset.py`: supply report (lane tokens vs curriculum quotas) + admission audit | P3-T03, P1-T07 | `done` | `scripts/dry_run_dataset.py`, `src/preflight/supply.py` |
| PX-T02 | Auto-generate `reports/data_card.md` + `reports/data_card.json` from corpus, tokenizer, shard registry | PX-T01, P4-T01 | `done` | `src/preflight/data_card.py` |
| PX-T03 | `scripts/verify_artifacts.py`: CI-style invariant runner over `submission_artifacts/` | P11-T05, P9-T04 | `done` | `scripts/verify_artifacts.py`, `src/preflight/verify.py` |

### PX-T01 acceptance criteria

- [x] Reads compiled `schedule.json` + admitted shard registry (no training run)
- [x] Writes `reports/dataset_supply.json` flagging under-supplied lanes per stage
- [x] Writes `reports/admission_audit.json` with admitted/blocked shards and block reasons

### PX-T02 acceptance criteria

- [x] Data card includes tokenizer hash, corpus doc counts by lane, admitted token totals
- [x] Regenerated identically from same artifacts (deterministic)

### PX-T03 acceptance criteria

- [x] Exit code non-zero when any SCOPE §11 invariant fails
- [x] Usable in CI: `python scripts/verify_artifacts.py submission_artifacts/`

---

## Suggested sprint plan

| Sprint | Focus | Target tasks | Exit criterion |
|--------|-------|--------------|----------------|
| **S1** | Foundation | P0, P1, start P2 | Shards + one packed batch with hashes |
| **S2** | Batch + schedule | P2, P3, P4 | Masks correct; schedule compiles; firewall blocks eval |
| **S3** | Pipeline core | P5, P6, P7 | Train 20 steps; ledger + checkpoint work |
| **S4** | Recovery | P8, P9 | Resume/replay/fork verification JSONs pass |
| **S5** | Ship | P10, P11, P12, PX | One command; evidence green; dry-run + verify CLI; README done |

---

## Submission gate (all must pass)

Copied from SCOPE.md §17; tick when complete:

- [x] One command runs full demo end-to-end (P11-T02) — 11 phases, exit 0
- [x] `submission_artifacts/` regenerates cleanly (P12-T04)
- [x] `evidence.json` all requirements passed (P11-T05) — 14/14
- [x] Crash → resume identical next batch (P9-T03, P9-T04)
- [x] Replay hash match (P9-T06, P9-T07)
- [x] Fork new branch logged (P9-T08)
- [x] Eval never in loss-bearing batch (P4-T04)
- [x] OPUS audit complete (P5-T04)
- [x] Consumption ledger reconstructs any step (P6-T04) — every step in the run, checked by the evidence collector
- [x] Learning ledger links shards to loss (P8-T04)
- [x] Packing + throughput reproducible (P10-T02, P10-T04)
- [x] Automated tests pass (P12-T01) — 220 passing
- [x] README complete (P12-T03)
- [x] P12-T05 final checklist review (SCOPE.md §17) — 2026-08-07

---

*Last updated: 2026-08-07 · Session 6 complete: P0–P12 + PX (81/81 tasks). Config loader uses Pydantic; pre-flight CLI ships supply audit, data card, and verify_artifacts.*
