# Session 6 Evidence Report

**Result:** PASS (14/14 requirements passed)

- Generated at: `2026-08-07T16:51:47Z`
- Demo command: `uv run python scripts/run_demo.py`
- Git commit: `0bf8393c794f49535fe68fcdf0a16c1cd8133bab`
- Artifacts: `submission_artifacts/`

Every value below was computed by reading the generated artifacts. Nothing in this file is written by hand, and no requirement is marked passed from a literal in the source.

## Summary

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Immutable tokenized shards with manifests | PASS | `manifests/shard_registry.json` |
| Frozen tokenizer and content hashes | PASS | `manifests/tokenizer_manifest.json` |
| Packing policies for different data types | PASS | `reports/packing_utilization.json` |
| Correct loss, attention, and position IDs | PASS | `ledgers/consumption.jsonl` |
| Curriculum stages, lane weights, protected floors | PASS | `schedule.json` |
| Evaluation and validation firewall | PASS | `eval_registry.json` |
| OPUS accept / reject / defer / override audit | PASS | `ledgers/opus_audit.jsonl` |
| Training consumption ledger | PASS | `ledgers/consumption.jsonl` |
| Learning ledger linked to consumption | PASS | `ledgers/learning.jsonl` |
| Checkpoints tied to ledger offsets | PASS | `checkpoints/ckpt-00010/checkpoint.json` |
| Crash recovery with no skipped or repeated batch | PASS | `reports/resume_verification.json` |
| Replay of a historical range matches recorded hashes | PASS | `reports/replay_verification.json` |
| Fork from an earlier checkpoint onto a new branch | PASS | `reports/fork_verification.json` |
| Packing utilization and useful tokens/sec | PASS | `reports/packing_utilization.json` |

## Detail

### Immutable tokenized shards with manifests (PASS)

Key: `immutable_shards_with_manifests` · Evidence: `manifests/shard_registry.json`, `manifests`, `shards`

- **PASS** shard files rehash to their manifest content_hash: 8 manifests, 8 shard files; mismatched: none
- **PASS** shard_id is derived from the content hash: mismatched: none
- **PASS** the registry admits exactly the manifests the gate admitted: registry 7 admitted, manifests 7 admitted, 1 blocked (['shard_83aabab3ff67'])
- **PASS** the admission gate blocked at least one shard: blocked: ['shard_83aabab3ff67']

### Frozen tokenizer and content hashes (PASS)

Key: `frozen_tokenizer_hashes` · Evidence: `manifests/tokenizer_manifest.json`, `manifests`

- **PASS** manifest tokenizer_hash recomputes from the tokenizer artifact: recomputed tok_f841ef5bcf29 vs manifest tok_f841ef5bcf29 (bpe_tokenizer.json)
- **PASS** the frozen tokenizer is BPE with a non-empty merge table: model_type BPE, 9980 merges, vocab 10000
- **PASS** every shard was sealed with that tokenizer hash: shard manifest hashes: ['tok_f841ef5bcf29']
- **PASS** every consumed batch records that tokenizer hash: ledger hashes: ['tok_f841ef5bcf29']

### Packing policies for different data types (PASS)

Key: `packing_policies` · Evidence: `reports/packing_utilization.json`

- **PASS** both packing policies ran: policies: ['concat_and_chop', 'structure_preserving']
- **PASS** the aggregate covers exactly the policies the batches used: batch rows used ['concat_and_chop', 'structure_preserving']
- **PASS** the policies pack to different utilization: concat_and_chop 0.627, structure_preserving 0.435

### Correct loss, attention, and position IDs (PASS)

Key: `correct_masks` · Evidence: `ledgers/consumption.jsonl`, `reports/packing_utilization.json`

- **PASS** every consumed batch used a causal attention mask: attention policies: ['causal']
- **PASS** every consumed batch recorded its position ID policy: position policies: ['reset_at_document_boundary']
- **PASS** every consumed batch carries a loss mask hash: 60 rows; unhashed: none
- **PASS** 0 < loss-bearing tokens <= non-pad tokens <= capacity for every batch: 60 batches recounted; violations: none
- **PASS** some tokens were seen but excluded from the loss: 60 of 60 batches mask part of what they read (loss-bearing fraction 0.495)

### Curriculum stages, lane weights, protected floors (PASS)

Key: `curriculum_and_floors` · Evidence: `schedule.json`, `ledgers/opus_audit.jsonl`

- **PASS** the schedule compiles at least three curriculum stages: stages: ['foundation', 'skill_build', 'anneal']
- **PASS** every step meets the Always-ON floor: floor 0.11; 50 steps; below floor: none
- **PASS** the run consumed batches under more than one stage: stages in the ledger: ['anneal', 'foundation', 'skill_build']
- **PASS** stage transitions were logged and name compiled stages: 2 transitions into ['anneal', 'skill_build']
- **PASS** protected-floor lanes bypassed OPUS at least once: 10 overrides on lanes ['indic', 'reasoning']; protected lanes ['agentic', 'indic', 'reasoning']

### Evaluation and validation firewall (PASS)

Key: `eval_firewall` · Evidence: `eval_registry.json`, `run.log`

- **PASS** the registry marks at least one document never_train: never_train documents: ['doc-eval-001']
- **PASS** no never-train document reached the consumption ledger: 41 distinct samples consumed; leaked: none
- **PASS** the firewall blocked at least one candidate during the run: 11 blocks, reasons ['canary_string_match']
- **PASS** no blocked candidate was later committed: blocked candidates also committed: none

### OPUS accept / reject / defer / override audit (PASS)

Key: `opus_audit_trail` · Evidence: `ledgers/opus_audit.jsonl`, `ledgers/consumption.jsonl`

- **PASS** all four OPUS decision types appear: decisions: ['accepted', 'deferred', 'protected_override', 'rejected']
- **PASS** every committed batch has a matching audit record: 60 committed batches; unmatched: none
- **PASS** a committed batch's audit record says it was accepted: committed under a non-accepting decision: none
- **PASS** rejected and deferred candidates stayed queryable: 40 of 100 audit records are rejections or deferrals, still on file

### Training consumption ledger (PASS)

Key: `consumption_ledger` · Evidence: `ledgers/consumption.jsonl`

- **PASS** the ledger loads under append-only ordering rules: 60 rows across attempts [0, 1]; load_consumption_ledger rejects a decreasing attempt, a non-incrementing offset, or an attempt starting past the previous tail
- **PASS** offsets increment by one within every attempt: offsets per attempt: attempt 0 0..25, attempt 1 19..52
- **PASS** every step in the run reconstructs from the ledger: 40 steps reconstructed; failed: none
- **PASS** a crashed attempt's rows were retained, not overwritten: attempts on file: [0, 1]

### Learning ledger linked to consumption (PASS)

Key: `learning_ledger` · Evidence: `ledgers/learning.jsonl`, `ledgers/consumption.jsonl`

- **PASS** every learning row joins to a committed batch: 120 learning rows against 60 committed batches; 0 orphans, 0 unreported, 0 mismatches
- **PASS** at least one shard shows a loss trend across exposures: shard_b97a1486440d 28 exposures, loss_delta -0.459; shard_f40fe4d44010 27 exposures, loss_delta -1.159; shard_70fd8df8589f 16 exposures, loss_delta -1.367
- **PASS** perplexity re-derives from the recorded loss: 7 shard aggregates; mismatched: none

### Checkpoints tied to ledger offsets (PASS)

Key: `checkpoint_ledger_binding` · Evidence: `checkpoints/ckpt-00010/checkpoint.json`, `checkpoints/ckpt-00020/checkpoint.json`, `checkpoints/ckpt-00030/checkpoint.json`, `checkpoints/ckpt-00040/checkpoint.json`, `checkpoints/ckpt-00050/checkpoint.json`, `ledgers/consumption.jsonl`

- **PASS** checkpoints exist: 5 checkpoints: ['ckpt-00010', 'ckpt-00020', 'ckpt-00030', 'ckpt-00040', 'ckpt-00050']
- **PASS** every checkpoint records ledger_offset and branch_id: unbound: none
- **PASS** every checkpoint's tensor sidecars are on disk: missing model/optimizer state: none
- **PASS** no checkpoint claims an offset past the ledger tail: ledger tail 52; ahead of it: none

### Crash recovery with no skipped or repeated batch (PASS)

Key: `crash_resume_no_skip_repeat` · Evidence: `reports/resume_verification.json`, `ledgers/consumption.jsonl`

- **PASS** the run crashed and then resumed: crash at step 25, resume from step 20
- **PASS** re-verifying the ledger reproduces the report's verdict: recomputed passed=True, report passed=True
- **PASS** no batch was skipped and none was repeated: 7 batches compared, 0 skipped, 0 repeated
- **PASS** post-resume batch hashes equal the pre-crash record: 7 of 7 matched on content hash, mask hash, sample IDs, and token spans

### Replay of a historical range matches recorded hashes (PASS)

Key: `replay_hash_match` · Evidence: `reports/replay_verification.json`, `ledgers/consumption.jsonl`

- **PASS** a historical range was replayed: steps 20..25, 8 batches replayed
- **PASS** every rebuilt batch hashes to what was recorded: content and loss-mask hash mismatches: none
- **PASS** the planner re-drew the same samples and spans: planner or span mismatches: none
- **PASS** the replayed hashes match the consumption ledger row by row: joined on (attempt, ledger_offset); disagreements: none

### Fork from an earlier checkpoint onto a new branch (PASS)

Key: `fork_new_branch` · Evidence: `reports/fork_verification.json`, `ledgers/forks.jsonl`, `branches/run-a-fork-1/ledgers/consumption.jsonl`

- **PASS** the fork was given a new branch_id: parent run-a, child run-a-fork-1
- **PASS** the parent's fork log records the parent branch and fork offset: 1 fork events; linked: from ckpt step 20 at parent offset 18
- **PASS** the child branch has its own ledger, written under its own branch_id: 7 rows at branches/run-a-fork-1/ledgers/consumption.jsonl; branch_ids ['run-a-fork-1']
- **PASS** the forked stream diverges from the parent after the fork point: diverges at step 20 across 25 of 25 compared steps

### Packing utilization and useful tokens/sec (PASS)

Key: `packing_and_throughput` · Evidence: `reports/packing_utilization.json`, `reports/throughput.json`, `reports/step_timings.jsonl`

- **PASS** run utilization recomputes from the per-batch rows: 10257/17152 = 0.598006 vs reported 0.598006 (useful_tokens / (seq_len * num_sequences))
- **PASS** the reported wall time is the sum of the recorded step timings: 4.0551s from 55 timing rows vs reported 4.0551s
- **PASS** loss-bearing tokens/sec recomputes from tokens and seconds: 8488 tokens / 4.06s = 2093.2 vs reported 2093.2
- **PASS** no step was measured without a timing, and useful <= raw throughput: 45 steps measured, 0 untimed; 2093 useful vs 2529 raw tokens/s
