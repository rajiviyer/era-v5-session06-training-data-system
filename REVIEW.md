# Session 6 Assignment — Implementation Review Guide

**Purpose:** Keep the Training Data Execution System **correct, auditable, and shippable** without unnecessary complexity.

**Read first:** [MENTOR.md](MENTOR.md) defines scale, realism (BPE not WordLevel), and how AI assistants should mentor you. This doc complements it with code-review guardrails.

**Use this doc when:**
- Starting a new task in [TASKS.md](TASKS.md)
- Reviewing a PR or your own diff before commit
- Deciding whether an abstraction is worth adding

**Related docs:**
- [MENTOR.md](MENTOR.md) — architecture defaults, learning path, AI mentor protocol
- [ASSIGNMENT.md](ASSIGNMENT.md) — grading contract
- [SCOPE.md](SCOPE.md) — architecture, invariants, artifacts
- [TASKS.md](TASKS.md) — phased task tracker

---

## 0. Realism guardrails (not optional)

Medium scale does **not** mean basic shortcuts.

| Must be realistic | Too basic (reject) |
|-------------------|-------------------|
| Session 2 **BPE** with merges in `tokenizer_hash` | WordLevel ~63-word vocab as final tokenizer |
| Natural corpus text in P1-T04 | Sentences built only from tokenizer whitelist |
| Generated evidence from demo run | Hardcoded `evidence.json` |
| OPUS from deterministic scorer + audit log | Hardcoded accept/reject lists |
| Checkpoint binds `ledger_offset` + `branch_id` | Resume without restoring ledger state |

If implementation contradicts [MENTOR.md](MENTOR.md) §3 locked decisions, fix direction before adding code.

---

## 1. North star (do not lose sight of this)

The goal is **not** frontier scale. The goal is to prove:

| Property | What graders check |
|----------|-------------------|
| **Correct** | Masks, packing, and stage logic match data type |
| **Reproducible** | Same config + tokenizer + ledger state → same batches |
| **Auditable** | Hashes, manifests, and ledgers explain every step |
| **Recoverable** | Crash resume, replay, and fork work with evidence |
| **Efficient (measured)** | Utilization and tokens/sec are computed, not guessed |

If a change does not help one of these properties, question whether it belongs in this assignment.

---

## 2. Complexity guardrails

### Do

- **Prefer explicit code** over clever abstractions. Students should read a function and understand it.
- **One module, one job.** Example: `schema.py` validates, `loader.py` reads files, `hash.py` fingerprints the tokenizer.
- **Reach for a maintained library when the code you would write is not the lesson.** Config parsing, schema validation, CLI plumbing, and table formatting are solved problems; hand-rolling them adds lines a reviewer must read and you must maintain, and teaches nothing about training data systems.
- **Use plain dataclasses + functions for the domain types** (manifests, ledger events, batches). These are small, hashed, and serialized in specific ways, so a library buys little here.
- **Write the simplest thing that passes acceptance criteria** for the current task in [TASKS.md](TASKS.md).
- **Generate evidence from real runs.** Logs, JSON, and hashes must come from executed logic.
- **Test invariants, not implementation trivia.** See SCOPE.md §11.
- **Keep public APIs small.** Export only what scripts, tests, or the next phase need.
- **Use temp file + rename** for atomic writes (shards, manifests, checkpoints).
- **Bind checkpoints to `ledger_offset` + `branch_id`.** This is non-negotiable for recovery.

### Do not

- **Do not reinvent a well-solved wheel.** Hand-rolled config validation, argument parsing, JSON schema checking, or retry logic is not "explicit", it is unreviewed duplicate code. See §7 for the test to apply.
- **Do not add infrastructure** (databases, message queues, workflow engines, web servers, distributed runners) that the assignment does not require. This is different from adding a library: infrastructure changes how the demo runs, a library does not.
- **Do not use a library for the thing being taught.** Tokenization freeze, packing, masking, routing, ledgers, and recovery are the lesson. Importing someone else's implementation of those defeats the assignment even though it would shorten the diff.
- **Do not build plugin registries, factory hierarchies, or strategy-pattern stacks** for two packing policies.
- **Do not create one-line wrapper functions** that only call another function with no added behavior.
- **Do not duplicate helpers** (`ensure_x` + `persist_x` + `load_x` + `default_x_path` for the same 10-line operation).
- **Do not export constants and internals** from package `__init__.py` unless other modules need them.
- **Do not add exception class hierarchies** unless callers catch the base type. Prefer two flat errors over three nested ones.
- **Do not hardcode evidence, hashes, OPUS decisions, or ledger rows.**
- **Do not simulate recovery** (pretend resume worked without reloading checkpoint + ledger state).
- **Do not silently drop** firewall rejections, OPUS deferrals, or blocked shards.
- **Do not scope-creep into Session 4/5 pipelines.** Use the committed toy corpus (decision D1) unless TASKS.md says otherwise.

### Rule of thumb

> If you cannot explain why a new file or class exists in one sentence tied to a TASKS.md item, leave it out.

---

## 3. Code style for this assignment

### Structure

```text
src/<subsystem>/
  __init__.py      ← small public exports only
  <logic>.py       ← main implementation
  errors.py        ← only if more than one error type is needed
```

- Empty scaffold packages from P0 are fine. Do not pre-fill them with abstract base classes.
- Scripts live in `scripts/` and should be thin entry points that call `src/`.
- Config lives in `configs/*.yaml`, loaded by `src/config/`.

### Functions and types

- Type hints on public functions and dataclasses.
- Docstrings on modules and non-obvious functions only.
- Comments explain **why**, not what the next line obviously does.
- Prefer `Path` over string paths in Python code.
- Keep deterministic behavior: fixed seeds, stable hash serialization (`sort_keys=True`, fixed separators).

### Naming

- Use course vocabulary consistently: `shard`, `ledger_offset`, `branch_id`, `tokenizer_hash`, `loss_mask_hash`, `batch_content_hash`.
- Hash IDs use the course prefix style where applicable: `tok_<12 hex chars>`.
- Content hashes use `sha256:` prefix in manifests when specified in SCOPE.md.

### Error handling

- Fail fast with clear `ValueError` / domain errors (`ConfigError`, `CorpusSchemaError`).
- Validate at boundaries: config load, corpus load, shard admission, checkpoint load.
- Do not add broad try/except that hides bugs or auto-repairs stale state without logging.

---

## 4. What must be real (grading Step 3)

These fail inspection if faked:

| Subsystem | Must be produced by |
|-----------|---------------------|
| Tokenizer hash | Fingerprint of vocab + merges + special tokens |
| Shard manifest | Shard builder + admission gate |
| OPUS decisions | Deterministic scorer + audit writer |
| Firewall blocks | Registry lookup on real batch candidates |
| Consumption ledger | Append on each committed batch |
| Learning ledger | Training step after forward/backward |
| Checkpoints | Saved model/optimizer/RNG + ledger offset |
| Resume / replay / fork | Reload state and re-derive batches |
| `evidence.json` | Collector reading generated artifacts |
| Throughput / utilization | Timings and batch stats from the run |

**Automatic fail examples:**
- Static `evidence.json` checked into git
- Hardcoded `batch_content_hash` in resume verification
- OPUS accept list as a constant in code
- Eval shard appearing in a loss-bearing batch

---

## 5. Testing expectations

### Test what matters

Align tests with SCOPE.md §11 invariants:

1. Tokenizer change invalidates shard binding
2. Loss mask sums and eval exclusion
3. `loss_mask_hash` stability
4. Ledger append-only / monotonic offsets
5. Checkpoint includes `ledger_offset` and `branch_id`
6. Resume exactness (no skip, no repeat)
7. Replay hash match
8. Fork creates new branch with logged divergence
9. Firewall blocks `never_train`
10. OPUS audit completeness
11. Always-ON floor (within toy tolerance)
12. Packing utilization recomputable

### Test what to skip

- Do not test every private helper unless it encodes a critical invariant.
- Do not test library behavior (HuggingFace tokenizers, PyTorch autograd).
- Do not duplicate the same assertion across three test methods.
- CPU-only tests are required; GPU is optional.

### Before marking a TASKS.md item `done`

- [ ] Acceptance criteria for that task are met
- [ ] Linked tests pass (if any)
- [ ] No new unused exports or dead code
- [ ] No hardcoded paths to your local machine

Run:

```powershell
.venv\Scripts\python.exe -m pytest session06/assignment/tests -v
```

```bash
.venv/bin/python -m pytest session06/assignment/tests -v
```

---

## 6. Per-phase review notes

Use this when completing each phase in [TASKS.md](TASKS.md).

### P0 — Scaffold and config

**Keep:** YAML + dataclass loader with explicit validation. Verbosity is OK here.

**Avoid:** Dynamic config plugins, env-var indirection, or loading config from multiple undocumented locations.

### P1 — Tokenizer, shards, manifests

**Keep:** Frozen tokenizer wrapper, one hash function, one sidecar JSON, admission gate checks, **BPE artifact with merges**.

**Avoid:** WordLevel as final design (see P1-T03R), multiple hash code paths, public fingerprint helpers, tokenizer mutation APIs, rebuilding shards without new manifests.

### P2 — Packing and batch builder

**Keep:** Two policies (`concat_and_chop`, `structure_preserving`) as plain functions or small classes.

**Avoid:** A generic packing framework for N policies when only two are required.

### P3 — Mixture compiler

**Keep:** Compile `curriculum.yaml` → `schedule.json` with deterministic per-step quotas.

**Avoid:** Full Session 5 10T scheduler port, live hyperparameter tuning, or opaque random sampling without seed.

### P4 — Eval firewall

**Keep:** Registry + explicit block before loss assignment + logged rejection.

**Avoid:** Silent filtering, post-hoc ledger scrubbing, or "warn only" for eval overlap.

### P5 — OPUS

**Keep:** Deterministic score from shard metadata + step; append-only audit log.

**Avoid:** ML-based rerankers, hardcoded accept/reject lists, deleting rejected candidates.

### P6–P9 — Ledger, training, recovery

**Keep:** Append-only JSONL ledgers, checkpoint payload with ledger binding, explicit crash/resume/replay/fork scripts.

**Avoid:** In-place ledger edits, resuming without restoring dataloader/ledger offset, replay that re-randomizes batch order.

### P10–P12 — Metrics, demo, ship

**Keep:** One command (`scripts/run_demo.py`), generated `submission_artifacts/`, evidence collector that reads outputs.

**Avoid:** Manual steps in README, checked-in artifacts under `submission_artifacts/`, evidence markdown written by hand.

---

## 7. File and dependency discipline

### Current dependencies

- Python stdlib
- PyTorch (tiny model)
- Hugging Face `tokenizers` / minimal transformers usage
- PyYAML
- pytest

### Choosing a dependency

Answer all four before adding one:

1. **Is the code it replaces the lesson?** If yes, write it yourself. Packing, masking, ledgers, and recovery are the assignment.
2. **Does it remove real code?** Roughly 50+ lines of plumbing, not 5. A one-import convenience is not worth a dependency.
3. **Is it a library or is it infrastructure?** A library the demo imports is cheap. A service the demo must have running is not; that needs a conversation.
4. **Can a grader still run one command on a clean checkout?** If the dependency complicates that, it fails regardless of the other three.

| Situation | Verdict |
|-----------|---------|
| Schema validation for `demo.yaml` / `curriculum.yaml` | **Use a library.** Session 6 now uses Pydantic in `config/schemas.py` with a ~95-line loader |
| CLI argument parsing | `argparse` (stdlib) is already the wheel. Keep scripts consistent; do not add a CLI framework for three flags |
| Progress output, tables, colour | Fine to use, but never let presentation code obscure what the demo actually verified |
| Manifest / ledger / batch dataclasses | **Write them.** They are small, content-hashed, and serialized in exact orders; a library adds indirection without removing lines |
| Tokenizer, packing, routing, masking, recovery | **Write them.** This is the material being assessed |

### Ask before adding

- Database servers, message queues, workflow engines
- Web servers (unless building an optional demo UI later)
- Large ML stacks beyond what the tiny trainer needs

### Git hygiene

- Commit source, configs, toy corpus, tokenizer artifacts, tests
- Gitignore `submission_artifacts/`, local checkpoints, `__pycache__/`
- Do not commit `.env`, secrets, or machine-local paths

---

## 8. Documentation discipline

| Doc | Role |
|-----|------|
| `SCOPE.md` | Architecture baseline; update only for scope decisions |
| `TASKS.md` | Task status; update when tasks complete |
| `REVIEW.md` | This guide; update when team learns a new anti-pattern |
| `README.md` | Submission-facing: how to run, architecture summary, design decisions |

**README should be short.** Point to `SCOPE.md` for depth. Include the one demo command and test command.

**Do not** add extra markdown files unless a task explicitly requires them.

---

## 9. Pre-commit checklist

Before every commit:

- [ ] Diff is scoped to the current task (no drive-by refactors)
- [ ] Tests pass
- [ ] No hardcoded evidence or grading outputs
- [ ] No new abstraction without a TASKS.md justification
- [ ] Public exports in `__init__.py` are still minimal
- [ ] Error messages name the failing field/path
- [ ] [TASKS.md](TASKS.md) status updated if a task is finished

Before calling the assignment complete (see TASKS.md submission gate):

- [ ] One command regenerates `submission_artifacts/` from scratch
- [ ] `evidence.json` passes with real evidence paths
- [ ] Crash → resume → replay → fork demonstrations verified
- [ ] All invariant tests pass
- [ ] README complete

---

## 10. Good vs bad patterns (from work so far)

### Good

```python
# One hash function, one persist function, private fingerprint helper.
def compute_tokenizer_hash_from_artifact(artifact_path: Path) -> str: ...
def persist_tokenizer_hash(artifact_path: Path, hash_path: Path | None = None) -> str: ...
```

```python
# Explicit validation at corpus load time.
validate_document_record(record, provenance_ids=provenance_ids)
```

### Bad

```python
# Hand-rolled schema validation: 654 lines in config/loader.py doing what a
# validation library does in ~80, for two YAML files that never change shape.
_as_int(_require_key(training_raw, "seq_len", label="demo.training"), label="demo.training.seq_len")
```

```python
# Too many public entry points for the same concern.
ensure_tokenizer_hash_artifact(...)
persist_tokenizer_hash(...)
load_persisted_tokenizer_hash(...)
default_tokenizer_hash_path(...)
extract_tokenizer_fingerprint(...)  # exported but only used internally
```

```python
# Hardcoded evidence (fails Step 3).
EVIDENCE = {"crash_resume_no_skip_repeat": {"passed": True, "evidence_path": "..."}}
```

```python
# Over-abstracted packing for two policies.
class PackingStrategy(ABC):
    @abstractmethod
    def pack(self, docs): ...

class PackingStrategyRegistry: ...
```

```python
# Silent repair of stale hash files without logging.
except Exception:
    persisted = ""
```

---

## 11. When complexity is justified

Add complexity only when the assignment demands it:

| Situation | Justified complexity |
|-----------|---------------------|
| Two packing policies with different mask rules | Separate functions or two small classes |
| Admission gate with multiple required manifest fields | One validation function with explicit checks |
| Recovery modes (resume / replay / fork) | Separate functions, shared checkpoint loader |
| Evidence collector | One module that runs checks against generated files |
| Cross-field config validation | Helper functions in loader (`_validate_recovery_config`) |

---

## 12. Review questions (ask on every PR)

1. Which **TASKS.md** item does this change complete?
2. Does it align with **[MENTOR.md](MENTOR.md) locked decisions** (especially D7 BPE)?
3. Can a student **read and modify** this in one sitting?
4. Is there **reproducible evidence** for the behavior claimed?
5. Did we add a **test for an invariant**, not for coverage percentage?
6. Can anything be **deleted** without breaking requirements?
7. Are we building **medium scale** with **production-shaped contracts**?

If questions 2, 4, or 7 fail, revise before merge.

---

## 13. AI assistant protocol (summary)

Full rules: [MENTOR.md](MENTOR.md) §5.

- **Lead** with a recommended default; do not ask the student to choose when docs already decide.
- **Ask** only for genuinely open decisions or preference forks; check the TASKS.md decision table first.
- **Flag** WordLevel, hardcoded evidence, or missing ledger binding before the student catches it.
- **Do not** trade realism for passing tests on a toy shortcut.
- **Do not hand-roll a solved problem** to avoid a dependency (§7), and do not import a library for the subsystem being taught.

---

*Last updated: 2026-08-07 · Dependency policy changed: library-first for solved problems, hand-written for the subsystems being taught (§2, §7, §10).*
