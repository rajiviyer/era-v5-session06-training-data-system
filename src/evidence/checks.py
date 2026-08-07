"""Requirement checks run against generated artifacts (P11-T04).

Every check in this module reads `submission_artifacts/` and decides from what it finds.
None of them import a result from the code that produced the artifact, and none read a
top-level `passed` flag as the answer on its own:

- Shard hashes are recomputed from the shard bytes on disk.
- The tokenizer hash is recomputed from the committed BPE artifact.
- Resume is re-verified from the ledger, using only the *parameters* the report records.
- Replay and packing numbers are recomputed from the per-row data and cross-checked
  against the consumption ledger, which was written by a different subsystem.

Where a report's own verdict is used (resume, replay, fork), it is compared against a
recomputation rather than trusted, so a report claiming a pass it cannot support fails
here. That is the whole point of grading Step 3: evidence has to be produced by real
logic (SCOPE.md §10).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from firewall.registry import REGISTRY_FILENAME as EVAL_REGISTRY_FILENAME, load_eval_registry
from firewall.types import EvalRegistry
from ledger import (
    LEARNING_LEDGER_FILENAME,
    LEDGER_FILENAME,
    ConsumptionLedgerEvent,
    LearningLedgerEvent,
    aggregate_by_shard,
    get_events_for_global_step,
    load_consumption_ledger,
    load_learning_ledger,
    verify_learning_links,
)
from opus.audit import load_opus_audit
from opus.types import AUDIT_FILENAME, OpusAuditRecord
from recovery.fork import FORK_LOG_FILENAME
from recovery.verify import verify_resume
from runlog import RUN_LOG_FILENAME, RunLogEvent, events_of_type, load_run_log
from shards.format import shard_id_from_content_hash
from shards.registry import REGISTRY_FILENAME as SHARD_REGISTRY_FILENAME
from tokenizer.bpe import default_tokenizer_path
from tokenizer.hash import compute_tokenizer_hash_from_artifact
from tokenizer.manifest import MANIFEST_FILENAME as TOKENIZER_MANIFEST_FILENAME

from .types import Check, EvidenceError, RequirementResult

# Rates are reported rounded, so an exact equality test would fail on the rounding
# rather than on the arithmetic.
_RATE_TOLERANCE = 0.5
_RATIO_TOLERANCE = 1e-5


@dataclass(frozen=True)
class LoadedArtifacts:
    """Everything the checks read, loaded once from `submission_artifacts/`."""

    root: Path
    assignment_root: Path
    consumption: tuple[ConsumptionLedgerEvent, ...]
    learning: tuple[LearningLedgerEvent, ...]
    opus_audit: tuple[OpusAuditRecord, ...]
    run_log: tuple[RunLogEvent, ...]
    shard_manifests: tuple[dict[str, Any], ...]
    shard_registry: dict[str, Any]
    tokenizer_manifest: dict[str, Any]
    eval_registry: EvalRegistry
    schedule: dict[str, Any]
    packing: dict[str, Any]
    throughput: dict[str, Any]
    resume_report: dict[str, Any]
    replay_report: dict[str, Any]
    fork_report: dict[str, Any]
    fork_events: tuple[dict[str, Any], ...]

    def rel(self, *parts: str) -> str:
        """Artifact path relative to `submission_artifacts/`, for evidence_path."""
        return "/".join(parts)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)


def load_artifacts(artifacts_dir: Path, assignment_root: Path) -> LoadedArtifacts:
    """Load every artifact the checks need, failing loudly on a missing one."""
    root = Path(artifacts_dir).resolve()
    if not root.is_dir():
        raise EvidenceError(f"artifacts directory not found: {root}")

    manifests_dir = root / "manifests"
    shard_manifests = tuple(
        _read_json(path)
        for path in sorted(manifests_dir.glob("shard_*.json"))
        if path.name != SHARD_REGISTRY_FILENAME
    )

    return LoadedArtifacts(
        root=root,
        assignment_root=Path(assignment_root).resolve(),
        consumption=load_consumption_ledger(root / "ledgers" / LEDGER_FILENAME),
        learning=tuple(load_learning_ledger(root / "ledgers" / LEARNING_LEDGER_FILENAME)),
        opus_audit=load_opus_audit(root / "ledgers" / AUDIT_FILENAME),
        run_log=load_run_log(root / RUN_LOG_FILENAME),
        shard_manifests=shard_manifests,
        shard_registry=_read_json(manifests_dir / SHARD_REGISTRY_FILENAME),
        tokenizer_manifest=_read_json(manifests_dir / TOKENIZER_MANIFEST_FILENAME),
        eval_registry=load_eval_registry(root / EVAL_REGISTRY_FILENAME),
        schedule=_read_json(root / "schedule.json"),
        packing=_read_json(root / "reports" / "packing_utilization.json"),
        throughput=_read_json(root / "reports" / "throughput.json"),
        resume_report=_read_json(root / "reports" / "resume_verification.json"),
        replay_report=_read_json(root / "reports" / "replay_verification.json"),
        fork_report=_read_json(root / "reports" / "fork_verification.json"),
        fork_events=_read_jsonl(root / "ledgers" / FORK_LOG_FILENAME),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"required artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise EvidenceError(f"required artifact missing: {path}")
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _result(key: str, checks: Sequence[Check], paths: Sequence[str]) -> RequirementResult:
    return RequirementResult(key=key, checks=tuple(checks), evidence_paths=tuple(paths))


# --------------------------------------------------------------------------------------
# 1. Immutable tokenized shards with manifests
# --------------------------------------------------------------------------------------


def check_immutable_shards(art: LoadedArtifacts) -> RequirementResult:
    """Rehash every shard file and hold the manifests against the bytes on disk."""
    hashes_on_disk = {
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(): path.name
        for path in sorted(art.path("shards").glob("*"))
        if path.is_file()
    }

    rehash_failures = [
        manifest["shard_id"]
        for manifest in art.shard_manifests
        if manifest["content_hash"] not in hashes_on_disk
    ]
    id_failures = [
        manifest["shard_id"]
        for manifest in art.shard_manifests
        if shard_id_from_content_hash(manifest["content_hash"]) != manifest["shard_id"]
    ]

    admitted = set(art.shard_registry.get("admitted_shard_ids", []))
    admitted_manifests = {
        manifest["shard_id"]
        for manifest in art.shard_manifests
        if manifest.get("admission") == "admitted"
    }
    blocked = [
        manifest["shard_id"]
        for manifest in art.shard_manifests
        if manifest.get("admission") != "admitted"
    ]

    return _result(
        "immutable_shards_with_manifests",
        [
            Check(
                "shard files rehash to their manifest content_hash",
                not rehash_failures and bool(art.shard_manifests),
                f"{len(art.shard_manifests)} manifests, {len(hashes_on_disk)} shard files; "
                f"mismatched: {rehash_failures or 'none'}",
            ),
            Check(
                "shard_id is derived from the content hash",
                not id_failures,
                f"mismatched: {id_failures or 'none'}",
            ),
            Check(
                "the registry admits exactly the manifests the gate admitted",
                admitted == admitted_manifests,
                f"registry {len(admitted)} admitted, manifests {len(admitted_manifests)} "
                f"admitted, {len(blocked)} blocked ({blocked or 'none'})",
            ),
            Check(
                "the admission gate blocked at least one shard",
                bool(blocked),
                f"blocked: {blocked or 'none'}",
            ),
        ],
        [art.rel("manifests", SHARD_REGISTRY_FILENAME), art.rel("manifests"), art.rel("shards")],
    )


# --------------------------------------------------------------------------------------
# 2. Frozen tokenizer and content hashes
# --------------------------------------------------------------------------------------


def check_frozen_tokenizer(art: LoadedArtifacts) -> RequirementResult:
    """Recompute the tokenizer hash from the committed BPE artifact.

    The path is resolved directly rather than through `FrozenTokenizer.load_default`,
    which loads the whole model and rewrites the hash and manifest sidecars under
    `data/tokenizer/`. Collecting evidence must not write to the sources it is grading.
    """
    manifest = art.tokenizer_manifest
    artifact_path = default_tokenizer_path(art.assignment_root)
    recomputed = compute_tokenizer_hash_from_artifact(artifact_path)
    declared = manifest["tokenizer_hash"]

    shard_hashes = {manifest_["tokenizer_hash"] for manifest_ in art.shard_manifests}
    ledger_hashes = {row.tokenizer_hash for row in art.consumption}

    return _result(
        "frozen_tokenizer_hashes",
        [
            Check(
                "manifest tokenizer_hash recomputes from the tokenizer artifact",
                recomputed == declared,
                f"recomputed {recomputed} vs manifest {declared} ({artifact_path.name})",
            ),
            Check(
                "the frozen tokenizer is BPE with a non-empty merge table",
                manifest.get("model_type") == "BPE" and int(manifest.get("merge_count", 0)) > 0,
                f"model_type {manifest.get('model_type')}, "
                f"{manifest.get('merge_count')} merges, vocab {manifest.get('vocab_size')}",
            ),
            Check(
                "every shard was sealed with that tokenizer hash",
                shard_hashes == {declared},
                f"shard manifest hashes: {sorted(shard_hashes)}",
            ),
            Check(
                "every consumed batch records that tokenizer hash",
                ledger_hashes == {declared},
                f"ledger hashes: {sorted(ledger_hashes)}",
            ),
        ],
        [art.rel("manifests", TOKENIZER_MANIFEST_FILENAME), art.rel("manifests")],
    )


# --------------------------------------------------------------------------------------
# 3. Packing policies
# --------------------------------------------------------------------------------------


def check_packing_policies(art: LoadedArtifacts) -> RequirementResult:
    """Both policies must have been exercised, with different utilization."""
    by_policy = art.packing["by_packing_policy"]
    rows = art.packing["batches"]
    policies_in_rows = {row["packing_policy"] for row in rows}

    utilizations = {name: stats["utilization"] for name, stats in by_policy.items()}
    return _result(
        "packing_policies",
        [
            Check(
                "both packing policies ran",
                {"concat_and_chop", "structure_preserving"} <= set(by_policy),
                f"policies: {sorted(by_policy)}",
            ),
            Check(
                "the aggregate covers exactly the policies the batches used",
                policies_in_rows == set(by_policy),
                f"batch rows used {sorted(policies_in_rows)}",
            ),
            Check(
                "the policies pack to different utilization",
                len(set(round(value, 4) for value in utilizations.values())) > 1,
                ", ".join(f"{name} {value:.3f}" for name, value in sorted(utilizations.items())),
            ),
        ],
        [art.rel("reports", "packing_utilization.json")],
    )


# --------------------------------------------------------------------------------------
# 4. Correct loss, attention, and position IDs
# --------------------------------------------------------------------------------------


def check_correct_masks(art: LoadedArtifacts) -> RequirementResult:
    """Mask policies on every consumed batch, and token counts that respect capacity."""
    attention = {row.attention_policy for row in art.consumption}
    positions = {row.position_policy for row in art.consumption}
    unhashed = [row.microbatch_id for row in art.consumption if not row.loss_mask_hash.startswith("sha256:")]

    rows = art.packing["batches"]
    over_capacity = [
        row["microbatch_id"]
        for row in rows
        if not 0 < row["loss_bearing_tokens"] <= row["useful_tokens"] <= row["capacity"]
    ]
    masked_out = [
        row for row in rows if row["loss_bearing_tokens"] < row["useful_tokens"]
    ]

    return _result(
        "correct_masks",
        [
            Check(
                "every consumed batch used a causal attention mask",
                attention == {"causal"},
                f"attention policies: {sorted(attention)}",
            ),
            Check(
                "every consumed batch recorded its position ID policy",
                bool(positions) and all(value for value in positions),
                f"position policies: {sorted(positions)}",
            ),
            Check(
                "every consumed batch carries a loss mask hash",
                not unhashed,
                f"{len(art.consumption)} rows; unhashed: {unhashed or 'none'}",
            ),
            Check(
                "0 < loss-bearing tokens <= non-pad tokens <= capacity for every batch",
                not over_capacity and bool(rows),
                f"{len(rows)} batches recounted; violations: {over_capacity or 'none'}",
            ),
            Check(
                "some tokens were seen but excluded from the loss",
                bool(masked_out),
                f"{len(masked_out)} of {len(rows)} batches mask part of what they read "
                f"(loss-bearing fraction {art.packing['loss_bearing_fraction']:.3f})",
            ),
        ],
        [art.rel("ledgers", LEDGER_FILENAME), art.rel("reports", "packing_utilization.json")],
    )


# --------------------------------------------------------------------------------------
# 5. Curriculum stages, lane weights, protected floors
# --------------------------------------------------------------------------------------


def check_curriculum_and_floors(art: LoadedArtifacts) -> RequirementResult:
    """Stages, the Always-ON floor per step, and protected-lane overrides."""
    schedule = art.schedule
    steps = schedule["steps"]
    floor = schedule["always_on_fraction"]
    protected_lanes = set(schedule["protected_floor_lanes"])
    phase_names = [phase["name"] for phase in schedule["phase_boundaries"]]

    below_floor = [
        step["step"] for step in steps if step["always_on_fraction"] + _RATIO_TOLERANCE < floor
    ]
    stages_consumed = {row.curriculum_stage for row in art.consumption}
    transitions = events_of_type(art.run_log, "stage_transition")
    transition_stages = {event.fields["to_stage"] for event in transitions}

    overrides = [
        record
        for record in art.opus_audit
        if record.protected_floor_override and record.capability_lane in protected_lanes
    ]

    return _result(
        "curriculum_and_floors",
        [
            Check(
                "the schedule compiles at least three curriculum stages",
                len(phase_names) >= 3,
                f"stages: {phase_names}",
            ),
            Check(
                "every step meets the Always-ON floor",
                not below_floor,
                f"floor {floor}; {len(steps)} steps; below floor: {below_floor or 'none'}",
            ),
            Check(
                "the run consumed batches under more than one stage",
                len(stages_consumed) > 1,
                f"stages in the ledger: {sorted(stages_consumed)}",
            ),
            Check(
                "stage transitions were logged and name compiled stages",
                bool(transitions) and transition_stages <= set(phase_names),
                f"{len(transitions)} transitions into {sorted(transition_stages)}",
            ),
            Check(
                "protected-floor lanes bypassed OPUS at least once",
                bool(overrides),
                f"{len(overrides)} overrides on lanes "
                f"{sorted({record.capability_lane for record in overrides})}; "
                f"protected lanes {sorted(protected_lanes)}",
            ),
        ],
        [art.rel("schedule.json"), art.rel("ledgers", AUDIT_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 6. Evaluation and validation firewall
# --------------------------------------------------------------------------------------


def check_eval_firewall(art: LoadedArtifacts) -> RequirementResult:
    """No never-train document may appear in a loss-bearing consumed batch."""
    never_train = {
        entry.document_id for entry in art.eval_registry.entries if entry.never_train
    }
    consumed_samples = {
        sample_id for row in art.consumption for sample_id in row.packed_sample_ids
    }
    leaked = sorted(never_train & consumed_samples)

    blocks = events_of_type(art.run_log, "firewall_block")
    blocked_candidates = {event.fields["candidate_id"] for event in blocks}
    committed_candidates = {row.candidate_id for row in art.consumption}
    blocked_but_committed = sorted(blocked_candidates & committed_candidates)
    block_reasons = sorted(
        {reason for event in blocks for reason in event.fields.get("reasons", [])}
    )

    return _result(
        "eval_firewall",
        [
            Check(
                "the registry marks at least one document never_train",
                bool(never_train),
                f"never_train documents: {sorted(never_train)}",
            ),
            Check(
                "no never-train document reached the consumption ledger",
                not leaked,
                f"{len(consumed_samples)} distinct samples consumed; leaked: {leaked or 'none'}",
            ),
            Check(
                "the firewall blocked at least one candidate during the run",
                bool(blocks),
                f"{len(blocks)} blocks, reasons {block_reasons}",
            ),
            Check(
                "no blocked candidate was later committed",
                not blocked_but_committed,
                f"blocked candidates also committed: {blocked_but_committed or 'none'}",
            ),
        ],
        [art.rel(EVAL_REGISTRY_FILENAME), art.rel(RUN_LOG_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 7. OPUS audit trail
# --------------------------------------------------------------------------------------


def check_opus_audit(art: LoadedArtifacts) -> RequirementResult:
    """Every committed batch traces to an audit record, and all four verdicts occurred."""
    decisions = {record.decision for record in art.opus_audit}
    by_id = {record.opus_decision_id: record for record in art.opus_audit}

    missing = [
        row.microbatch_id for row in art.consumption if row.opus_decision_id not in by_id
    ]
    not_accepting = [
        row.microbatch_id
        for row in art.consumption
        if row.opus_decision_id in by_id
        and by_id[row.opus_decision_id].decision not in ("accepted", "protected_override")
    ]
    retained = [
        record for record in art.opus_audit if record.decision in ("rejected", "deferred")
    ]

    return _result(
        "opus_audit_trail",
        [
            Check(
                "all four OPUS decision types appear",
                decisions == {"accepted", "rejected", "deferred", "protected_override"},
                f"decisions: {sorted(decisions)}",
            ),
            Check(
                "every committed batch has a matching audit record",
                not missing,
                f"{len(art.consumption)} committed batches; unmatched: {missing or 'none'}",
            ),
            Check(
                "a committed batch's audit record says it was accepted",
                not not_accepting,
                f"committed under a non-accepting decision: {not_accepting or 'none'}",
            ),
            Check(
                "rejected and deferred candidates stayed queryable",
                bool(retained),
                f"{len(retained)} of {len(art.opus_audit)} audit records are "
                "rejections or deferrals, still on file",
            ),
        ],
        [art.rel("ledgers", AUDIT_FILENAME), art.rel("ledgers", LEDGER_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 8. Consumption ledger
# --------------------------------------------------------------------------------------


def check_consumption_ledger(art: LoadedArtifacts) -> RequirementResult:
    """Append-only offsets, and any step reconstructible from the file alone."""
    by_attempt: dict[int, list[int]] = {}
    for row in art.consumption:
        by_attempt.setdefault(row.attempt, []).append(row.ledger_offset)
    non_contiguous = [
        attempt
        for attempt, offsets in by_attempt.items()
        if offsets != list(range(offsets[0], offsets[0] + len(offsets)))
    ]

    steps = sorted({row.global_step for row in art.consumption})
    unreconstructible: list[int] = []
    for step in steps:
        try:
            rebuilt = get_events_for_global_step(art.consumption, step)
        except Exception:  # noqa: BLE001 - any failure means the step is not reconstructible
            unreconstructible.append(step)
            continue
        if not rebuilt or any(row.global_step != step for row in rebuilt):
            unreconstructible.append(step)

    attempts = sorted(by_attempt)
    return _result(
        "consumption_ledger",
        [
            Check(
                "the ledger loads under append-only ordering rules",
                bool(art.consumption),
                f"{len(art.consumption)} rows across attempts {attempts}; "
                "load_consumption_ledger rejects a decreasing attempt, a non-incrementing "
                "offset, or an attempt starting past the previous tail",
            ),
            Check(
                "offsets increment by one within every attempt",
                not non_contiguous,
                "offsets per attempt: "
                + ", ".join(
                    f"attempt {attempt} {offsets[0]}..{offsets[-1]}"
                    for attempt, offsets in sorted(by_attempt.items())
                ),
            ),
            Check(
                "every step in the run reconstructs from the ledger",
                not unreconstructible,
                f"{len(steps)} steps reconstructed; failed: {unreconstructible or 'none'}",
            ),
            Check(
                "a crashed attempt's rows were retained, not overwritten",
                len(attempts) > 1,
                f"attempts on file: {attempts}",
            ),
        ],
        [art.rel("ledgers", LEDGER_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 9. Learning ledger
# --------------------------------------------------------------------------------------


def check_learning_ledger(art: LoadedArtifacts) -> RequirementResult:
    """Re-join the two ledgers and re-derive perplexity from the recorded loss."""
    links = verify_learning_links(art.learning, art.consumption)
    aggregates = aggregate_by_shard(art.learning)

    repeated = [item for item in aggregates if item.exposure_count > 1]
    bad_perplexity = [
        item.shard_id
        for item in aggregates
        if not math.isclose(item.perplexity, math.exp(item.mean_loss), rel_tol=1e-4)
    ]

    return _result(
        "learning_ledger",
        [
            Check(
                "every learning row joins to a committed batch",
                links.linked,
                f"{links.learning_rows} learning rows against "
                f"{links.committed_batches} committed batches; "
                f"{len(links.orphan_offsets)} orphans, "
                f"{len(links.unreported_offsets)} unreported, "
                f"{len(links.mismatches)} mismatches",
            ),
            Check(
                "at least one shard shows a loss trend across exposures",
                bool(repeated),
                "; ".join(
                    f"{item.shard_id} {item.exposure_count} exposures, "
                    f"loss_delta {item.loss_delta:+.3f}"
                    for item in sorted(repeated, key=lambda i: i.exposure_count, reverse=True)[:3]
                )
                or "no shard was exposed more than once",
            ),
            Check(
                "perplexity re-derives from the recorded loss",
                not bad_perplexity and bool(aggregates),
                f"{len(aggregates)} shard aggregates; mismatched: {bad_perplexity or 'none'}",
            ),
        ],
        [art.rel("ledgers", LEARNING_LEDGER_FILENAME), art.rel("ledgers", LEDGER_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 10. Checkpoints tied to ledger offsets
# --------------------------------------------------------------------------------------


def check_checkpoint_binding(art: LoadedArtifacts) -> RequirementResult:
    """Every checkpoint must bind a ledger offset, a branch, and real tensor files."""
    directories = sorted(art.path("checkpoints").glob("ckpt-*"))
    payloads = [
        (directory, _read_json(directory / "checkpoint.json")) for directory in directories
    ]

    unbound = [
        directory.name
        for directory, payload in payloads
        if payload.get("ledger_offset") is None or not payload.get("branch_id")
    ]
    missing_tensors = [
        directory.name
        for directory, payload in payloads
        if not payload.get("tensor_files")
        or not all((directory / name).is_file() for name in payload["tensor_files"])
    ]
    tail = max((row.ledger_offset for row in art.consumption), default=-1)
    ahead_of_ledger = [
        directory.name
        for directory, payload in payloads
        if payload["ledger_offset"] > tail
    ]

    return _result(
        "checkpoint_ledger_binding",
        [
            Check(
                "checkpoints exist",
                bool(payloads),
                f"{len(payloads)} checkpoints: {[d.name for d, _ in payloads]}",
            ),
            Check(
                "every checkpoint records ledger_offset and branch_id",
                not unbound,
                f"unbound: {unbound or 'none'}",
            ),
            Check(
                "every checkpoint's tensor sidecars are on disk",
                not missing_tensors,
                f"missing model/optimizer state: {missing_tensors or 'none'}",
            ),
            Check(
                "no checkpoint claims an offset past the ledger tail",
                not ahead_of_ledger,
                f"ledger tail {tail}; ahead of it: {ahead_of_ledger or 'none'}",
            ),
        ],
        [
            *(
                art.rel("checkpoints", directory.name, "checkpoint.json")
                for directory, _ in payloads
            ),
            art.rel("ledgers", LEDGER_FILENAME),
        ],
    )


# --------------------------------------------------------------------------------------
# 11. Crash recovery
# --------------------------------------------------------------------------------------


def check_crash_resume(art: LoadedArtifacts) -> RequirementResult:
    """Re-verify resume from the ledger, taking only the parameters from the report.

    The report's own `passed` is compared against this recomputation rather than
    accepted, so a report that claims a pass its ledger cannot support fails here.
    """
    report = art.resume_report
    recomputed = verify_resume(
        art.path("ledgers", LEDGER_FILENAME),
        resume_ledger_offset=report["resume_ledger_offset"],
        prior_attempt=report["prior_attempt"],
        resumed_attempt=report["resumed_attempt"],
        resumed_from_checkpoint_step=report["resumed_from_checkpoint_step"],
    )
    crashes = events_of_type(art.run_log, "simulated_crash")
    resumes = events_of_type(art.run_log, "resume_initiated")

    return _result(
        "crash_resume_no_skip_repeat",
        [
            Check(
                "the run crashed and then resumed",
                bool(crashes) and bool(resumes) and crashes[0].seq < resumes[0].seq,
                f"crash at step {crashes[0].fields['global_step'] if crashes else 'n/a'}, "
                f"resume from step "
                f"{resumes[0].fields['resume_from_step'] if resumes else 'n/a'}",
            ),
            Check(
                "re-verifying the ledger reproduces the report's verdict",
                recomputed.passed == report["passed"] and recomputed.passed,
                f"recomputed passed={recomputed.passed}, report passed={report['passed']}",
            ),
            Check(
                "no batch was skipped and none was repeated",
                not recomputed.skipped_batches and not recomputed.repeated_batches,
                f"{len(recomputed.comparisons)} batches compared, "
                f"{len(recomputed.skipped_batches)} skipped, "
                f"{len(recomputed.repeated_batches)} repeated",
            ),
            Check(
                "post-resume batch hashes equal the pre-crash record",
                bool(recomputed.comparisons) and not recomputed.mismatched,
                f"{len(recomputed.comparisons) - len(recomputed.mismatched)} of "
                f"{len(recomputed.comparisons)} matched on content hash, mask hash, "
                "sample IDs, and token spans",
            ),
        ],
        [art.rel("reports", "resume_verification.json"), art.rel("ledgers", LEDGER_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 12. Replay
# --------------------------------------------------------------------------------------


def check_replay_hash_match(art: LoadedArtifacts) -> RequirementResult:
    """Hold every replayed comparison against the consumption ledger's own row."""
    report = art.replay_report
    comparisons = report["comparisons"]
    ledger_by_key = {(row.attempt, row.ledger_offset): row for row in art.consumption}

    hash_mismatches = [
        row["microbatch_id"]
        for row in comparisons
        if row["recorded_batch_content_hash"] != row["recomputed_batch_content_hash"]
        or row["recorded_loss_mask_hash"] != row["recomputed_loss_mask_hash"]
    ]
    plan_mismatches = [
        row["microbatch_id"]
        for row in comparisons
        if not row["planned_sample_ids_match"] or not row["token_spans_match"]
    ]
    ledger_disagreements = [
        row["microbatch_id"]
        for row in comparisons
        if (row["attempt"], row["ledger_offset"]) not in ledger_by_key
        or ledger_by_key[(row["attempt"], row["ledger_offset"])].batch_content_hash
        != row["recorded_batch_content_hash"]
    ]

    return _result(
        "replay_hash_match",
        [
            Check(
                "a historical range was replayed",
                bool(comparisons) and bool(events_of_type(art.run_log, "replay_initiated")),
                f"steps {report['start_step']}..{report['end_step']}, "
                f"{len(comparisons)} batches replayed",
            ),
            Check(
                "every rebuilt batch hashes to what was recorded",
                not hash_mismatches,
                f"content and loss-mask hash mismatches: {hash_mismatches or 'none'}",
            ),
            Check(
                "the planner re-drew the same samples and spans",
                not plan_mismatches,
                f"planner or span mismatches: {plan_mismatches or 'none'}",
            ),
            Check(
                "the replayed hashes match the consumption ledger row by row",
                not ledger_disagreements,
                f"joined on (attempt, ledger_offset); disagreements: "
                f"{ledger_disagreements or 'none'}",
            ),
        ],
        [art.rel("reports", "replay_verification.json"), art.rel("ledgers", LEDGER_FILENAME)],
    )


# --------------------------------------------------------------------------------------
# 13. Fork
# --------------------------------------------------------------------------------------


def check_fork_new_branch(art: LoadedArtifacts) -> RequirementResult:
    """A fork must be a separate lineage on disk, not a relabelled continuation."""
    report = art.fork_report
    child_branch = report["child_branch_id"]
    parent_branch = report["parent_branch_id"]

    child_ledger_path = art.path("branches", child_branch, "ledgers", LEDGER_FILENAME)
    child_rows = (
        load_consumption_ledger(child_ledger_path) if child_ledger_path.is_file() else ()
    )
    child_branches = {row.branch_id for row in child_rows}

    linked = [
        event
        for event in art.fork_events
        if event.get("child_branch_id") == child_branch
        and event.get("parent_branch_id") == parent_branch
    ]

    return _result(
        "fork_new_branch",
        [
            Check(
                "the fork was given a new branch_id",
                bool(child_branch) and child_branch != parent_branch,
                f"parent {parent_branch}, child {child_branch}",
            ),
            Check(
                "the parent's fork log records the parent branch and fork offset",
                bool(linked),
                f"{len(art.fork_events)} fork events; linked: "
                + (
                    f"from ckpt step {linked[0]['forked_from_step']} at parent offset "
                    f"{linked[0]['parent_ledger_offset']}"
                    if linked
                    else "none"
                ),
            ),
            Check(
                "the child branch has its own ledger, written under its own branch_id",
                bool(child_rows) and child_branches == {child_branch},
                f"{len(child_rows)} rows at "
                f"branches/{child_branch}/ledgers/{LEDGER_FILENAME}; "
                f"branch_ids {sorted(child_branches)}",
            ),
            Check(
                "the forked stream diverges from the parent after the fork point",
                report["divergence_step"] is not None and bool(report["diverged_steps"]),
                f"diverges at step {report['divergence_step']} across "
                f"{len(report['diverged_steps'])} of {len(report['compared_steps'])} "
                "compared steps",
            ),
        ],
        [
            art.rel("reports", "fork_verification.json"),
            art.rel("ledgers", FORK_LOG_FILENAME),
            art.rel("branches", child_branch, "ledgers", LEDGER_FILENAME),
        ],
    )


# --------------------------------------------------------------------------------------
# 14. Packing utilization and throughput
# --------------------------------------------------------------------------------------


def check_packing_and_throughput(art: LoadedArtifacts) -> RequirementResult:
    """Redo the arithmetic the two reports claim, from their own per-row data."""
    packing = art.packing
    rows = packing["batches"]
    total_useful = sum(row["useful_tokens"] for row in rows)
    total_capacity = sum(row["capacity"] for row in rows)
    recomputed_utilization = total_useful / total_capacity if total_capacity else 0.0

    throughput = art.throughput
    timings = _read_jsonl(art.path("reports", "step_timings.jsonl"))
    timed_seconds = sum(
        row["wall_seconds"]
        for row in timings
        if (row["attempt"], row["global_step"])
        in {(step["attempt"], step["global_step"]) for step in throughput["steps"]}
    )
    recomputed_rate = (
        throughput["total_loss_bearing_tokens"] / throughput["total_wall_seconds"]
        if throughput["total_wall_seconds"]
        else 0.0
    )

    return _result(
        "packing_and_throughput",
        [
            Check(
                "run utilization recomputes from the per-batch rows",
                math.isclose(
                    recomputed_utilization, packing["utilization"], rel_tol=_RATIO_TOLERANCE
                ),
                f"{total_useful}/{total_capacity} = {recomputed_utilization:.6f} vs "
                f"reported {packing['utilization']:.6f} ({packing['formula']})",
            ),
            Check(
                "the reported wall time is the sum of the recorded step timings",
                math.isclose(timed_seconds, throughput["total_wall_seconds"], rel_tol=1e-4),
                f"{timed_seconds:.4f}s from {len(timings)} timing rows vs reported "
                f"{throughput['total_wall_seconds']:.4f}s",
            ),
            Check(
                "loss-bearing tokens/sec recomputes from tokens and seconds",
                abs(recomputed_rate - throughput["loss_bearing_tokens_per_second"])
                <= _RATE_TOLERANCE,
                f"{throughput['total_loss_bearing_tokens']} tokens / "
                f"{throughput['total_wall_seconds']:.2f}s = {recomputed_rate:.1f} vs "
                f"reported {throughput['loss_bearing_tokens_per_second']:.1f}",
            ),
            Check(
                "no step was measured without a timing, and useful <= raw throughput",
                not throughput["steps_without_timings"]
                and throughput["loss_bearing_tokens_per_second"]
                <= throughput["raw_tokens_per_second"],
                f"{throughput['steps_measured']} steps measured, "
                f"{len(throughput['steps_without_timings'])} untimed; "
                f"{throughput['loss_bearing_tokens_per_second']:.0f} useful vs "
                f"{throughput['raw_tokens_per_second']:.0f} raw tokens/s",
            ),
        ],
        [
            art.rel("reports", "packing_utilization.json"),
            art.rel("reports", "throughput.json"),
            art.rel("reports", "step_timings.jsonl"),
        ],
    )


CHECKS = (
    check_immutable_shards,
    check_frozen_tokenizer,
    check_packing_policies,
    check_correct_masks,
    check_curriculum_and_floors,
    check_eval_firewall,
    check_opus_audit,
    check_consumption_ledger,
    check_learning_ledger,
    check_checkpoint_binding,
    check_crash_resume,
    check_replay_hash_match,
    check_fork_new_branch,
    check_packing_and_throughput,
)
