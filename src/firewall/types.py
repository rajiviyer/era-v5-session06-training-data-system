"""Eval registry and firewall types (P4-T01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from batch.types import Batch

FirewallDecision = Literal["allowed", "blocked"]


@dataclass(frozen=True)
class EvalRegistryEntry:
    """One never-train eval record tracked by the firewall."""

    entry_id: str
    document_id: str
    shard_id: str | None
    never_train: bool
    benchmark_id: str
    content_hash: str
    canary_strings: tuple[str, ...]


@dataclass(frozen=True)
class EvalRegistry:
    """Eval/test registry loaded before training."""

    schema_version: str
    registry_type: str
    entries: tuple[EvalRegistryEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_type": self.registry_type,
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "document_id": entry.document_id,
                    "shard_id": entry.shard_id,
                    "never_train": entry.never_train,
                    "benchmark_id": entry.benchmark_id,
                    "content_hash": entry.content_hash,
                    "canary_strings": list(entry.canary_strings),
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True)
class BatchCandidate:
    """Candidate microbatch checked by the firewall before loss assignment."""

    candidate_id: str
    global_step: int
    sample_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    batch: Batch | None = None


@dataclass(frozen=True)
class FirewallResult:
    """Outcome of a firewall gate evaluation."""

    candidate_id: str
    global_step: int
    decision: FirewallDecision
    reasons: tuple[str, ...]
    matched_entry_ids: tuple[str, ...]


@dataclass(frozen=True)
class FirewallRejectionEvent:
    """Structured firewall rejection for run.log (P4-T05)."""

    event_type: str
    run_id: str
    branch_id: str
    global_step: int
    candidate_id: str
    decision: FirewallDecision
    reasons: tuple[str, ...]
    matched_entry_ids: tuple[str, ...]
    shard_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def to_log_fields(self) -> dict[str, Any]:
        """Payload for the `firewall_block` run.log event, without the envelope keys."""
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "matched_entry_ids": list(self.matched_entry_ids),
            "shard_ids": list(self.shard_ids),
            "sample_ids": list(self.sample_ids),
        }
