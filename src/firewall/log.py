"""Firewall rejection logging (P4-T05).

Rejections go to the run's structured log through `RunLogWriter`, not through a private
append path: `run.log` is one ordered stream (P11-T01), and a block that bypassed the
sequencer would land in the file with no position relative to the batch it stopped.
"""

from __future__ import annotations

from runlog import RunLogWriter

from .types import FirewallRejectionEvent, FirewallResult

FIREWALL_EVENT_TYPE = "firewall_block"


def rejection_event_from_result(
    result: FirewallResult,
    *,
    run_id: str,
    branch_id: str,
    shard_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> FirewallRejectionEvent:
    """Build a structured rejection event from a blocked firewall result."""
    return FirewallRejectionEvent(
        event_type=FIREWALL_EVENT_TYPE,
        run_id=run_id,
        branch_id=branch_id,
        global_step=result.global_step,
        candidate_id=result.candidate_id,
        decision=result.decision,
        reasons=result.reasons,
        matched_entry_ids=result.matched_entry_ids,
        shard_ids=shard_ids,
        sample_ids=sample_ids,
    )


def log_firewall_rejection(
    run_log: RunLogWriter,
    result: FirewallResult,
    *,
    run_id: str,
    branch_id: str,
    shard_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> FirewallRejectionEvent:
    """Write a blocked candidate to run.log and return the event."""
    if result.decision != "blocked":
        raise ValueError("log_firewall_rejection requires a blocked firewall result")
    event = rejection_event_from_result(
        result,
        run_id=run_id,
        branch_id=branch_id,
        shard_ids=shard_ids,
        sample_ids=sample_ids,
    )
    run_log.emit(FIREWALL_EVENT_TYPE, **event.to_log_fields())
    return event
