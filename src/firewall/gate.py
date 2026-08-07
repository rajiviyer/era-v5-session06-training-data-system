"""Firewall gate on candidate batches (P4-T04)."""

from __future__ import annotations

from typing import Any

from .overlap import check_overlaps
from .types import BatchCandidate, EvalRegistry, FirewallResult


def evaluate_firewall(
    candidate: BatchCandidate,
    registry: EvalRegistry,
    *,
    documents_by_id: dict[str, dict[str, Any]] | None = None,
) -> FirewallResult:
    """Block candidates that overlap eval registry entries before loss assignment."""
    reasons, matched_entry_ids = check_overlaps(
        candidate,
        registry,
        documents_by_id=documents_by_id,
    )
    if reasons:
        return FirewallResult(
            candidate_id=candidate.candidate_id,
            global_step=candidate.global_step,
            decision="blocked",
            reasons=reasons,
            matched_entry_ids=matched_entry_ids,
        )
    return FirewallResult(
        candidate_id=candidate.candidate_id,
        global_step=candidate.global_step,
        decision="allowed",
        reasons=(),
        matched_entry_ids=(),
    )


def assert_no_eval_loss(candidate: BatchCandidate, registry: EvalRegistry) -> None:
    """Ensure blocked eval samples cannot carry loss in an attached batch."""
    if candidate.batch is None:
        return

    blocked_documents = {entry.document_id for entry in registry.entries if entry.never_train}
    for row_index, document_row in enumerate(candidate.batch.document_ids):
        loss_row = candidate.batch.loss_mask[row_index]
        for token_index, document_id in enumerate(document_row):
            if document_id in blocked_documents and loss_row[token_index] == 1:
                raise ValueError(
                    f"eval document {document_id} has loss_mask=1 in candidate {candidate.candidate_id}"
                )
