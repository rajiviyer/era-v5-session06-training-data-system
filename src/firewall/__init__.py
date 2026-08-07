"""Eval registry and training firewall checks."""

from .gate import assert_no_eval_loss, evaluate_firewall
from .log import log_firewall_rejection
from .registry import (
    REGISTRY_FILENAME,
    build_eval_registry,
    candidate_from_planned_samples,
    load_eval_registry,
    write_eval_registry,
)
from .types import BatchCandidate, EvalRegistry, FirewallRejectionEvent, FirewallResult

__all__ = [
    "REGISTRY_FILENAME",
    "BatchCandidate",
    "EvalRegistry",
    "FirewallRejectionEvent",
    "FirewallResult",
    "assert_no_eval_loss",
    "build_eval_registry",
    "candidate_from_planned_samples",
    "evaluate_firewall",
    "load_eval_registry",
    "log_firewall_rejection",
    "write_eval_registry",
]
