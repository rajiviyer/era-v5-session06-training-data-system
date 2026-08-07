"""OPUS batch selector and audit trail (P5)."""

from .audit import append_opus_audit, load_opus_audit, query_opus_audit
from .pipeline import run_batch_gate
from .selector import evaluate_opus
from .types import BatchPipelineResult, OpusAuditRecord, OpusResult

__all__ = [
    "BatchPipelineResult",
    "OpusAuditRecord",
    "OpusResult",
    "append_opus_audit",
    "evaluate_opus",
    "load_opus_audit",
    "query_opus_audit",
    "run_batch_gate",
]
