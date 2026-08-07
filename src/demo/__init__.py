"""One-command demo orchestration (P11)."""

from .pipeline import PENDING_PHASES, DemoResult, PhaseResult, run_demo

__all__ = [
    "PENDING_PHASES",
    "DemoResult",
    "PhaseResult",
    "run_demo",
]
