"""Recovery errors (P9)."""

from __future__ import annotations


class RecoveryError(Exception):
    """Invalid crash, resume, replay, or fork configuration or state."""
