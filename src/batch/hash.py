"""Batch fingerprint hashes (P2-T07)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_PREFIX = "sha256:"


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{HASH_PREFIX}{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def loss_mask_hash(loss_mask: tuple[tuple[int, ...], ...]) -> str:
    """Hash the loss mask tensor for a batch."""
    return _digest({"loss_mask": [list(row) for row in loss_mask]})


def batch_content_hash(
    *,
    input_ids: tuple[tuple[int, ...], ...],
    loss_mask: tuple[tuple[int, ...], ...],
    attention_mask: tuple[tuple[tuple[int, ...], ...], ...],
    position_ids: tuple[tuple[int, ...], ...],
) -> str:
    """Hash batch tensors for reproducibility checks."""
    return _digest(
        {
            "input_ids": [list(row) for row in input_ids],
            "loss_mask": [list(row) for row in loss_mask],
            "attention_mask": [[list(row) for row in matrix] for matrix in attention_mask],
            "position_ids": [list(row) for row in position_ids],
        }
    )
