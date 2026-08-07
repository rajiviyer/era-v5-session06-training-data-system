"""Deterministic pseudo-random helpers for sample planning."""

from __future__ import annotations

import hashlib
from typing import TypeVar

T = TypeVar("T")


def deterministic_unit_float(*parts: str | int) -> float:
    """Map stable key parts to a float in [0, 1)."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _sort_key(item: T) -> str:
    sample_id = getattr(item, "sample_id", None)
    if isinstance(sample_id, str):
        return sample_id
    return repr(item)


def deterministic_choice(candidates: tuple[T, ...] | list[T], *parts: str | int) -> T:
    """Pick one item deterministically from candidates."""
    if not candidates:
        raise ValueError("deterministic_choice requires at least one candidate")
    ordered = sorted(candidates, key=_sort_key)
    index = int(deterministic_unit_float(*parts) * len(ordered))
    if index >= len(ordered):
        index = len(ordered) - 1
    return ordered[index]
