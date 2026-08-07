"""Agentic role-aware tokenization and loss eligibility helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizer.frozen import FrozenTokenizer


def encode_agentic_turns(
    text: str,
    tokenizer: FrozenTokenizer,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Tokenize agentic JSONL turns per line and mark assistant tokens loss-eligible.

    Encoding happens line by line so that role boundaries stay aligned with token
    boundaries; encoding the whole trajectory at once would let a BPE merge span two
    turns and blur the assistant/user boundary the loss mask depends on.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    token_ids: list[int] = []
    eligible: list[bool] = []

    for index, line in enumerate(lines):
        record = json.loads(line)
        role = record.get("role", "")
        piece = line if index == len(lines) - 1 else f"{line}\n"
        piece_ids = tokenizer.encode(piece)
        token_ids.extend(piece_ids)
        eligible.extend([role == "assistant"] * len(piece_ids))

    return tuple(token_ids), tuple(eligible)
