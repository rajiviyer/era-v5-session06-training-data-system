"""Rebuild a committed batch from its ledger row (P9-T05, P10-T01).

Replay and the packing metrics both need to reconstruct a historical microbatch, and
they must reconstruct it *identically* or one of them is measuring a batch the run never
saw. Keeping the call in one place means a change to packing defaults cannot silently
move only one of them.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from batch import AssembledBatch, assemble_microbatch

from .types import ConsumptionLedgerEvent

if TYPE_CHECKING:
    from tokenizer.frozen import FrozenTokenizer


def rebuild_batch(
    row: ConsumptionLedgerEvent,
    *,
    documents_by_id: dict[str, dict[str, Any]],
    tokenizer: FrozenTokenizer,
    seq_len: int,
) -> AssembledBatch:
    """Re-tokenize, re-pack, and re-mask the microbatch one ledger row describes."""
    return assemble_microbatch(
        row.packed_sample_ids,
        row.shard_ids,
        documents_by_id=documents_by_id,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )
