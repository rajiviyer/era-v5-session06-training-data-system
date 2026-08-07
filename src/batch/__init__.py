"""Batch builder: loss, attention, and position masks plus content hashing."""

from .agentic import encode_agentic_turns
from .assemble import AssembledBatch, assemble_microbatch
from .builder import build_batch
from .hash import batch_content_hash, loss_mask_hash
from .types import Batch, BatchBuildConfig

__all__ = [
    "AssembledBatch",
    "Batch",
    "BatchBuildConfig",
    "assemble_microbatch",
    "batch_content_hash",
    "build_batch",
    "encode_agentic_turns",
    "loss_mask_hash",
]
