"""Committed BPE tokenizer artifact for Session 6 (decision D7)."""

from __future__ import annotations

import shutil
from pathlib import Path

from .hash import persist_tokenizer_hash

BPE_ARTIFACT_NAME = "bpe_tokenizer.json"


def default_tokenizer_path(assignment_root: Path) -> Path:
    """Path to the committed frozen BPE artifact shipped with this repo."""
    return (assignment_root / "data" / "tokenizer" / BPE_ARTIFACT_NAME).resolve()


def committed_bpe_source(assignment_root: Path) -> Path:
    """Return the canonical committed BPE artifact path."""
    return default_tokenizer_path(assignment_root)


def install_bpe_tokenizer_artifact(
    dest: Path,
    *,
    assignment_root: Path | None = None,
) -> Path:
    """Copy the committed BPE JSON to dest and return dest."""
    resolved_dest = dest.resolve()
    if assignment_root is None:
        assignment_root = resolved_dest.parents[2]
    source = committed_bpe_source(assignment_root)
    if not source.is_file():
        raise FileNotFoundError(f"Committed BPE artifact not found: {source}")
    resolved_dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != resolved_dest:
        shutil.copy2(source, resolved_dest)
    return resolved_dest


def ensure_bpe_tokenizer_artifact(
    path: Path,
    *,
    assignment_root: Path | None = None,
) -> Path:
    """Ensure the BPE artifact exists; copy from the committed copy if missing."""
    resolved = path.resolve()
    if not resolved.is_file():
        root = assignment_root or resolved.parents[2]
        install_bpe_tokenizer_artifact(resolved, assignment_root=root)
    return resolved


def rebuild_bpe_tokenizer_artifact(
    path: Path,
    *,
    assignment_root: Path | None = None,
) -> Path:
    """Refresh BPE artifact from the committed copy and rewrite hash + manifest sidecars."""
    resolved = path.resolve()
    root = assignment_root or resolved.parents[2]
    install_bpe_tokenizer_artifact(resolved, assignment_root=root)
    persist_tokenizer_hash(resolved)
    from .manifest import write_tokenizer_manifest

    write_tokenizer_manifest(resolved)
    return resolved
