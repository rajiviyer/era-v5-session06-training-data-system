"""Atomic shard and manifest file writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write bytes via a temp file in the same directory, then rename."""
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(resolved.name + ".tmp")
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, resolved)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file in the same directory, then rename."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_bytes_atomic(path, text.encode("utf-8"))
