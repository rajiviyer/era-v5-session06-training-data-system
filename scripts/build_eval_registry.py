#!/usr/bin/env python3
"""Build eval_registry.json from toy corpus and shard manifests (P4-T02)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config.loader import load_demo_config  # noqa: E402
from corpus import load_corpus  # noqa: E402
from firewall import build_eval_registry, write_eval_registry  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Session 6 eval registry.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ASSIGNMENT / "configs" / "demo.yaml",
        help="Demo config YAML (default: configs/demo.yaml)",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=None,
        help="Shard manifests directory (default: submission_artifacts/manifests)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output eval_registry.json path (default: submission_artifacts/eval_registry.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    demo = load_demo_config(args.config, assignment_root=_ASSIGNMENT)
    output_root = demo.paths.submission_artifacts.resolve()
    manifests_dir = (args.manifests or output_root / "manifests").resolve()
    output_path = (args.output or output_root / "eval_registry.json").resolve()

    _, documents = load_corpus(demo.paths.toy_corpus)
    registry = build_eval_registry(documents, manifests_dir=manifests_dir if manifests_dir.is_dir() else None)
    write_eval_registry(output_path, registry)
    print(f"wrote {len(registry.entries)} eval entries -> {output_path}")


if __name__ == "__main__":
    main()
