#!/usr/bin/env python3
"""Build tokenized shards, manifests, and registry from the toy corpus (P1-T08)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config.loader import load_demo_config  # noqa: E402
from corpus import load_corpus  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Session 6 toy shards and manifests.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ASSIGNMENT / "configs" / "demo.yaml",
        help="Demo config YAML (default: configs/demo.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: paths.submission_artifacts from config)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing shards/ and manifests/ under output before building",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    demo = load_demo_config(args.config, assignment_root=_ASSIGNMENT)
    output_root = (args.output or demo.paths.submission_artifacts).resolve()
    shards_dir = output_root / "shards"
    manifests_dir = output_root / "manifests"

    if args.clean:
        for target in (shards_dir, manifests_dir):
            if target.exists():
                shutil.rmtree(target)

    _, documents = load_corpus(demo.paths.toy_corpus)
    tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
    result = build_shards_with_manifests(
        documents,
        tokenizer=tokenizer,
        shards_dir=shards_dir,
        manifests_dir=manifests_dir,
    )

    admitted = len(result.registry["admitted_shard_ids"])
    blocked = len(result.registry["blocked_shard_ids"])
    print(f"Built {len(result.shards)} shards under {shards_dir}")
    print(f"Wrote {len(result.manifests)} manifests under {manifests_dir}")
    print(f"Registry: {admitted} admitted, {blocked} blocked")


if __name__ == "__main__":
    main()
