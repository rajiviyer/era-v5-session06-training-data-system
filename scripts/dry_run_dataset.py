#!/usr/bin/env python3
"""Pre-flight dataset supply and admission audit (PX-T01, PX-T02)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config.loader import load_configs  # noqa: E402
from corpus import load_corpus  # noqa: E402
from preflight import (  # noqa: E402
    PreflightInputs,
    build_admission_audit,
    build_dataset_supply,
    write_data_card,
    write_preflight_reports,
)
from schedule import compile_schedule, write_schedule_json  # noqa: E402
from shards.pipeline import build_shards_with_manifests  # noqa: E402
from tokenizer.frozen import FrozenTokenizer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run dataset supply audit without a training run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ASSIGNMENT / "configs" / "demo.yaml",
        help="Demo config YAML (default: configs/demo.yaml)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts root (default: paths.submission_artifacts from config)",
    )
    parser.add_argument(
        "--build-shards",
        action="store_true",
        help="Build shards and compile schedule if artifacts are missing",
    )
    parser.add_argument(
        "--skip-data-card",
        action="store_true",
        help="Skip data_card.json / data_card.md generation",
    )
    return parser.parse_args()


def _ensure_artifacts(
    demo,
    curriculum,
    artifacts_root: Path,
    *,
    build_shards: bool,
) -> None:
    schedule_path = artifacts_root / "schedule.json"
    manifests_dir = artifacts_root / "manifests"
    has_manifests = any(
        path.name != "shard_registry.json"
        for path in manifests_dir.glob("shard_*.json")
    ) if manifests_dir.is_dir() else False
    if schedule_path.is_file() and has_manifests:
        return
    if not build_shards:
        raise FileNotFoundError(
            "schedule.json or shard manifests missing; rerun with --build-shards"
        )

    artifacts_root.mkdir(parents=True, exist_ok=True)
    _, documents = load_corpus(demo.paths.toy_corpus)
    tokenizer = FrozenTokenizer.load_default(_ASSIGNMENT)
    build_shards_with_manifests(
        documents,
        tokenizer=tokenizer,
        shards_dir=artifacts_root / "shards",
        manifests_dir=manifests_dir,
    )
    schedule = compile_schedule(curriculum, total_steps=demo.training.total_steps)
    write_schedule_json(schedule_path, schedule)


def main() -> None:
    args = _parse_args()
    demo, curriculum = load_configs(_ASSIGNMENT, demo_path=args.config.resolve())
    artifacts_root = (args.artifacts or demo.paths.submission_artifacts).resolve()
    _ensure_artifacts(demo, curriculum, artifacts_root, build_shards=args.build_shards)

    inputs = PreflightInputs(
        assignment_root=_ASSIGNMENT,
        artifacts_root=artifacts_root,
        schedule_path=artifacts_root / "schedule.json",
        manifests_dir=artifacts_root / "manifests",
        corpus_dir=demo.paths.toy_corpus,
    )

    reports_dir = artifacts_root / "reports"
    supply_path, admission_path = write_preflight_reports(inputs, reports_dir)
    supply = build_dataset_supply(inputs)
    admission = build_admission_audit(inputs)

    print(f"wrote {supply_path}")
    print(f"wrote {admission_path}")
    print(
        f"supply: {len(supply['under_supplied_lanes'])} under-supplied lane/stage pairs; "
        f"compiler warnings: {len(supply['compiler_warnings'])}"
    )
    print(f"admission: {admission['admitted_count']} admitted, {admission['blocked_count']} blocked")

    if not args.skip_data_card:
        json_path, md_path = write_data_card(inputs, reports_dir)
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")

    if supply["has_supply_warnings"]:
        print("warning: under-supplied lanes detected (see dataset_supply.json)")


if __name__ == "__main__":
    main()
