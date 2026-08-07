"""Deterministic data card generation from corpus and shard artifacts (PX-T02)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus import load_corpus
from shards.io import write_json_atomic
from shards.registry import REGISTRY_FILENAME, load_registry_index
from tokenizer.bpe import default_tokenizer_path
from tokenizer.hash import compute_tokenizer_hash_from_artifact
from tokenizer.manifest import MANIFEST_FILENAME, load_tokenizer_manifest

from .supply import PreflightInputs, _lane_supply_from_manifests, _load_manifests, corpus_lane_counts

DATA_CARD_JSON_FILENAME = "data_card.json"
DATA_CARD_MD_FILENAME = "data_card.md"


def build_data_card(inputs: PreflightInputs) -> dict[str, Any]:
    """Build a deterministic data card from corpus and shard manifests."""
    tokenizer_path = default_tokenizer_path(inputs.assignment_root)
    tokenizer_hash = compute_tokenizer_hash_from_artifact(tokenizer_path)
    tokenizer_manifest_path = inputs.artifacts_root / "manifests" / MANIFEST_FILENAME
    tokenizer_manifest: dict[str, Any] | None = None
    if tokenizer_manifest_path.is_file():
        tokenizer_manifest = load_tokenizer_manifest(tokenizer_manifest_path)

    manifests = _load_manifests(inputs.manifests_dir)
    lane_supply = _lane_supply_from_manifests(manifests)
    corpus_counts = corpus_lane_counts(inputs.corpus_dir)

    admitted_tokens = sum(
        row["admitted_tokens"] for row in lane_supply.values()
    )
    admitted_shards = sum(
        row["admitted_shards"] for row in lane_supply.values()
    )

    registry_path = inputs.manifests_dir / REGISTRY_FILENAME
    registry = load_registry_index(registry_path) if registry_path.is_file() else {}

    return {
        "schema_version": "1.0",
        "tokenizer": {
            "artifact_path": str(tokenizer_path.relative_to(inputs.assignment_root)),
            "tokenizer_hash": tokenizer_hash,
            "model_type": (tokenizer_manifest or {}).get("model_type", "BPE"),
            "vocab_size": (tokenizer_manifest or {}).get("vocab_size"),
            "merge_count": (tokenizer_manifest or {}).get("merge_count"),
        },
        "corpus": {
            "path": str(inputs.corpus_dir.relative_to(inputs.assignment_root)),
            "document_counts_by_lane": corpus_counts,
            "ready_document_total": sum(corpus_counts.values()),
        },
        "shards": {
            "manifest_count": len(manifests),
            "admitted_shard_count": admitted_shards,
            "blocked_shard_count": len(registry.get("blocked_shard_ids", [])),
            "admitted_token_total": admitted_tokens,
            "admitted_tokens_by_lane": {
                lane: lane_supply.get(lane, {}).get("admitted_tokens", 0)
                for lane in sorted(lane_supply)
            },
        },
    }


def render_data_card_markdown(card: dict[str, Any]) -> str:
    """Render a short human-readable data card."""
    tokenizer = card["tokenizer"]
    corpus = card["corpus"]
    shards = card["shards"]

    lines = [
        "# Session 6 Data Card",
        "",
        "## Tokenizer",
        f"- Hash: `{tokenizer['tokenizer_hash']}`",
        f"- Model: {tokenizer.get('model_type', 'BPE')}",
        f"- Vocab size: {tokenizer.get('vocab_size', 'unknown')}",
        f"- Merge count: {tokenizer.get('merge_count', 'unknown')}",
        "",
        "## Corpus",
        f"- Ready documents: {corpus['ready_document_total']}",
        "",
        "| Lane | Documents |",
        "|------|-----------|",
    ]
    for lane, count in corpus["document_counts_by_lane"].items():
        lines.append(f"| {lane} | {count} |")

    lines.extend(
        [
            "",
            "## Admitted shards",
            f"- Admitted shards: {shards['admitted_shard_count']}",
            f"- Blocked shards: {shards['blocked_shard_count']}",
            f"- Admitted tokens: {shards['admitted_token_total']}",
            "",
            "| Lane | Admitted tokens |",
            "|------|-----------------|",
        ]
    )
    for lane, tokens in shards["admitted_tokens_by_lane"].items():
        lines.append(f"| {lane} | {tokens} |")
    lines.append("")
    return "\n".join(lines)


def write_data_card(inputs: PreflightInputs, reports_dir: Path) -> tuple[Path, Path]:
    """Write data_card.json and data_card.md deterministically."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    card = build_data_card(inputs)
    json_path = reports_dir / DATA_CARD_JSON_FILENAME
    md_path = reports_dir / DATA_CARD_MD_FILENAME
    write_json_atomic(json_path, card)
    md_path.write_text(render_data_card_markdown(card), encoding="utf-8")
    return json_path, md_path
