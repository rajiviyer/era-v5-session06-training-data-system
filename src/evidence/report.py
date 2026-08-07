"""Render `evidence.md` from the bundle (P11-T06).

The Markdown is generated from `evidence.json` and nothing else, so the two can never
disagree. Every check's `detail` carries the measured value, which is what makes the
document readable as a report rather than a checklist of green ticks.
"""

from __future__ import annotations

from pathlib import Path

from .types import (
    EVIDENCE_MD_FILENAME,
    REQUIREMENT_KEYS,
    REQUIREMENT_TITLES,
    EvidenceBundle,
)

_MARK = {True: "PASS", False: "FAIL"}


def render_evidence_markdown(bundle: EvidenceBundle) -> str:
    """Build the human-readable summary of a collected bundle."""
    passed_count = sum(1 for key in REQUIREMENT_KEYS if bundle.requirements[key].passed)
    lines: list[str] = [
        "# Session 6 Evidence Report",
        "",
        f"**Result:** {_MARK[bundle.passed]} ({passed_count}/{len(REQUIREMENT_KEYS)} "
        "requirements passed)",
        "",
        f"- Generated at: `{bundle.generated_at}`",
        f"- Demo command: `{bundle.demo_command}`",
        f"- Git commit: `{bundle.git_commit or 'unavailable'}`",
        f"- Artifacts: `{bundle.artifacts_dir}/`",
        "",
        "Every value below was computed by reading the generated artifacts. Nothing in "
        "this file is written by hand, and no requirement is marked passed from a "
        "literal in the source.",
        "",
        "## Summary",
        "",
        "| Requirement | Result | Evidence |",
        "|-------------|--------|----------|",
    ]

    for key in REQUIREMENT_KEYS:
        result = bundle.requirements[key]
        lines.append(
            f"| {REQUIREMENT_TITLES.get(key, key)} | {_MARK[result.passed]} | "
            f"`{result.evidence_path}` |"
        )

    if bundle.missing_evidence_paths:
        lines += [
            "",
            "## Missing evidence paths",
            "",
            *(f"- `{path}`" for path in bundle.missing_evidence_paths),
        ]

    lines += ["", "## Detail", ""]
    for key in REQUIREMENT_KEYS:
        result = bundle.requirements[key]
        lines += [
            f"### {REQUIREMENT_TITLES.get(key, key)} ({_MARK[result.passed]})",
            "",
            f"Key: `{key}` · Evidence: "
            + ", ".join(f"`{path}`" for path in result.evidence_paths),
            "",
        ]
        for check in result.checks:
            lines.append(f"- **{_MARK[check.passed]}** {check.name}: {check.detail}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_evidence_markdown(artifacts_dir: Path, bundle: EvidenceBundle) -> Path:
    """Write `evidence.md` beside `evidence.json`."""
    target = Path(artifacts_dir).resolve() / EVIDENCE_MD_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_evidence_markdown(bundle), encoding="utf-8")
    return target
