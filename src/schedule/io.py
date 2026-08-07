"""Read and write compiled schedule.json artifacts (P3-T03)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import ScheduleError
from .types import CompiledSchedule, StageRecord, StepSchedule


def write_schedule_json(path: Path, schedule: CompiledSchedule) -> None:
    """Atomically write schedule.json."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(schedule.to_dict(), indent=2, sort_keys=True)
    payload = payload + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, target)


def load_schedule_json(path: Path) -> CompiledSchedule:
    """Load a compiled schedule written by write_schedule_json."""
    if not path.is_file():
        raise ScheduleError(f"schedule file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return _schedule_from_dict(raw)


def _schedule_from_dict(raw: dict[str, Any]) -> CompiledSchedule:
    try:
        boundaries = tuple(
            StageRecord(
                name=str(phase["name"]),
                step_start=int(phase["step_start"]),
                step_end=int(phase["step_end"]),
                opus_mixture={str(k): float(v) for k, v in phase["opus_mixture"].items()},
                lr_multiplier=(
                    float(phase["lr_multiplier"]) if phase.get("lr_multiplier") is not None else None
                ),
                anneal_eligible_only=bool(phase.get("anneal_eligible_only", False)),
                tier_d_indic_fraction=(
                    float(phase["tier_d_indic_fraction"])
                    if phase.get("tier_d_indic_fraction") is not None
                    else None
                ),
            )
            for phase in raw["phase_boundaries"]
        )
        steps = tuple(
            StepSchedule(
                step=int(entry["step"]),
                phase=str(entry["phase"]),
                in_transition=bool(entry["in_transition"]),
                transition_from=(
                    str(entry["transition_from"]) if entry.get("transition_from") is not None else None
                ),
                transition_to=(
                    str(entry["transition_to"]) if entry.get("transition_to") is not None else None
                ),
                always_on_fraction=float(entry["always_on_fraction"]),
                opus_fraction=float(entry["opus_fraction"]),
                always_on_sub_shares={
                    str(k): float(v) for k, v in entry["always_on_sub_shares"].items()
                },
                opus_lane_quotas={str(k): float(v) for k, v in entry["opus_lane_quotas"].items()},
                lr_multiplier=(
                    float(entry["lr_multiplier"]) if entry.get("lr_multiplier") is not None else None
                ),
                anneal_eligible_only=bool(entry.get("anneal_eligible_only", False)),
            )
            for entry in raw["steps"]
        )
        blend_raw = raw["transition_blend"]
        if not isinstance(blend_raw, list) or len(blend_raw) != 2:
            raise ScheduleError("transition_blend must be a list of two numbers")
        return CompiledSchedule(
            schema_version=str(raw["schema_version"]),
            total_steps=int(raw["total_steps"]),
            always_on_fraction=float(raw["always_on_fraction"]),
            opus_fraction=float(raw["opus_fraction"]),
            warmup_steps=int(raw["warmup_steps"]),
            transition_blend=(float(blend_raw[0]), float(blend_raw[1])),
            protected_floor_lanes=tuple(str(lane) for lane in raw["protected_floor_lanes"]),
            phase_boundaries=boundaries,
            steps=steps,
            warnings=tuple(str(item) for item in raw.get("warnings", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScheduleError(f"invalid schedule.json: {exc}") from exc
