"""Load and validate Session 6 YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schemas import (
    CurriculumConfig,
    CurriculumYamlRoot,
    DemoConfig,
    DemoYamlRoot,
    curriculum_from_yaml,
    demo_from_yaml,
    validate_phase_coverage,
)

__all__ = [
    "ConfigError",
    "CurriculumConfig",
    "DemoConfig",
    "load_configs",
    "load_curriculum_config",
    "load_demo_config",
]


class ConfigError(ValueError):
    """Raised when a configuration file fails validation."""


def _format_validation_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        prefix = location or "config"
        messages.append(f"{prefix}: {error['msg']}")
    return "; ".join(messages)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping")
    return raw


def _assignment_root_for_config(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent.name == "configs":
        return resolved.parent.parent
    return resolved.parent


def _validate_model(model_cls: type, raw: dict[str, Any]) -> Any:
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def load_demo_config(
    path: Path,
    *,
    assignment_root: Path | None = None,
) -> DemoConfig:
    """Load and validate demo.yaml."""
    config_path = path.resolve()
    root = (assignment_root or _assignment_root_for_config(config_path)).resolve()
    parsed = _validate_model(DemoYamlRoot, _load_yaml(config_path))
    return demo_from_yaml(parsed, assignment_root=root)


def load_curriculum_config(
    path: Path,
    *,
    total_steps: int | None = None,
) -> CurriculumConfig:
    """Load and validate curriculum.yaml."""
    parsed = _validate_model(CurriculumYamlRoot, _load_yaml(path.resolve()))
    curriculum = curriculum_from_yaml(parsed)
    try:
        validate_phase_coverage(curriculum.phases, total_steps=total_steps)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return curriculum


def load_configs(
    assignment_root: Path,
    *,
    demo_path: Path | None = None,
    curriculum_path: Path | None = None,
) -> tuple[DemoConfig, CurriculumConfig]:
    """Load demo and curriculum configs with cross-validation."""
    root = assignment_root.resolve()
    demo_file = (demo_path or root / "configs" / "demo.yaml").resolve()
    demo = load_demo_config(demo_file, assignment_root=root)
    curriculum_file = (curriculum_path or demo.paths.curriculum_config).resolve()
    curriculum = load_curriculum_config(
        curriculum_file,
        total_steps=demo.training.total_steps,
    )
    if curriculum.always_on.fraction != curriculum.batch.always_on_fraction:
        raise ConfigError(
            "curriculum.always_on.fraction must match curriculum.batch.always_on_fraction"
        )
    return demo, curriculum
