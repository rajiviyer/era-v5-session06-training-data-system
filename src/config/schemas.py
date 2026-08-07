"""Typed configuration schemas for Session 6 demo runs (Pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LANE_KEYS = frozenset(
    {"web", "code", "indic", "stem", "reasoning", "long_context", "agentic"}
)
WEIGHT_TOLERANCE = 1e-6


def _validate_mixture_weights(weights: dict[str, float], *, label: str) -> dict[str, float]:
    if len(weights) < 2:
        raise ValueError(f"{label} must define at least two lanes")
    unknown = set(weights) - LANE_KEYS
    if unknown:
        raise ValueError(f"{label} contains unknown lanes: {sorted(unknown)}")
    if sum(1 for weight in weights.values() if weight > 0) < 2:
        raise ValueError(f"{label} must have at least two non-zero lane weights")
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"{label} weights must sum to 1.0 (got {total:.6f})")
    return dict(weights)


def _validate_unit_fraction(value: float, *, label: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return value


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: Annotated[str, Field(min_length=1)]
    branch_id: Annotated[str, Field(min_length=1)]
    seed: int


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq_len: Annotated[int, Field(gt=0)]
    global_batch_size: Annotated[int, Field(gt=0)]
    microbatch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    total_steps: Annotated[int, Field(gt=0)]
    checkpoint_interval: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> Self:
        if self.microbatch_size > self.global_batch_size:
            raise ValueError("demo.training.microbatch_size cannot exceed global_batch_size")
        expected = self.microbatch_size * self.gradient_accumulation_steps
        if self.global_batch_size != expected:
            raise ValueError(
                "demo.training.global_batch_size must equal "
                "microbatch_size * gradient_accumulation_steps "
                f"(expected {expected}, got {self.global_batch_size})"
            )
        return self


class RecoveryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    crash_at_step: int
    resume_from_checkpoint_step: int
    replay_start_step: int
    replay_end_step: int
    fork_from_checkpoint_step: int
    fork_branch_id: Annotated[str, Field(min_length=1)]


def _validate_recovery_config(recovery: RecoveryConfig, total_steps: int) -> None:
    if not 0 < recovery.crash_at_step <= total_steps:
        raise ValueError("demo.recovery.crash_at_step must be in (0, total_steps]")
    if not 0 <= recovery.resume_from_checkpoint_step < recovery.crash_at_step:
        raise ValueError(
            "demo.recovery.resume_from_checkpoint_step must be >= 0 and < crash_at_step"
        )
    if recovery.replay_start_step > recovery.replay_end_step:
        raise ValueError("demo.recovery.replay_start_step cannot exceed replay_end_step")
    if recovery.replay_end_step > total_steps:
        raise ValueError("demo.recovery.replay_end_step cannot exceed total_steps")
    if recovery.fork_from_checkpoint_step > total_steps:
        raise ValueError("demo.recovery.fork_from_checkpoint_step cannot exceed total_steps")


class PathsConfigRaw(BaseModel):
    model_config = ConfigDict(frozen=True)

    toy_corpus: Annotated[str, Field(min_length=1)]
    curriculum_config: Annotated[str, Field(min_length=1)]
    submission_artifacts: Annotated[str, Field(min_length=1)]


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    toy_corpus: Path
    curriculum_config: Path
    submission_artifacts: Path


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_layers: Annotated[int, Field(gt=0)]
    n_heads: Annotated[int, Field(gt=0)]
    d_model: Annotated[int, Field(gt=0)]
    d_ff: Annotated[int, Field(gt=0)]
    dropout: Annotated[float, Field(ge=0.0, le=1.0)]


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(min_length=1)]
    learning_rate: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    beta1: Annotated[float, Field(ge=0.0, le=1.0)]
    beta2: Annotated[float, Field(ge=0.0, le=1.0)]
    grad_clip: Annotated[float, Field(gt=0)]


class OpusConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    accept_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    expected_rejection_rate: Annotated[float, Field(ge=0.0, le=1.0)]


class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_every_step: bool


class DemoYamlRoot(BaseModel):
    """Shape of configs/demo.yaml before path resolution."""

    model_config = ConfigDict(frozen=True)

    run: RunConfig
    training: TrainingConfig
    recovery: RecoveryConfig
    paths: PathsConfigRaw
    model: ModelConfig
    optimizer: OptimizerConfig
    opus: OpusConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_recovery_steps(self) -> Self:
        _validate_recovery_config(self.recovery, self.training.total_steps)
        return self


class DemoConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run: RunConfig
    training: TrainingConfig
    recovery: RecoveryConfig
    paths: PathsConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    opus: OpusConfig
    logging: LoggingConfig
    assignment_root: Path


class BatchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    always_on_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    opus_fraction: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> Self:
        total = self.always_on_fraction + self.opus_fraction
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError("curriculum.batch fractions must sum to 1.0")
        return self


class TransitionsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warmup_steps: Annotated[int, Field(gt=0)]
    blend: tuple[float, float]

    @field_validator("blend", mode="before")
    @classmethod
    def coerce_blend(cls, value: object) -> tuple[float, float]:
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError("curriculum.transitions.blend must be a list of two numbers")
            return float(value[0]), float(value[1])
        if isinstance(value, tuple):
            return value
        raise ValueError("curriculum.transitions.blend must be a list of two numbers")

    @field_validator("blend")
    @classmethod
    def validate_blend(cls, value: tuple[float, float]) -> tuple[float, float]:
        if abs(sum(value) - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError("curriculum.transitions.blend must sum to 1.0")
        return value


class ProtectedFloorsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    lanes: tuple[str, ...]

    @field_validator("lanes")
    @classmethod
    def non_empty_lanes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("curriculum.protected_floors.lanes must be a non-empty list")
        return value


class CurriculumPhase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(min_length=1)]
    step_start: int
    step_end: int
    opus_mixture: dict[str, float]
    lr_multiplier: float | None = None
    anneal_eligible_only: bool = False
    tier_d_indic_fraction: float | None = None

    @field_validator("opus_mixture")
    @classmethod
    def validate_opus_mixture(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_mixture_weights(value, label="curriculum.phases[].opus_mixture")

    @model_validator(mode="after")
    def validate_step_range(self) -> Self:
        if self.step_start >= self.step_end:
            raise ValueError("curriculum.phases[].step_start must be < step_end")
        return self


class AlwaysOnConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    stage_invariant: bool
    sub_shares: dict[str, float]

    @field_validator("sub_shares")
    @classmethod
    def validate_sub_shares(cls, value: dict[str, float]) -> dict[str, float]:
        for key, share in value.items():
            _validate_unit_fraction(share, label=f"curriculum.always_on.sub_shares.{key}")
        total = sum(value.values())
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(
                f"curriculum.always_on.sub_shares must sum to 1.0 (got {total:.6f})"
            )
        return value


class AnnealReserveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    holdback_until_phase: Annotated[str, Field(min_length=1)]
    manifest_tag: Annotated[str, Field(min_length=1)]


class ManifestFiltersConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    difficulty_field: Annotated[str, Field(min_length=1)]
    reasoning_length_field: Annotated[str, Field(min_length=1)]
    anneal_holdback_field: Annotated[str, Field(min_length=1)]
    opus_excludes: tuple[str, ...]


class CurriculumYamlRoot(BaseModel):
    """Shape of configs/curriculum.yaml."""

    model_config = ConfigDict(frozen=True)

    batch: BatchConfig
    transitions: TransitionsConfig
    protected_floors: ProtectedFloorsConfig
    phases: tuple[CurriculumPhase, ...]
    always_on: AlwaysOnConfig
    anneal_reserve: AnnealReserveConfig
    manifest_filters: ManifestFiltersConfig

    @field_validator("phases")
    @classmethod
    def non_empty_phases(
        cls, value: tuple[CurriculumPhase, ...]
    ) -> tuple[CurriculumPhase, ...]:
        if not value:
            raise ValueError("curriculum.phases must be a non-empty list")
        return value


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch: BatchConfig
    transitions: TransitionsConfig
    protected_floors: ProtectedFloorsConfig
    phases: tuple[CurriculumPhase, ...]
    always_on: AlwaysOnConfig
    anneal_reserve: AnnealReserveConfig
    manifest_filters: ManifestFiltersConfig


def validate_phase_coverage(
    phases: tuple[CurriculumPhase, ...],
    *,
    total_steps: int | None,
) -> None:
    """Ensure curriculum phases are contiguous and cover total_steps when given."""
    if not phases:
        raise ValueError("curriculum.phases must not be empty")

    sorted_phases = sorted(phases, key=lambda phase: phase.step_start)
    if sorted_phases[0].step_start != 0:
        raise ValueError("curriculum first phase must start at step 0")

    for previous, current in zip(sorted_phases, sorted_phases[1:]):
        if current.step_start != previous.step_end:
            raise ValueError(
                "curriculum phases must be contiguous: "
                f"'{previous.name}' ends at {previous.step_end}, "
                f"'{current.name}' starts at {current.step_start}"
            )

    if total_steps is not None and sorted_phases[-1].step_end != total_steps:
        raise ValueError(
            "curriculum final phase must end at demo.training.total_steps "
            f"(expected {total_steps}, got {sorted_phases[-1].step_end})"
        )


def curriculum_from_yaml(root: CurriculumYamlRoot) -> CurriculumConfig:
    return CurriculumConfig.model_validate(root.model_dump())


def demo_from_yaml(root: DemoYamlRoot, *, assignment_root: Path) -> DemoConfig:
    resolved_root = assignment_root.resolve()

    def resolve(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (resolved_root / path).resolve()

    return DemoConfig(
        run=root.run,
        training=root.training,
        recovery=root.recovery,
        paths=PathsConfig(
            toy_corpus=resolve(root.paths.toy_corpus),
            curriculum_config=resolve(root.paths.curriculum_config),
            submission_artifacts=resolve(root.paths.submission_artifacts),
        ),
        model=root.model,
        optimizer=root.optimizer,
        opus=root.opus,
        logging=root.logging,
        assignment_root=resolved_root,
    )
