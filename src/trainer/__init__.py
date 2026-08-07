"""Tiny causal LM training loop."""

from .errors import TrainerError
from .loop import (
    MicrobatchOutcome,
    RunSummary,
    TrainingContext,
    TrainingPaths,
    TrainingRunner,
    build_training_runner,
)
from .loss import DocumentLoss, batch_to_tensors, masked_causal_loss, per_document_losses
from .model import TinyModelConfig, build_model
from .step import MicrobatchResult, StepResult, TinyTrainer

__all__ = [
    "DocumentLoss",
    "MicrobatchOutcome",
    "MicrobatchResult",
    "RunSummary",
    "StepResult",
    "TinyModelConfig",
    "TinyTrainer",
    "TrainerError",
    "TrainingContext",
    "TrainingPaths",
    "TrainingRunner",
    "batch_to_tensors",
    "build_model",
    "build_training_runner",
    "masked_causal_loss",
    "per_document_losses",
]
