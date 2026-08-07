"""Training step with gradient accumulation (P7-T03).

One global step = `gradient_accumulation_steps` microbatches, then one optimizer
update. Microbatches that the firewall or OPUS rejects never reach this module, so a
global step can end with fewer accumulated microbatches than planned; the loop decides
whether to still apply the update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from batch.types import Batch
from config.schemas import OptimizerConfig

from .errors import TrainerError
from .loss import DocumentLoss, batch_to_tensors, masked_causal_loss, per_document_losses
from .model import TinyCausalLM


@dataclass(frozen=True)
class MicrobatchResult:
    """Forward/backward outcome for one microbatch."""

    global_step: int
    microbatch_index: int
    loss: float
    loss_bearing_tokens: int
    total_positions: int
    per_document_loss: tuple[DocumentLoss, ...]


@dataclass(frozen=True)
class StepResult:
    """Optimizer-update outcome for one global step.

    `mean_loss` and `perplexity` are `None` when every microbatch in the step was
    blocked or rejected. A step that trained on nothing has no loss, and reporting 0.0
    would read as a perfect score to any downstream aggregate.
    """

    global_step: int
    microbatches: int
    mean_loss: float | None
    perplexity: float | None
    loss_bearing_tokens: int
    grad_norm: float
    learning_rate: float
    optimizer_stepped: bool


class TinyTrainer:
    """Owns the model, optimizer, and gradient accumulation state."""

    def __init__(
        self,
        model: TinyCausalLM,
        config: OptimizerConfig,
        *,
        gradient_accumulation_steps: int,
        device: torch.device | None = None,
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise TrainerError("gradient_accumulation_steps must be positive")
        if config.name.lower() != "adamw":
            raise TrainerError(f"unsupported optimizer: {config.name}")

        self.model = model
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulated = 0
        self._loss_sum = 0.0
        self._token_sum = 0

    def train_microbatch(
        self,
        batch: Batch,
        *,
        global_step: int,
        microbatch_index: int,
    ) -> MicrobatchResult:
        """Forward, masked loss, and scaled backward for one microbatch."""
        self.model.train()
        tensors = batch_to_tensors(batch, device=self.device)
        logits = self.model(
            tensors.input_ids,
            attention_mask=tensors.attention_mask,
            position_ids=tensors.position_ids,
        )
        masked = masked_causal_loss(logits, tensors)
        if not math.isfinite(masked.value):
            raise TrainerError(
                f"non-finite loss at step {global_step} microbatch {microbatch_index}"
            )

        # Scale by the planned accumulation count, not the accepted count: keeping the
        # per-microbatch math independent of downstream gate outcomes means a rejected
        # microbatch simply contributes nothing, rather than reweighting its peers.
        (masked.loss / self.gradient_accumulation_steps).backward()

        self._accumulated += 1
        self._loss_sum += masked.value * masked.loss_bearing_tokens
        self._token_sum += masked.loss_bearing_tokens

        return MicrobatchResult(
            global_step=global_step,
            microbatch_index=microbatch_index,
            loss=masked.value,
            loss_bearing_tokens=masked.loss_bearing_tokens,
            total_positions=tensors.batch_size * tensors.seq_len,
            per_document_loss=per_document_losses(masked, batch),
        )

    def finish_step(self, global_step: int, *, lr_multiplier: float | None = None) -> StepResult:
        """Clip, apply the optimizer update, and reset accumulation state."""
        learning_rate = self.config.learning_rate * (lr_multiplier or 1.0)
        microbatches = self._accumulated
        token_sum = self._token_sum

        if microbatches == 0:
            # Every microbatch in this step was blocked or rejected: nothing to apply.
            self._reset_accumulation()
            return StepResult(
                global_step=global_step,
                microbatches=0,
                mean_loss=None,
                perplexity=None,
                loss_bearing_tokens=0,
                grad_norm=0.0,
                learning_rate=learning_rate,
                optimizer_stepped=False,
            )

        mean_loss = self._loss_sum / token_sum

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        grad_norm = float(
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_accumulation()

        return StepResult(
            global_step=global_step,
            microbatches=microbatches,
            mean_loss=mean_loss,
            perplexity=math.exp(mean_loss),
            loss_bearing_tokens=token_sum,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            optimizer_stepped=True,
        )

    def state_dicts(self) -> tuple[dict, dict]:
        """Model and optimizer state for checkpointing."""
        return self.model.state_dict(), self.optimizer.state_dict()

    def load_state_dicts(self, model_state: dict, optimizer_state: dict) -> None:
        """Restore model and optimizer state after a checkpoint load."""
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        self._reset_accumulation()

    def _reset_accumulation(self) -> None:
        self._accumulated = 0
        self._loss_sum = 0.0
        self._token_sum = 0
