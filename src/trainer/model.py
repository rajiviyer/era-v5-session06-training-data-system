"""Tiny causal LM for the Session 6 demo run (P7-T01).

Deliberately small: 2 layers, d_model 128, learned position embeddings. The point of
this model is to make training *consumption* real (forward, backward, optimizer step,
checkpointable weights), not to produce a good language model. Everything that matters
for grading (masks, ledgers, recovery) is upstream and downstream of this file.

The model consumes the attention mask and position IDs that the batch builder already
computed and hashed, so the tensors the ledger fingerprints are the tensors the model
actually sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from config.schemas import ModelConfig

from .errors import TrainerError

# CPU-safety guard: SCOPE §2 caps the demo model at 4 layers.
MAX_LAYERS = 4


@dataclass(frozen=True)
class TinyModelConfig:
    """Model shape resolved from demo.yaml plus tokenizer and sequence length."""

    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    pad_token_id: int = 0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for field, value in (
            ("vocab_size", self.vocab_size),
            ("max_seq_len", self.max_seq_len),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("d_model", self.d_model),
            ("d_ff", self.d_ff),
        ):
            if value <= 0:
                raise TrainerError(f"TinyModelConfig.{field} must be > 0")
        if self.n_layers > MAX_LAYERS:
            raise TrainerError(
                f"TinyModelConfig.n_layers must be <= {MAX_LAYERS} to stay CPU-safe "
                f"(got {self.n_layers})"
            )
        if self.d_model % self.n_heads != 0:
            raise TrainerError(
                f"TinyModelConfig.d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise TrainerError("TinyModelConfig.dropout must be in [0, 1)")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise TrainerError("TinyModelConfig.pad_token_id must be a valid token ID")

    @classmethod
    def from_demo_config(
        cls,
        model: ModelConfig,
        *,
        vocab_size: int,
        max_seq_len: int,
        pad_token_id: int = 0,
    ) -> TinyModelConfig:
        """Build the model shape from demo.yaml, the frozen tokenizer, and seq_len."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layers=model.n_layers,
            n_heads=model.n_heads,
            d_model=model.d_model,
            d_ff=model.d_ff,
            dropout=model.dropout,
            pad_token_id=pad_token_id,
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention driven by an explicit per-batch attention mask."""

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden)
        query, key, value = qkv.split(d_model, dim=-1)

        # [B, S, D] -> [B, H, S, head_dim]
        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        query, key, value = split_heads(query), split_heads(key), split_heads(value)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attention_mask is [B, S, S] with 1 = attend; broadcast across heads.
        scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # Pad rows are fully masked, so their softmax is NaN. They carry no loss, so
        # zeroing them keeps gradients finite without changing any real position.
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = weights @ value
        context = context.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.dropout(self.projection(context))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention then GELU feed-forward."""

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        return hidden + self.ffn(self.ffn_norm(hidden))


class TinyCausalLM(nn.Module):
    """Decoder-only LM sized for CPU demo runs."""

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters, counted once even when embeddings are tied."""
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Return logits [B, S, vocab_size] for the given batch tensors."""
        if input_ids.dim() != 2:
            raise TrainerError("input_ids must be [batch, seq_len]")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            raise TrainerError(
                f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )
        if attention_mask.shape != (input_ids.shape[0], seq_len, seq_len):
            raise TrainerError("attention_mask must be [batch, seq_len, seq_len]")
        if position_ids.shape != input_ids.shape:
            raise TrainerError("position_ids must match input_ids shape")

        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))


def build_model(
    config: TinyModelConfig,
    *,
    seed: int,
    device: torch.device | None = None,
) -> TinyCausalLM:
    """Create a model with deterministic initialization from the run seed."""
    torch.manual_seed(seed)
    model = TinyCausalLM(config)
    if device is not None:
        model = model.to(device)
    return model
