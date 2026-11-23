"""A small decoder-only transformer in the style of Llama/Qwen.

Changes from the GPT-2-style model in lesson01/model.py, and why:
  - RMSNorm instead of LayerNorm: no mean subtraction or bias; cheaper, equally stable.
  - RoPE instead of learned position embeddings: position is encoded by rotating
    queries and keys, so attention scores depend on relative distance.
  - SwiGLU feed-forward instead of GELU MLP: a gated unit that trains better per parameter.
  - No biases in linear layers, and tied input/output embeddings.
  - A preallocated KV cache, so generating token n costs O(n), not O(n^2).

Shapes: B = batch, T = sequence length, C = d_model, H = heads, D = head_dim.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100  # targets with this value add nothing to the loss (used for SFT masking)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8_192
    context_length: int = 512
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ffn_hidden: int = 1_024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = ("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "ffn_hidden")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in float32: in fp16 the mean of squares can overflow.
        normalised = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalised.type_as(x) * self.weight


def rotary_tables(config: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of position * frequency, shape (context_length, head_dim // 2)."""
    half = config.head_dim // 2
    frequencies = config.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = torch.outer(torch.arange(config.context_length, dtype=torch.float32), frequencies)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates each (first-half, second-half) pair of x: (B, H, T, D) by its angle."""
    first, second = x.float().chunk(2, dim=-1)
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return rotated.type_as(x)


class KVCache:
    """Keys and values of every layer for the tokens generated so far.

    Preallocated to the full context so each step writes in place rather than
    concatenating, which would copy the whole cache every token.
    """

    def __init__(
        self, config: ModelConfig, batch_size: int, device: torch.device, dtype: torch.dtype
    ):
        shape = (
            config.n_layers,
            batch_size,
            config.n_heads,
            config.context_length,
            config.head_dim,
        )
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0
        self._capacity = config.context_length

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.length, self.length + k.size(2)
        if end > self._capacity:
            raise ValueError(f"KV cache full: {end} > context length {self._capacity}")
        self.keys[layer, :, :, start:end] = k
        self.values[layer, :, :, start:end] = v
        return self.keys[layer, :, :, :end], self.values[layer, :, :, :end]

    def advance(self, steps: int) -> None:
        self.length += steps


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q, k, v = (
            t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        has_history = cache is not None and cache.length > 0
        if has_history and length > 1:
            raise ValueError("With a non-empty KV cache, feed one token at a time")
        if cache is not None:
            k, v = cache.append(self.layer_index, k, v)

        # One new token may attend to everything cached, so it needs no mask;
        # otherwise the causal mask stops position t reading positions after t.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=not has_history,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.transpose(1, 2).contiguous().view(batch, length, channels))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down = nn.Linear(config.ffn_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config, layer_index)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None
    ) -> torch.Tensor:
        # Pre-norm residual stream: each sublayer adds an edit to x.
        x = x + self.dropout(self.attention(self.attention_norm(x), cos, sin, cache))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tied embeddings: reading and writing tokens share one matrix, saving
        # vocab_size * d_model parameters (3.1M of ~17M here).
        self.lm_head.weight = self.token_embedding.weight

        cos, sin = rotary_tables(config)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 scaling: each residual branch's output projection is shrunk so the
        # residual stream's variance does not grow with depth.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out.weight, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length = idx.shape
        start = cache.length if cache is not None else 0
        if start + length > self.config.context_length:
            raise ValueError(
                f"Sequence of {start + length} tokens exceeds context {self.config.context_length}"
            )

        cos = self.rope_cos[start : start + length]
        sin = self.rope_sin[start : start + length]
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.advance(length)
        logits = self.lm_head(self.final_norm(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token: int | None = None,
    ) -> torch.Tensor:
        """Samples up to `max_new_tokens`, reusing the KV cache between steps.
        temperature == 0 means greedy decoding."""
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if idx.size(1) + max_new_tokens > self.config.context_length:
            raise ValueError("prompt + max_new_tokens exceeds the context length")

        was_training = self.training
        self.eval()
        cache = KVCache(self.config, idx.size(0), idx.device, self.token_embedding.weight.dtype)
        logits, _ = self(idx, cache=cache)
        generated = idx
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    kth_best = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[
                        :, [-1]
                    ]
                    next_logits = next_logits.masked_fill(next_logits < kth_best, float("-inf"))
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            is_stopped = stop_token is not None and bool((next_token == stop_token).all())
            is_last_step = step == max_new_tokens - 1
            if is_stopped or is_last_step:
                break
            logits, _ = self(next_token, cache=cache)
        self.train(was_training)
        return generated
