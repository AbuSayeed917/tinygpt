"""Steps 2-4: the GPT itself.

Data flow for one batch (B = batch size, T = sequence length, C = n_embd):

    token ids      (B, T)          integers
      -> embeddings (B, T, C)      each token becomes a vector of C numbers
      -> N x Block  (B, T, C)      tokens exchange information (attention)
                                   and each token is processed alone (MLP)
      -> head       (B, T, vocab)  a score ("logit") for every possible next token
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # max number of tokens the model can look back over (context window)
    n_embd: int = 128       # width of every token vector
    n_head: int = 4         # attention heads per layer
    n_layer: int = 4        # how many blocks are stacked
    dropout: float = 0.1    # randomly zero some activations during training to fight overfitting


class CausalSelfAttention(nn.Module):
    """Each token builds a weighted average of the tokens before it.

    Every token produces three vectors:
      query (q): "what am I looking for?"
      key   (k): "what do I contain?"
      value (v): "what will I hand over if you pick me?"
    score(i, j) = q_i . k_j  -> how relevant token j is to token i.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # One matrix produces q, k and v together; cheaper than three separate ones.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # Lower-triangular matrix: row i has ones up to column i. A buffer is saved
        # with the model but is not a trainable weight.
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # Split C into n_head independent heads so different heads can learn
        # different relationships (one tracks rhyme, another who is speaking...).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Dividing by sqrt(head_dim) keeps scores from growing with vector size,
        # otherwise softmax saturates and gradients vanish.
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)

        # -inf becomes 0 after softmax: token i can never see token j > i.
        # Without this, training would "cheat" by reading the answer.
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v                                   # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # glue heads back together
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Per-token processing. Most of a model's parameters (and much of its
    stored "knowledge") live in these layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """attention = tokens talk to each other; MLP = each token thinks alone."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # "x + ..." is the residual connection: each layer adds a small edit to
        # a running stream instead of replacing it, which keeps deep stacks trainable.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Attention itself has no sense of order, so position gets its own vector.
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        positions = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        if targets is None:
            return logits, None
        # Cross-entropy = -log(probability the model gave to the correct next token),
        # averaged over every position in every sequence of the batch.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressive sampling: predict one token, append it, repeat.
        This loop is exactly why chatbots stream their answer word by word."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature  # only the last position predicts the future
            if top_k is not None:
                # Ignore everything outside the k most likely tokens.
                kth_best = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
