"""Models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gpt_experiment.cache import KVCache
from gpt_experiment.config import ModelConfig
from gpt_experiment.masking import IGNORE_INDEX

__all__ = [
    "MODELS",
    "BidirectionalLM",
    "BigramLM",
    "Block",
    "CachedBlock",
    "CachedTransformerLM",
    "CausalSelfAttention",
    "ContextLM",
    "FeedForward",
    "Head",
    "LanguageModel",
    "MultiHeadAttention",
    "MultiHeadAttentionLM",
    "PlainBlockLM",
    "SingleHeadAttentionLM",
    "TransformerLM",
    "TransformerNoResidualLM",
]

_FFWD_EXPANSION = 4  # convention since the original transformer paper


class LanguageModel(nn.Module):
    """Predict next-token logits, and sample from them."""

    config: ModelConfig

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        """std 0.02. Torch defaults are too wide and the first steps diverge."""
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    @staticmethod
    def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
        # cross_entropy takes 2-D, so flatten batch and time together.
        # IGNORE_INDEX is how MLM scores only masked positions; next-token targets
        # never contain it.
        batch, time, vocab = logits.shape
        return F.cross_entropy(
            logits.view(batch * time, vocab),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """Append max_new_tokens sampled continuations to idx."""
        for _ in range(max_new_tokens):
            # positions past block_size have no embedding
            logits, _ = self(idx[:, -self.config.block_size :])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # last position predicts next
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


class BigramLM(LanguageModel):
    """v1 -- one token of context. A lookup table, not a transformer.

    Exists to give later versions a number to beat.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # row per token, one score per possible next token
        self.token_embedding = nn.Embedding(config.vocab_size, config.vocab_size)
        self.apply(self.init_weights)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """idx is (batch, time). Returns (logits, loss); loss is None without targets."""
        logits = self.token_embedding(idx)  # (batch, time, vocab)
        if targets is None:
            return logits, None
        return logits, self.cross_entropy(logits, targets)


class Head(nn.Module):
    """One attention head.

    Each token emits a question (query) and a label (key). Matching them scores how
    much to take from every other token's content (value).
    """

    tril: Tensor  # buffer; annotated so mypy sees a Tensor

    def __init__(
        self, config: ModelConfig, head_size: int, causal: bool = True
    ) -> None:
        super().__init__()
        self.causal = causal
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # buffer, not parameter: moves with .to(device), never trained
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x: Tensor) -> Tensor:
        """(batch, time, n_embd) -> (batch, time, head_size)."""
        _, time, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        # dot products grow like sqrt(d); unscaled, softmax saturates and the
        # gradient dies
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        # the entire decoder/encoder split: predicting the future must not see it,
        # reconstructing a gap needs both sides
        if self.causal:
            scores = scores.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out: Tensor = weights @ self.value(x)
        return out


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated.

    head_size shrinks to match, so n heads cost what one wide head did. Roles are
    learned, not assigned.
    """

    def __init__(self, config: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            Head(config, config.head_size, causal) for _ in range(config.n_head)
        )
        # mixes the heads; without it they stay in separate lanes
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(batch, time, n_embd) -> (batch, time, n_embd)."""
        joined = torch.cat([head(x) for head in self.heads], dim=-1)
        out: Tensor = self.dropout(self.proj(joined))
        return out


class FeedForward(nn.Module):
    """Per-token processing, no looking around.

    Attention moves information between tokens; this transforms each one alone.
    About two thirds of a real model's parameters live here.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = _FFWD_EXPANSION * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """(batch, time, n_embd) -> (batch, time, n_embd)."""
        out: Tensor = self.net(x)
        return out


class Block(nn.Module):
    """Look around, then think, with a residual path around each.

    The additions are what make depth work: each block contributes to a running
    representation rather than replacing it. Norm goes before each sublayer, not
    after -- post-norm is markedly harder to train.

    residual=False is the ablation.
    """

    def __init__(
        self, config: ModelConfig, residual: bool = True, causal: bool = True
    ) -> None:
        super().__init__()
        self.residual = residual
        self.attn = MultiHeadAttention(config, causal)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """(batch, time, n_embd) -> (batch, time, n_embd)."""
        if not self.residual:  # ablation path
            ablated: Tensor = self.ffwd(self.ln2(self.attn(self.ln1(x))))
            return ablated
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ContextLM(LanguageModel):
    """Embed, run a body, project to vocabulary.

    Subclasses supply only the body -- the one thing that changes between versions.
    """

    def build_body(self, config: ModelConfig) -> nn.Module:
        raise NotImplementedError

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # attention is order-blind, so position is added separately
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.body = self.build_body(config)
        self.head_out = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self.init_weights)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _, time = idx.shape
        positions = torch.arange(time, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.body(x)
        logits = self.head_out(x)

        if targets is None:
            return logits, None
        return logits, self.cross_entropy(logits, targets)


class SingleHeadAttentionLM(ContextLM):
    """v2 -- one attention head. First model that can see context at all."""

    def build_body(self, config: ModelConfig) -> nn.Module:
        return Head(config, config.n_embd)


class MultiHeadAttentionLM(ContextLM):
    """v3 -- four heads in parallel. Same parameter count as v2."""

    def build_body(self, config: ModelConfig) -> nn.Module:
        return MultiHeadAttention(config)


class PlainBlockLM(ContextLM):
    """v4 -- attention plus feed-forward: a block without residuals or layer norm."""

    def build_body(self, config: ModelConfig) -> nn.Module:
        return nn.Sequential(MultiHeadAttention(config), FeedForward(config))


class TransformerLM(ContextLM):
    """v5 -- n_layer proper blocks stacked. The first version that is a transformer."""

    def build_body(self, config: ModelConfig) -> nn.Module:
        return nn.Sequential(
            *(Block(config) for _ in range(config.n_layer)),
            nn.LayerNorm(config.n_embd),  # normalise before the vocabulary projection
        )


class TransformerNoResidualLM(ContextLM):
    """Ablation -- v5 with the residual paths removed, to show what they buy."""

    def build_body(self, config: ModelConfig) -> nn.Module:
        return nn.Sequential(
            *(Block(config, residual=False) for _ in range(config.n_layer)),
            nn.LayerNorm(config.n_embd),
        )


class CausalSelfAttention(nn.Module):
    """Fused multi-head attention with optional KV caching and grouped queries.

    One Linear produces Q, K and V at once: faster than n separate modules, and it
    gives keys and values one contiguous layout, which is what makes caching
    practical.

    With n_kv_head < n_head this is grouped-query attention. Several query heads
    share one key/value head, so the cache shrinks by queries_per_kv while the
    query side keeps its full width. Cache size is the binding constraint on
    context length, so the trade is usually worth it.
    """

    tril: Tensor

    def __init__(self, config: ModelConfig, layer: int) -> None:
        super().__init__()
        self.config = config
        self.layer = layer
        kv_width = config.kv_heads * config.head_size
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv = nn.Linear(config.n_embd, 2 * kv_width, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        batch, time, n_embd = x.shape
        cfg = self.config
        head_size, kv_heads = cfg.head_size, cfg.kv_heads

        q = self.q(x).view(batch, time, cfg.n_head, head_size).transpose(1, 2)
        k, v = self.kv(x).split(kv_heads * head_size, dim=2)
        k = k.view(batch, time, kv_heads, head_size).transpose(1, 2)
        v = v.view(batch, time, kv_heads, head_size).transpose(1, 2)

        offset = 0
        if cache is not None:
            offset = cache.length  # tokens already stored
            # cache before expanding, so only kv_heads copies are stored
            k, v = cache.append(self.layer, k, v)

        # Group the query heads that share a kv head and add a broadcast axis to
        # k and v, rather than materialising expanded copies with repeat_interleave.
        # The copy costs more per step than the shared cache saves.
        group = cfg.queries_per_kv
        if group > 1:
            q = q.view(batch, kv_heads, group, time, head_size)
            k, v = k.unsqueeze(2), v.unsqueeze(2)

        scores = q @ k.transpose(-2, -1) * head_size**-0.5

        # query row i sits at absolute position offset+i. Without the offset a
        # cached decode masks away the history it just looked up.
        mask = self.tril[offset : offset + time, : offset + time]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = self.dropout(F.softmax(scores, dim=-1))
        out = weights @ v
        if group > 1:
            out = out.view(batch, cfg.n_head, time, head_size)
        out = out.transpose(1, 2).contiguous().view(batch, time, n_embd)
        projected: Tensor = self.dropout(self.proj(out))
        return projected


class CachedBlock(nn.Module):
    """A Block using fused attention, able to pass a cache through."""

    def __init__(self, config: ModelConfig, layer: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), cache)
        x = x + self.ffwd(self.ln2(x))
        return x


class CachedTransformerLM(ContextLM):
    """v6 -- same architecture as v5, but generation can reuse past keys and values.

    Blocks are held in a ModuleList rather than the inherited Sequential body,
    because each one needs the cache threaded through its forward call.
    """

    body: nn.ModuleList  # narrows the inherited nn.Module to something iterable

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # LayerNorm has no Linear or Embedding weights, so init_weights skips it
        self.ln_f = nn.LayerNorm(config.n_embd)

    def build_body(self, config: ModelConfig) -> nn.Module:
        return nn.ModuleList(CachedBlock(config, i) for i in range(config.n_layer))

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _, time = idx.shape
        # with a cache, positions start where the cache leaves off
        offset = cache.length if cache is not None else 0
        positions = torch.arange(offset, offset + time, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.body:
            x = block(x, cache)
        logits = self.head_out(self.ln_f(x))

        if cache is not None:
            cache.advance(time)  # once per pass, after all layers have written

        if targets is None:
            return logits, None
        return logits, self.cross_entropy(logits, targets)

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        use_cache: bool = True,
    ) -> Tensor:
        """Append tokens. With use_cache=False this degrades to the v5 behaviour."""
        if not use_cache:
            return super().generate(idx, max_new_tokens)

        cache = KVCache(
            self.config,
            batch_size=idx.shape[0],
            device=idx.device,
            dtype=self.token_embedding.weight.dtype,
        )
        # prefill the prompt in one pass, then decode one token at a time
        chunk = idx[:, -self.config.block_size :]
        for _ in range(max_new_tokens):
            logits, _ = self(chunk, cache=cache)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            chunk = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, chunk], dim=1)
        return idx


class BidirectionalLM(ContextLM):
    """v7 -- an encoder. v5 with the causal mask off, trained on masked
    reconstruction. One flag of architecture; the objective is the real change.
    """

    def build_body(self, config: ModelConfig) -> nn.Module:
        return nn.Sequential(
            *(Block(config, causal=False) for _ in range(config.n_layer)),
            nn.LayerNorm(config.n_embd),
        )

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        raise NotImplementedError("an encoder has no next token; use fill_masks")

    @torch.no_grad()
    def fill_masks(self, idx: Tensor, mask_token_id: int) -> Tensor:
        """Replace every [MASK] with the model's best guess."""
        logits, _ = self(idx)
        return torch.where(idx == mask_token_id, logits.argmax(dim=-1), idx)


# Version keys preserve build order.
MODELS: dict[str, type[LanguageModel]] = {
    "v1": BigramLM,
    "v2": SingleHeadAttentionLM,
    "v3": MultiHeadAttentionLM,
    "v4": PlainBlockLM,
    "v5": TransformerLM,
    "v5-noresid": TransformerNoResidualLM,
    "v6": CachedTransformerLM,
    "v7": BidirectionalLM,
}
