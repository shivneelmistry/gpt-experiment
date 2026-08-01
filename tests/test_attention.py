"""Tests for attention."""

from __future__ import annotations

import math

import torch

from gpt_experiment.config import ModelConfig
from gpt_experiment.model import FeedForward, Head, MultiHeadAttention, PlainBlockLM

VOCAB = 65


def make_config(**kw: int) -> ModelConfig:
    return ModelConfig(vocab_size=VOCAB, n_embd=32, block_size=16, dropout=0.0, **kw)


def attention_weights(head: Head, x: torch.Tensor) -> torch.Tensor:
    """Recompute the weights a Head applies, for inspection."""
    _, time, _ = x.shape
    k, q = head.key(x), head.query(x)
    scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    scores = scores.masked_fill(head.tril[:time, :time] == 0, float("-inf"))
    return torch.softmax(scores, dim=-1)


def test_head_output_shape() -> None:
    cfg = make_config()
    head = Head(cfg, head_size=8)
    out = head(torch.randn(4, 16, cfg.n_embd))
    assert out.shape == (4, 16, 8)


def test_weights_are_lower_triangular() -> None:
    """No token may attend to a later one. This is what makes it a decoder."""
    cfg = make_config()
    head = Head(cfg, head_size=8)
    weights = attention_weights(head, torch.randn(2, 16, cfg.n_embd))

    upper = torch.triu(torch.ones(16, 16), diagonal=1).bool()
    assert torch.all(weights.masked_select(upper) == 0)


def test_weight_rows_sum_to_one(  ) -> None:
    cfg = make_config()
    head = Head(cfg, head_size=8)
    weights = attention_weights(head, torch.randn(2, 16, cfg.n_embd))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 16), atol=1e-5)


def test_first_token_attends_only_to_itself() -> None:
    cfg = make_config()
    head = Head(cfg, head_size=8)
    weights = attention_weights(head, torch.randn(2, 16, cfg.n_embd))
    assert torch.allclose(weights[:, 0, 0], torch.ones(2))


def test_future_tokens_do_not_change_the_past() -> None:
    """Changing token 10 must leave outputs for tokens 0-9 untouched."""
    torch.manual_seed(0)
    cfg = make_config()
    head = Head(cfg, head_size=8).eval()

    x = torch.randn(1, 16, cfg.n_embd)
    tampered = x.clone()
    tampered[:, 10:, :] = torch.randn(1, 6, cfg.n_embd)

    assert torch.allclose(head(x)[:, :10], head(tampered)[:, :10], atol=1e-6)


def test_untrained_loss_matches_uniform_guess() -> None:
    torch.manual_seed(0)
    model = PlainBlockLM(make_config())
    _, loss = model(torch.randint(VOCAB, (8, 16)), torch.randint(VOCAB, (8, 16)))

    assert loss is not None
    assert abs(loss.item() - math.log(VOCAB)) < 0.05


def test_generate_respects_block_size() -> None:
    """Generating past block_size must not index a missing position embedding."""
    torch.manual_seed(0)
    cfg = make_config()
    model = PlainBlockLM(cfg).eval()

    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=40)
    assert out.shape == (1, 41)
    assert int(out.max()) < VOCAB


def test_gradients_reach_every_parameter() -> None:
    torch.manual_seed(0)
    model = PlainBlockLM(make_config())
    _, loss = model(torch.randint(VOCAB, (4, 16)), torch.randint(VOCAB, (4, 16)))

    assert loss is not None
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_multihead_preserves_embedding_width() -> None:
    """Heads split n_embd and concatenate back, so downstream shapes are unchanged."""
    cfg = make_config(n_head=4)
    mha = MultiHeadAttention(cfg)
    out = mha(torch.randn(4, 16, cfg.n_embd))
    assert out.shape == (4, 16, cfg.n_embd)


def test_head_size_splits_evenly() -> None:
    cfg = make_config(n_head=4)
    assert cfg.head_size == cfg.n_embd // 4
    mha = MultiHeadAttention(cfg)
    assert len(mha.heads) == 4


def test_heads_learn_different_things() -> None:
    """Independent init means heads must not start out identical."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(make_config(n_head=4))
    first, second = mha.heads[0], mha.heads[1]
    assert not torch.allclose(first.query.weight, second.query.weight)


def test_multihead_is_still_causal() -> None:
    torch.manual_seed(0)
    cfg = make_config(n_head=4)
    mha = MultiHeadAttention(cfg).eval()

    x = torch.randn(1, 16, cfg.n_embd)
    tampered = x.clone()
    tampered[:, 10:, :] = torch.randn(1, 6, cfg.n_embd)

    assert torch.allclose(mha(x)[:, :10], mha(tampered)[:, :10], atol=1e-6)


def test_feedforward_preserves_shape() -> None:
    cfg = make_config()
    out = FeedForward(cfg)(torch.randn(4, 16, cfg.n_embd))
    assert out.shape == (4, 16, cfg.n_embd)


def test_feedforward_widens_then_narrows() -> None:
    cfg = make_config()
    linears = [m for m in FeedForward(cfg).net if isinstance(m, torch.nn.Linear)]
    assert linears[0].out_features == 4 * cfg.n_embd
    assert linears[1].out_features == cfg.n_embd


def test_feedforward_does_not_mix_tokens() -> None:
    """Each token is transformed alone. Changing one must not affect the others."""
    torch.manual_seed(0)
    cfg = make_config()
    ffwd = FeedForward(cfg).eval()

    x = torch.randn(1, 16, cfg.n_embd)
    tampered = x.clone()
    tampered[:, 5, :] = torch.randn(cfg.n_embd)

    out, out_tampered = ffwd(x), ffwd(tampered)
    assert torch.allclose(out[:, :5], out_tampered[:, :5], atol=1e-6)
    assert torch.allclose(out[:, 6:], out_tampered[:, 6:], atol=1e-6)


def test_feedforward_is_nonlinear() -> None:
    """Without GELU the two Linears would collapse into one."""
    torch.manual_seed(0)
    cfg = make_config()
    ffwd = FeedForward(cfg).eval()

    x = torch.randn(1, 4, cfg.n_embd)
    assert not torch.allclose(ffwd(2 * x), 2 * ffwd(x), atol=1e-4)
