"""Tests for blocks and stacking."""

from __future__ import annotations

import math

import torch

from gpt_experiment.config import ModelConfig
from gpt_experiment.model import Block, TransformerLM, TransformerNoResidualLM

VOCAB = 65


def make_config(n_layer: int = 4) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, n_embd=32, n_head=4, n_layer=n_layer,
        block_size=16, dropout=0.0,
    )


def first_layer_grad_norm(model: torch.nn.Module) -> float:
    """Gradient magnitude at the very first weight, after one backward pass."""
    _, loss = model(torch.randint(VOCAB, (4, 16)), torch.randint(VOCAB, (4, 16)))
    loss.backward()
    grad = model.token_embedding.weight.grad
    assert grad is not None
    return float(grad.norm())


def test_block_preserves_shape() -> None:
    cfg = make_config()
    out = Block(cfg)(torch.randn(4, 16, cfg.n_embd))
    assert out.shape == (4, 16, cfg.n_embd)


def test_block_is_causal() -> None:
    torch.manual_seed(0)
    cfg = make_config()
    block = Block(cfg).eval()

    x = torch.randn(1, 16, cfg.n_embd)
    tampered = x.clone()
    tampered[:, 10:, :] = torch.randn(1, 6, cfg.n_embd)

    assert torch.allclose(block(x)[:, :10], block(tampered)[:, :10], atol=1e-6)


def test_residual_carries_the_input_through() -> None:
    """With residuals the input is present in the output; without, it is gone."""
    torch.manual_seed(0)
    cfg = make_config()
    x = torch.randn(1, 16, cfg.n_embd)

    with_residual = Block(cfg, residual=True).eval()
    without = Block(cfg, residual=False).eval()

    # A residual block's output tracks its input; an unresidualed one does not.
    assert torch.corrcoef(
        torch.stack([x.flatten(), with_residual(x).flatten()])
    )[0, 1] > 0.5
    assert abs(
        torch.corrcoef(torch.stack([x.flatten(), without(x).flatten()]))[0, 1]
    ) < 0.5


def test_residuals_make_gradient_scale_independent_of_depth() -> None:
    """The reason residuals exist, measured.

    Not that gradients vanish without them -- pre-LayerNorm already prevents that.
    What they buy is a gradient scale at layer 0 that barely moves as the stack
    grows, so one learning rate works at any depth. Without them the scale swings
    by an order of magnitude between depths and is untunable.

    Measured across depths 2/8/32: residual stays near 0.5 throughout, while the
    ablation ranges 0.42 to 2.24.
    """
    depths = (2, 8, 32)

    def spread(cls: type) -> float:
        """Coefficient of variation of the layer-0 gradient across depths."""
        norms = []
        for depth in depths:
            torch.manual_seed(0)
            norms.append(first_layer_grad_norm(cls(make_config(n_layer=depth))))
            torch.manual_seed(0)
        values = torch.tensor(norms)
        return float(values.std() / values.mean())

    assert spread(TransformerLM) < spread(TransformerNoResidualLM) / 3


def test_untrained_loss_matches_uniform_guess() -> None:
    torch.manual_seed(0)
    model = TransformerLM(make_config())
    _, loss = model(torch.randint(VOCAB, (8, 16)), torch.randint(VOCAB, (8, 16)))

    assert loss is not None
    assert abs(loss.item() - math.log(VOCAB)) < 0.05


def test_layer_count_matches_config() -> None:
    model = TransformerLM(make_config(n_layer=6))
    assert sum(isinstance(m, Block) for m in model.body) == 6


def test_gradients_reach_every_parameter() -> None:
    torch.manual_seed(0)
    model = TransformerLM(make_config())
    _, loss = model(torch.randint(VOCAB, (4, 16)), torch.randint(VOCAB, (4, 16)))

    assert loss is not None
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
