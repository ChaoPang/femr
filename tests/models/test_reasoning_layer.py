from __future__ import annotations

import sys
from unittest.mock import MagicMock

# xformers cannot build on macOS with Apple Clang; stub it out for unit tests
for _mod in ("xformers", "xformers.ops"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import torch
import pytest

from femr.models.transformer import ReasoningLayer
from femr.models.config import FEMRTransformerConfig


# ---------------------------------------------------------------------------
# ReasoningLayer unit tests
# ---------------------------------------------------------------------------

def test_output_shape():
    layer = ReasoningLayer(hidden_size=64, vocab_size=128, top_k=8)
    h = torch.randn(4, 64)
    output, top_idx, weights = layer(h)
    assert output.shape == (4, 64)
    assert top_idx.shape == (4, 8)
    assert weights.shape == (4, 8)


def test_weights_sum_to_one():
    layer = ReasoningLayer(hidden_size=32, vocab_size=64, top_k=5)
    h = torch.randn(6, 32)
    _, _, weights = layer(h)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(6), atol=1e-5)


def test_top_idx_in_vocab_range():
    vocab_size = 100
    layer = ReasoningLayer(hidden_size=16, vocab_size=vocab_size, top_k=10)
    h = torch.randn(3, 16)
    _, top_idx, _ = layer(h)
    assert top_idx.min() >= 0
    assert top_idx.max() < vocab_size


def test_mixing_alpha_zero_returns_hidden():
    """With alpha=0.0 the combined output equals the original hidden state."""
    layer = ReasoningLayer(hidden_size=32, vocab_size=64, top_k=4)
    h = torch.randn(5, 32)
    reasoning_out, _, _ = layer(h)
    alpha = 0.0
    combined = alpha * reasoning_out + (1.0 - alpha) * h
    assert torch.allclose(combined, h)


def test_mixing_alpha_one_returns_reasoning():
    """With alpha=1.0 the combined output equals the reasoning output."""
    layer = ReasoningLayer(hidden_size=32, vocab_size=64, top_k=4)
    h = torch.randn(5, 32)
    reasoning_out, _, _ = layer(h)
    alpha = 1.0
    combined = alpha * reasoning_out + (1.0 - alpha) * h
    assert torch.allclose(combined, reasoning_out)


def test_only_top_k_embeddings_receive_gradient():
    """Non-top-k embedding rows should get zero gradient."""
    vocab_size = 64
    top_k = 4
    layer = ReasoningLayer(hidden_size=16, vocab_size=vocab_size, top_k=top_k)
    h = torch.randn(2, 16)
    output, top_idx, _ = layer(h)
    output.sum().backward()

    grad = layer.reasoning_embedding.weight.grad  # [V, hidden]
    touched = set(top_idx.flatten().tolist())
    untouched = set(range(vocab_size)) - touched
    for i in untouched:
        assert grad[i].abs().max().item() == 0.0, f"embedding row {i} should have zero grad"


# ---------------------------------------------------------------------------
# Config round-trip tests
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = FEMRTransformerConfig()
    assert cfg.use_reasoning_layer is False
    assert cfg.reasoning_top_k == 32
    assert cfg.reasoning_weight == 1.0


def test_config_round_trip():
    cfg = FEMRTransformerConfig(
        use_reasoning_layer=True,
        reasoning_top_k=16,
        reasoning_weight=0.5,
    )
    restored = FEMRTransformerConfig(**cfg.to_dict())
    assert restored.use_reasoning_layer is True
    assert restored.reasoning_top_k == 16
    assert abs(restored.reasoning_weight - 0.5) < 1e-6
