from __future__ import annotations

import sys
from unittest.mock import MagicMock

# xformers cannot build on macOS with Apple Clang; stub it out for unit tests
for _mod in ("xformers", "xformers.ops"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import torch
import pytest

from femr.models.transformer import compute_attention_rollout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_attn(n_layers: int, n_heads: int, seq_len: int):
    """Each position attends uniformly to all positions in the sequence."""
    return [
        torch.ones(1, n_heads, seq_len, seq_len) / seq_len
        for _ in range(n_layers)
    ]


def _identity_attn(n_layers: int, n_heads: int, seq_len: int):
    """Each position attends only to itself (identity matrix)."""
    eye = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(1, n_heads, -1, -1)
    return [eye.clone() for _ in range(n_layers)]


def _block_diagonal_attn(n_layers: int, n_heads: int, lengths):
    """Block-diagonal attention: each segment attends uniformly within itself, zero cross-segment."""
    seq_len = sum(lengths)
    attn = []
    for _ in range(n_layers):
        A = torch.zeros(1, n_heads, seq_len, seq_len)
        start = 0
        for l in lengths:
            A[:, :, start:start + l, start:start + l] = 1.0 / l
            start += l
        attn.append(A)
    return attn


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_output_shapes():
    attn = _uniform_attn(3, 4, 6)
    label_indices = torch.tensor([1, 4])
    subject_lengths = torch.tensor([3, 3])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=2)
    assert pos.shape == (2, 2)
    assert scores.shape == (2, 2)


def test_top_k_larger_than_segment_pads_with_zeros():
    """When top_k > segment length, unfilled slots stay zero."""
    attn = _identity_attn(2, 2, 3)
    label_indices = torch.tensor([1])
    subject_lengths = torch.tensor([3])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=10)
    assert pos.shape == (1, 10)
    assert scores[0, 3:].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Correctness: identity attention
# ---------------------------------------------------------------------------

def test_identity_attention_self_attribution():
    """Identity attention at every layer: each label attributes entirely to itself."""
    attn = _identity_attn(3, 2, 5)
    label_indices = torch.tensor([1, 4])
    subject_lengths = torch.tensor([5])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=1)
    assert pos[0, 0].item() == 1
    assert pos[1, 0].item() == 4
    assert torch.allclose(scores[:, 0], torch.ones(2), atol=1e-5)


def test_multiple_labels_same_segment_identity():
    """Multiple labels in the same segment each get independent self-attribution."""
    attn = _identity_attn(2, 2, 6)
    label_indices = torch.tensor([0, 2, 5])
    subject_lengths = torch.tensor([6])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=1)
    assert pos[0, 0].item() == 0
    assert pos[1, 0].item() == 2
    assert pos[2, 0].item() == 5


# ---------------------------------------------------------------------------
# Correctness: segment isolation
# ---------------------------------------------------------------------------

def test_segment_isolation():
    """Top-k positions for a label must lie within its own patient segment."""
    attn = _block_diagonal_attn(2, 2, [5, 5])
    label_indices = torch.tensor([7])   # in segment 2 [5, 10)
    subject_lengths = torch.tensor([5, 5])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=3)
    assert (pos[0] >= 5).all(), "rollout leaked into segment 1"
    assert (pos[0] < 10).all()


def test_labels_in_different_segments():
    """Two labels from different segments should get positions only within their own segment."""
    attn = _block_diagonal_attn(2, 2, [4, 6])
    label_indices = torch.tensor([2, 8])   # seg1=[0,4), seg2=[4,10)
    subject_lengths = torch.tensor([4, 6])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=2)
    assert (pos[0] < 4).all(),  "label in seg1 got positions from seg2"
    assert (pos[1] >= 4).all(), "label in seg2 got positions from seg1"


def test_three_segments_middle_label():
    """Label in the middle segment should never reference positions from the other two."""
    attn = _block_diagonal_attn(2, 2, [3, 4, 3])
    label_indices = torch.tensor([5])   # middle segment [3, 7)
    subject_lengths = torch.tensor([3, 4, 3])
    pos, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=3)
    assert (pos[0] >= 3).all()
    assert (pos[0] < 7).all()


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------

def test_scores_non_negative():
    attn = _uniform_attn(2, 4, 6)
    label_indices = torch.tensor([1, 4])
    subject_lengths = torch.tensor([3, 3])
    _, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=2)
    assert (scores >= 0).all()


def test_top_k_scores_sum_leq_one():
    """top-k scores are a subset of a distribution summing to 1 over the segment."""
    attn = _uniform_attn(2, 4, 8)
    label_indices = torch.tensor([3])
    subject_lengths = torch.tensor([8])
    _, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=4)
    assert scores[0].sum().item() <= 1.0 + 1e-5


def test_full_top_k_scores_sum_to_one():
    """When top_k == segment length, scores must sum to exactly 1."""
    seg_len = 5
    attn = _uniform_attn(3, 2, seg_len)
    label_indices = torch.tensor([2])
    subject_lengths = torch.tensor([seg_len])
    _, scores = compute_attention_rollout(attn, label_indices, subject_lengths, top_k=seg_len)
    assert torch.allclose(scores[0].sum(), torch.tensor(1.0), atol=1e-5)
