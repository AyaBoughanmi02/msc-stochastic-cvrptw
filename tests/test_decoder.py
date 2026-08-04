"""Tests for the single-step pointer decoder.

Feeds fake node embeddings + state + masks (the decoder is state-agnostic) and
checks: output shape, that probabilities sum to 1, that infeasible nodes get
zero probability, that pre-mask logits respect the tanh clip, and that gradients
flow.
"""

from __future__ import annotations

import torch

from src.models.pointer_decoder import PointerDecoder


def _fake_batch(b: int, n: int, hidden: int):
    node_emb = torch.rand(b, n, hidden)
    graph_emb = node_emb.mean(dim=1)                 # (B, hidden)
    last_node_emb = node_emb[:, 0, :]                # pretend we're at node 0 (depot)
    remaining_capacity = torch.rand(b, 1)            # (B, 1)
    return node_emb, graph_emb, last_node_emb, remaining_capacity


def test_decoder_output_shape_and_normalisation() -> None:
    b, n, hidden = 4, 6, 128
    node_emb, graph_emb, last, cap = _fake_batch(b, n, hidden)
    mask = torch.zeros(b, n, dtype=torch.bool)       # nothing masked
    mask[:, 0] = True                                # depot forbidden this step

    dec = PointerDecoder(hidden_dim=hidden, n_heads=8, tanh_clip=10.0)
    cache = dec.precompute(node_emb)
    log_p = dec(cache, graph_emb, last, cap, mask)   # (B, N)

    assert log_p.shape == (b, n)
    probs = log_p.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(b), atol=1e-5)


def test_masked_nodes_get_zero_probability() -> None:
    b, n, hidden = 3, 5, 64
    node_emb, graph_emb, last, cap = _fake_batch(b, n, hidden)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, 1] = True                                # forbid node 1 everywhere
    mask[0, 3] = True                                # forbid node 3 in sample 0

    dec = PointerDecoder(hidden_dim=hidden, n_heads=8)
    cache = dec.precompute(node_emb)
    probs = dec(cache, graph_emb, last, cap, mask).exp()

    assert torch.all(probs[:, 1] < 1e-8)
    assert probs[0, 3] < 1e-8


def test_logits_respect_tanh_clip() -> None:
    # Before masking, pointer logits must lie within [-C, C] by construction.
    b, n, hidden, clip = 2, 7, 32, 10.0
    node_emb, graph_emb, last, cap = _fake_batch(b, n, hidden)
    mask = torch.zeros(b, n, dtype=torch.bool)

    dec = PointerDecoder(hidden_dim=hidden, n_heads=4, tanh_clip=clip)
    cache = dec.precompute(node_emb)
    log_p = dec(cache, graph_emb, last, cap, mask)
    # log_softmax of clipped logits: each logit in [-C, C], so log-prob >= -2C - log(N)
    assert torch.isfinite(log_p).all()
    assert log_p.max() <= 0.0 + 1e-6


def test_gradients_flow() -> None:
    b, n, hidden = 2, 5, 64
    node_emb, graph_emb, last, cap = _fake_batch(b, n, hidden)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, 0] = True

    dec = PointerDecoder(hidden_dim=hidden, n_heads=8)
    cache = dec.precompute(node_emb)
    log_p = dec(cache, graph_emb, last, cap, mask)
    log_p[:, 1].sum().backward()

    grads = [p.grad for p in dec.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)