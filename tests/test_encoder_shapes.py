"""Shape and attention-normalisation tests for the E-GAT encoder.

Builds tiny fully-connected graphs (incl. self-loops, mirroring Lei's dense
graph) and asserts every tensor shape, plus that per-head attention sums to 1
over each target node's incoming edges (catches softmax-wiring bugs).
"""

from __future__ import annotations

import torch

from src.models.egat_encoder import EGATEncoder, EGATLayer


def _fully_connected_edge_index(n: int) -> torch.Tensor:
    """All (i, j) pairs incl. self-loops -> (2, n*n). Row 0 = source, row 1 = target."""
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)  # j
    dst = idx.repeat(n)             # i
    return torch.stack([src, dst], dim=0)  # (2, n*n)


def test_encoder_single_graph_shapes() -> None:
    n, node_in, edge_in, hidden = 5, 3, 1, 128
    x = torch.rand(n, node_in)                              # (5, 3)
    edge_index = _fully_connected_edge_index(n)            # (2, 25)
    edge_attr = torch.rand(edge_index.size(1), edge_in)    # (25, 1)

    enc = EGATEncoder(node_in, edge_in, hidden_dim=hidden)
    node_emb, global_emb = enc(x, edge_index, edge_attr)

    assert node_emb.shape == (n, hidden)                   # (5, 128)
    assert global_emb.shape == (1, hidden)                 # (1, 128)
    assert torch.isfinite(node_emb).all()


def test_encoder_batched_graphs_shapes() -> None:
    n, b, node_in, edge_in, hidden = 5, 2, 3, 1, 128
    x = torch.rand(n * b, node_in)                         # (10, 3)
    ei0 = _fully_connected_edge_index(n)
    ei1 = _fully_connected_edge_index(n) + n               # offset node ids for graph 1
    edge_index = torch.cat([ei0, ei1], dim=1)             # (2, 50)
    edge_attr = torch.rand(edge_index.size(1), edge_in)    # (50, 1)
    batch = torch.tensor([0] * n + [1] * n)               # (10,)

    enc = EGATEncoder(node_in, edge_in, hidden_dim=hidden)
    node_emb, global_emb = enc(x, edge_index, edge_attr, batch)

    assert node_emb.shape == (n * b, hidden)              # (10, 128)
    assert global_emb.shape == (b, hidden)                # (2, 128)


def test_edge_dim_two_is_accepted() -> None:
    """input_edge_dim is a constructor arg: 1 for Stage-1, 2 for (mu, sigma^2)."""
    n, node_in, edge_in, hidden = 4, 3, 2, 64
    x = torch.rand(n, node_in)
    edge_index = _fully_connected_edge_index(n)
    edge_attr = torch.rand(edge_index.size(1), edge_in)   # (16, 2)

    enc = EGATEncoder(node_in, edge_in, hidden_dim=hidden)
    node_emb, _ = enc(x, edge_index, edge_attr)
    assert node_emb.shape == (n, hidden)


def test_attention_sums_to_one_per_target_and_head() -> None:
    n, node_in, edge_in, hidden, heads = 4, 3, 1, 32, 4
    x = torch.rand(n, node_in)
    edge_index = _fully_connected_edge_index(n)
    edge_attr = torch.rand(edge_index.size(1), edge_in)

    # Feed the layer a hidden-sized node tensor and hidden_edge-sized edge tensor,
    # as the encoder would after its projections.
    layer = EGATLayer(hidden_dim=hidden, edge_dim=edge_in, n_heads=heads)
    h = torch.rand(n, hidden)
    _, (ei, alpha) = layer(h, edge_index, edge_attr, return_attention_weights=True)

    assert alpha.shape == (edge_index.size(1), heads)     # (16, 4)

    # Sum attention per target node i, per head; each group must sum to ~1.
    target = ei[1]                                        # (E,)
    summed = torch.zeros(n, heads).index_add_(0, target, alpha)  # (n, heads)
    assert torch.allclose(summed, torch.ones(n, heads), atol=1e-5)


def test_gradients_flow() -> None:
    n, node_in, edge_in, hidden = 5, 3, 1, 64
    x = torch.rand(n, node_in)
    edge_index = _fully_connected_edge_index(n)
    edge_attr = torch.rand(edge_index.size(1), edge_in)

    enc = EGATEncoder(node_in, edge_in, hidden_dim=hidden)
    node_emb, _ = enc(x, edge_index, edge_attr)
    node_emb.sum().backward()

    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)