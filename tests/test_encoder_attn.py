"""Encoder attention granularity: scalar (locked) vs channel (Lei diagnostic).

Channel-wise reproduces Lei's released GatConv (a full out_channels attention
vector per edge) and is used only to measure the expressiveness cost of the
scalar-per-head simplification; variants A-D use scalar. Default must stay scalar
so the locked model is unchanged.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp
from src.data.graph_build import build_graph
from src.models.egat_encoder import EGATLayer
from src.models.factory import build_policy


def _graph():
    coords, demand, _ = generate_cvrp(4, 20, seed=1, device="cpu")
    return build_graph(coords, demand)


def test_both_attn_modes_build_and_run() -> None:
    g = _graph()
    for attn in ("scalar", "channel"):
        policy = build_policy({"attn": attn})
        h, ge = policy.encoder(g.x, g.edge_index, g.edge_attr, g.batch)
        assert h.shape == (4 * 21, 128)
        assert ge.shape == (4, 128)


def test_channel_has_more_capacity_than_scalar() -> None:
    scalar = sum(p.numel() for p in build_policy({"attn": "scalar"}).encoder.parameters())
    channel = sum(p.numel() for p in build_policy({"attn": "channel"}).encoder.parameters())
    assert channel > scalar  # channel-wise attention carries more parameters


def test_unknown_attn_rejected() -> None:
    try:
        EGATLayer(hidden_dim=128, edge_dim=16, n_heads=8, attn="bogus")
    except ValueError:
        return
    raise AssertionError("unknown attn should raise ValueError")


def test_default_attn_is_scalar() -> None:
    assert build_policy().encoder.attn == "scalar"