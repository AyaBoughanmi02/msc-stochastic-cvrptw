"""Encoder normalization variants and their train/eval behaviour.

The diagnostic hypothesis is that BatchNorm's running-statistics differ between
train and eval, degrading greedy decoding under POMO. This pins the mechanism:
'batch' produces different outputs in train vs eval mode; 'layer' and 'none' do
not. Whichever norm wins is inherited by all variants via the model config.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp
from src.data.graph_build import build_graph
from src.models.egat_encoder import EGATEncoder
from src.models.factory import build_policy


def _graph():
    coords, demand, _ = generate_cvrp(4, 20, seed=1, device="cpu")
    return build_graph(coords, demand)


def test_all_norms_build_and_run() -> None:
    g = _graph()
    for norm in ("batch", "layer", "none"):
        policy = build_policy({"norm": norm})
        h, ge = policy.encoder(g.x, g.edge_index, g.edge_attr, g.batch)
        assert h.shape == (4 * 21, 128)
        assert ge.shape == (4, 128)


def test_unknown_norm_rejected() -> None:
    try:
        EGATEncoder(input_node_dim=3, input_edge_dim=1, norm="spam")
    except ValueError:
        return
    raise AssertionError("unknown norm should raise ValueError")


def test_batchnorm_has_train_eval_gap_others_dont() -> None:
    g = _graph()
    for norm, should_match in (("layer", True), ("none", True), ("batch", False)):
        torch.manual_seed(0)
        enc = EGATEncoder(input_node_dim=3, input_edge_dim=1, norm=norm)
        enc.train()
        h_train, _ = enc(g.x, g.edge_index, g.edge_attr, g.batch)
        enc.eval()
        h_eval, _ = enc(g.x, g.edge_index, g.edge_attr, g.batch)
        matches = torch.allclose(h_train, h_eval, atol=1e-5)
        assert matches is should_match, f"{norm}: train==eval was {matches}"


def test_default_norm_is_batch() -> None:
    # Back-compat: default config reproduces the original BatchNorm encoder.
    policy = build_policy()
    assert policy.encoder.norm == "batch"