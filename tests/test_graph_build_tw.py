"""Tests for build_graph_tw (CVRPTW 6-dim node features).

Guards the seam most likely to break silently: the node-feature width and column
order must be exactly [x, y, demand, earliest, latest, service], and the edges /
batch must be identical to the CVRP graph (build_graph stays the single source).
"""

from __future__ import annotations

import torch

from src.data.graph_build import build_graph, build_graph_tw


def _instance():
    b, n = 4, 6
    coords = torch.rand(b, n, 2)
    demand = torch.rand(b, n)
    demand[:, 0] = 0.0
    time_windows = torch.rand(b, n, 2)
    service = torch.full((b, n), 0.2)
    service[:, 0] = 0.0
    return coords, demand, time_windows, service


def test_cvrp_build_graph_still_3dim() -> None:
    coords, demand, _, _ = _instance()
    g = build_graph(coords, demand)
    assert g.x.shape == (4 * 6, 3)                    # CVRP path untouched


def test_tw_node_features_shape_and_columns() -> None:
    coords, demand, tw, service = _instance()
    g = build_graph_tw(coords, demand, tw, service)
    b, n = 4, 6
    assert g.x.shape == (b * n, 6)
    x = g.x.reshape(b, n, 6)
    assert torch.allclose(x[..., 0:2], coords)        # [x, y]
    assert torch.allclose(x[..., 2], demand)          # demand
    assert torch.allclose(x[..., 3:5], tw)            # [earliest, latest]
    assert torch.allclose(x[..., 5], service)         # service


def test_tw_edges_and_batch_match_cvrp() -> None:
    coords, demand, tw, service = _instance()
    cvrp = build_graph(coords, demand)
    twg = build_graph_tw(coords, demand, tw, service)
    assert torch.equal(twg.edge_index, cvrp.edge_index)
    assert torch.equal(twg.edge_attr, cvrp.edge_attr)  # distance edges, same as CVRP
    assert torch.equal(twg.batch, cvrp.batch)