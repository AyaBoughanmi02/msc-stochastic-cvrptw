# the data→graph bridge so the policy has a clean interface
"""Bridge from dense CVRP instances to the flat (PyG-style) inputs the encoder
expects.

The graph is fully connected including self-loops (Lei convention): every
ordered pair (a, b) is an edge, edge feature = Euclidean distance between the
two nodes' coordinates. Node feature = [x, y, demand]. Nodes are laid out
contiguously per instance so the encoder's flat output (B*N, hidden) can be
reshaped to (B, N, hidden).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class EncoderInputs(NamedTuple):
    """Flat graph tensors for EGATEncoder.forward."""

    x: Tensor           # (B*N, 3)      node features [x, y, demand]
    edge_index: Tensor  # (2, B*N*N)    row 0 = source, row 1 = target
    edge_attr: Tensor   # (B*N*N, 1)    Euclidean distance (Stage-1 edge feature)
    batch: Tensor       # (B*N,)        graph id per node


def build_graph(coords: Tensor, demand: Tensor) -> EncoderInputs:
    """Build fully-connected (incl. self-loop) graph inputs.

    coords: (B, N, 2)
    demand: (B, N)     demand[:, 0] = 0 (depot)
    """
    b, n, _ = coords.shape
    dev = coords.device

    # Node features [x, y, demand], contiguous per instance -> (B*N, 3).
    x = torch.cat([coords, demand.unsqueeze(-1)], dim=-1).reshape(b * n, 3)

    # All ordered (source, target) pairs within an instance.
    src = torch.arange(n, device=dev).repeat_interleave(n)  # (N*N,)
    tgt = torch.arange(n, device=dev).repeat(n)             # (N*N,)
    offsets = (torch.arange(b, device=dev) * n).view(b, 1)  # (B, 1)
    src_b = (src.unsqueeze(0) + offsets).reshape(-1)        # (B*N*N,)
    tgt_b = (tgt.unsqueeze(0) + offsets).reshape(-1)        # (B*N*N,)
    edge_index = torch.stack([src_b, tgt_b], dim=0)         # (2, B*N*N)

    # Edge feature = distance between endpoint coordinates.
    coords_flat = coords.reshape(b * n, 2)                  # (B*N, 2)
    dist = (coords_flat[src_b] - coords_flat[tgt_b]).norm(dim=-1, keepdim=True)  # (E, 1)

    batch = torch.arange(b, device=dev).repeat_interleave(n)  # (B*N,)
    return EncoderInputs(x, edge_index, dist, batch)

def build_graph_tw(coords, demand, time_windows, service_time):
    """CVRPTW graph inputs: same fully-connected graph + distance edges as
    build_graph, but with 6-dim node features [x, y, demand, earliest, latest,
    service]. Reuses build_graph for edges/batch (single source of truth) and only
    replaces the node feature x, so build_graph itself is untouched.

    coords:       (B, N, 2)
    demand:       (B, N)
    time_windows: (B, N, 2)  [earliest, latest]
    service_time: (B, N)
    returns EncoderInputs with x: (B*N, 6); edge_index/edge_attr/batch identical
    to the CVRP graph.
    """
    b, n, _ = coords.shape
    base = build_graph(coords, demand)                 # edges + batch (+ 3-dim x, discarded)
    x = torch.cat(
        [coords, demand.unsqueeze(-1), time_windows, service_time.unsqueeze(-1)],
        dim=-1,
    ).reshape(b * n, 6)                                 # [x, y, demand, earliest, latest, service]
    return base._replace(x=x)                           # same edges/edge_attr/batch, new x