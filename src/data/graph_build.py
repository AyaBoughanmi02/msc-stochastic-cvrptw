# the data→graph bridge so the policy has a clean interface
"""Bridge from dense CVRP(TW) instances to the flat (PyG-style) inputs the encoder
expects.

The graph is fully connected including self-loops (Lei convention): every
ordered pair (a, b) is an edge, edge feature = Euclidean distance between the
two nodes' coordinates. Node feature = [x, y, demand] for CVRP, or
[x, y, demand, earliest, latest, service] for CVRPTW. Nodes are laid out
contiguously per instance so the encoder's flat output (B*N, hidden) can be
reshaped to (B, N, hidden).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class EncoderInputs(NamedTuple):
    """Flat graph tensors for EGATEncoder.forward."""

    x: Tensor           # (B*N, d)      node features (d=3 CVRP, d=6 CVRPTW)
    edge_index: Tensor  # (2, B*N*N)    row 0 = source, row 1 = target
    edge_attr: Tensor   # (B*N*N, F)    F=1 distance (Stage-1); F=2 (mu_bar, sigma_bar^2)
    batch: Tensor       # (B*N,)        graph id per node


def build_graph(coords: Tensor, demand: Tensor) -> EncoderInputs:
    """Build fully-connected (incl. self-loop) graph inputs (CVRP, 3-dim nodes).

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


def _flatten_dense_edge_attr(edge_attr: Tensor, b: int, n: int, device: torch.device) -> Tensor:
    """Flatten a dense per-edge feature tensor into build_graph's edge_index order.

    edge_attr: (B, N, N, F) indexed [instance, SOURCE, TARGET] -> (B*N*N, F)

    build_graph lays edges out as src = arange(N).repeat_interleave(N),
    tgt = arange(N).repeat(N), with instances stacked in order, so edge slot
    b*N*N + i*N + j is the edge FROM i TO j of instance b. A row-major reshape of a
    (B, N, N, F) tensor produces exactly that ordering -- but a TRANSPOSED input would
    flatten without error into a wrong-but-plausible graph, so the [source, target]
    convention is part of the contract and is covered by a dedicated test.
    """
    if edge_attr.dim() != 4:
        raise ValueError(
            f"edge_attr must be dense (B, N, N, F), got {tuple(edge_attr.shape)}"
        )
    if edge_attr.shape[:3] != (b, n, n):
        raise ValueError(
            f"edge_attr has shape {tuple(edge_attr.shape)}; expected leading dims "
            f"({b}, {n}, {n}) to match coords"
        )
    if edge_attr.device != device:
        raise ValueError(
            f"edge_attr is on {edge_attr.device} but coords are on {device}"
        )
    return edge_attr.reshape(b * n * n, edge_attr.size(-1))    # (B*N*N, F)


def build_graph_tw(
    coords: Tensor,        # (B, N, 2)
    demand: Tensor,        # (B, N)
    time_windows: Tensor,  # (B, N, 2)  [earliest, latest]
    service_time: Tensor,  # (B, N)
    edge_attr: Tensor | None = None,  # (B, N, N, F) dense [instance, source, target]
) -> EncoderInputs:
    """CVRPTW graph inputs: same fully-connected graph as build_graph, but with 6-dim
    node features [x, y, demand, earliest, latest, service]. Reuses build_graph for
    edges/batch (single source of truth) and only replaces the node feature x, so
    build_graph itself is untouched.

    edge_attr (optional) replaces the 1-dim distance edge feature with an F-dim dense
    one. Variants B/C/D pass horizon_average(bank) -> (B, N, N, 2), i.e. the
    horizon-averaged (mu_bar, sigma_bar^2) travel-time moments; the encoder is then
    built with input_edge_dim=2. Leaving it None reproduces variant A exactly.

    returns EncoderInputs with x: (B*N, 6); edge_index/batch identical to the CVRP
    graph; edge_attr (B*N*N, 1) distance by default, or (B*N*N, F) if overridden.
    """
    b, n, _ = coords.shape
    base = build_graph(coords, demand)                 # edges + batch (+ 3-dim x, discarded)
    x = torch.cat(
        [coords, demand.unsqueeze(-1), time_windows, service_time.unsqueeze(-1)],
        dim=-1,
    ).reshape(b * n, 6)                                 # [x, y, demand, earliest, latest, service]
    if edge_attr is None:
        return base._replace(x=x)                       # same edges/edge_attr/batch, new x
    flat = _flatten_dense_edge_attr(edge_attr, b, n, coords.device)   # (B*N*N, F)
    return base._replace(x=x, edge_attr=flat)