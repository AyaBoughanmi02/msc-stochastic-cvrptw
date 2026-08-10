"""Masking and cost primitives for the CVRP environment.

Pure functions, testable in isolation. Mask convention (shared with the pointer
decoder): mask is (B, N) boolean, True = infeasible (forbidden next node).
Depot is index 0 and has demand 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Tolerance so float error doesn't wrongly forbid a node that exactly fits.
_CAP_EPS = 1e-6


def feasibility_mask(
    visited: Tensor,             # (B, N) bool, customers served (depot always False)
    demand: Tensor,              # (B, N) float, demand[:, 0] = 0
    remaining_capacity: Tensor,  # (B,) float
    current_node: Tensor,        # (B,) long
    done: Tensor | None = None,  # (B,) bool, all customers served
) -> Tensor:
    """Return (B, N) bool mask; True = infeasible as the next node.

    A customer is infeasible if already visited or its demand exceeds remaining
    capacity. The depot is infeasible only while the vehicle is currently at the
    depot (prevents depot->depot). Finished trajectories (done) are pinned to the
    depot so at least one node stays feasible (avoids an all-masked softmax).
    """
    over_capacity = demand > remaining_capacity.unsqueeze(1) + _CAP_EPS  # (B, N)
    mask = visited | over_capacity                                        # (B, N)
    at_depot = current_node == 0                                          # (B,)
    mask[:, 0] = at_depot
    if done is not None:
        mask[done] = True         # forbid everything...
        mask[done, 0] = False     # ...except idling at the depot
    return mask


def tour_length(coords: Tensor, actions: Tensor) -> Tensor:
    """Euclidean length of each tour, including depot start and return.

    coords:  (B, N, 2)
    actions: (B, T) long, sequence of visited node indices (depot may recur)
    returns: (B,) float
    """
    b = actions.size(0)
    depot = actions.new_zeros(b, 1)                    # (B, 1) depot index 0
    seq = torch.cat([depot, actions, depot], dim=1)    # (B, T+2)  depot ... depot
    idx = seq.unsqueeze(-1).expand(-1, -1, 2)          # (B, T+2, 2)
    coord_seq = torch.gather(coords, 1, idx)           # (B, T+2, 2)
    steps = coord_seq[:, 1:] - coord_seq[:, :-1]       # (B, T+1, 2)
    return steps.norm(dim=-1).sum(dim=1)               # (B,)