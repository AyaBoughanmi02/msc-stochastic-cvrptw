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

# Tolerance so float error doesn't wrongly forbid a node that exactly fits its
# window / the shift (mirrors _CAP_EPS for capacity).
_TW_EPS = 1e-6


def tw_feasibility_mask(
    base_mask: Tensor,            # (B, N) bool, CVRP feasibility; True = infeasible
    time: Tensor,                 # (B,) float, current clock (shift-relative)
    travel_from_current: Tensor,  # (B, N) travel time current_node -> each node
    travel_to_depot: Tensor,      # (B, N) travel time each node -> depot
    time_windows: Tensor,         # (B, N, 2) [earliest, latest] service-start window
    service_time: Tensor,         # (B, N) service duration
    shift_end: float,             # per-route shift horizon
) -> Tensor:
    """Add hard time-window infeasibility to a CVRP mask. Returns (B, N) bool.

    Two conditions make a CUSTOMER infeasible as the next node:
      (a) window closed:  arrival > latest        (cannot arrive in time; waiting
          only helps when EARLY, never when late)
      (b) can't return:   serve_start + service + travel_to_depot > shift_end
          (serving it would make a within-shift return impossible; this single
          check subsumes "would exceed shift" and "couldn't get back in time")
    where arrival = time + travel_from_current and serve_start = max(arrival,
    earliest) (wait if early). The DEPOT is never time-masked: returning to the
    depot resets the clock (a new vehicle), so it is always the escape hatch that
    keeps at least one node feasible.
    """
    earliest = time_windows[..., 0]                       # (B, N)
    latest = time_windows[..., 1]                         # (B, N)
    arrival = time.unsqueeze(1) + travel_from_current     # (B, N)
    serve_start = torch.maximum(arrival, earliest)        # wait if early
    depart = serve_start + service_time                   # (B, N)

    window_closed = arrival > latest + _TW_EPS                          # (a)
    cant_return = depart + travel_to_depot > shift_end + _TW_EPS        # (b)
    tw_infeasible = window_closed | cant_return                        # (B, N)
    tw_infeasible[:, 0] = False   # never time-mask the depot (escape hatch)
    return base_mask | tw_infeasible