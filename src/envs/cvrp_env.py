"""Batched CVRP environment (deterministic Euclidean cost), Stage-1.

Manages rollout state for B instances in parallel, produces feasibility masks in
the (B, N) True=infeasible convention, and computes tour-length cost. The
autoregressive loop and the policy live elsewhere; this class is the state
machine + masking only.

POMO multi-start: replicate each instance across trajectories in the batch and
pass `force_first` to reset() so trajectory k is forced to visit customer k
first (Kwon et al. 2020). This env does not know about POMO otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.masking import feasibility_mask, tour_length


class CVRPEnv:
    """CVRP state machine for a batch of instances."""

    def __init__(
        self,
        coords: Tensor,             # (B, N, 2)
        demand: Tensor,             # (B, N) float, demand[:, 0] = 0
        capacity: Tensor | float,   # (B,) or scalar; same units as demand
    ) -> None:
        b, n, _ = coords.shape
        self.coords = coords
        self.demand = demand
        self.batch_size = b
        self.num_nodes = n
        if isinstance(capacity, (int, float)):
            capacity = torch.full((b,), float(capacity), device=coords.device)
        self.capacity = capacity  # (B,)

        # State tensors (no autograd; RL treats these as constants).
        self.visited: Tensor
        self.current_node: Tensor
        self.remaining_capacity: Tensor
        self._reset_state()

    def _reset_state(self) -> None:
        b, n = self.batch_size, self.num_nodes
        dev = self.coords.device
        self.visited = torch.zeros(b, n, dtype=torch.bool, device=dev)   # (B, N)
        self.current_node = torch.zeros(b, dtype=torch.long, device=dev)  # (B,) depot
        self.remaining_capacity = self.capacity.clone()                   # (B,)

    def reset(self, force_first: Tensor | None = None) -> Tensor:
        """Initialise state; return the initial (B, N) feasibility mask.

        force_first: (B,) long customer indices to force as the first visit
        (POMO). If given, only that customer is feasible at step 0.
        """
        self._reset_state()
        mask = feasibility_mask(
            self.visited, self.demand, self.remaining_capacity, self.current_node
        )
        if force_first is not None:
            forced = torch.ones_like(mask)                       # all infeasible
            forced.scatter_(1, force_first.view(-1, 1), False)   # only forced feasible
            mask = forced
        return mask

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one step.

        action: (B,) long, the chosen next node per instance.
        returns: (mask, done) with mask (B, N) bool and done (B,) bool.
        """
        served = self.demand.gather(1, action.view(-1, 1)).squeeze(1)  # (B,)
        self.remaining_capacity = self.remaining_capacity - served
        is_depot = action == 0
        self.remaining_capacity = torch.where(
            is_depot, self.capacity, self.remaining_capacity
        )  # depot refills the vehicle
        self.visited.scatter_(1, action.view(-1, 1), True)
        self.visited[:, 0] = False           # depot is never permanently "visited"
        self.current_node = action
        done = self.visited[:, 1:].all(dim=1)  # (B,) all customers served
        mask = feasibility_mask(
            self.visited,
            self.demand,
            self.remaining_capacity,
            self.current_node,
            done,
        )
        return mask, done

    def normalized_remaining(self) -> Tensor:
        """Remaining capacity in [0, 1] for the decoder context. Shape (B, 1)."""
        return (self.remaining_capacity / self.capacity).unsqueeze(1)

    def cost(self, actions: Tensor) -> Tensor:
        """Tour length per instance. actions: (B, T) long -> (B,) float."""
        return tour_length(self.coords, actions)