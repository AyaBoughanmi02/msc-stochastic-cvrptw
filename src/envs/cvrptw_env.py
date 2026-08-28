"""Batched CVRPTW environment (variant A): CVRP + hard time windows + shift.

Clean extension of CVRPEnv. Adds a per-route CLOCK that advances by travel + wait
+ service and RESETS TO 0 on every depot return -- each depot->...->depot loop is
one vehicle living its own fresh shift, exactly as capacity refills at the depot.
Masking is CVRP masking plus the two hard time-window conditions
(masking.tw_feasibility_mask). The distance objective is unchanged (cost() is
inherited); the clock only drives feasibility.

Travel time for variant A is Euclidean distance (time == distance), read from
self.dist. The stochastic variants replace self.dist with an Edge-Bank lookup;
nothing else in this class changes.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.cvrptw_generator import pairwise_distances
from src.envs.cvrp_env import CVRPEnv
from src.utils.masking import feasibility_mask, tw_feasibility_mask


class CVRPTWEnv(CVRPEnv):
    """CVRPTW state machine for a batch of instances (variant A)."""

    def __init__(
        self,
        coords: Tensor,             # (B, N, 2)
        demand: Tensor,             # (B, N) float, demand[:, 0] = 0
        capacity: Tensor | float,   # (B,) or scalar
        time_windows: Tensor,       # (B, N, 2) [earliest, latest] service-start
        service_time: Tensor,       # (B, N) service duration, depot 0
        shift_end: float,           # per-route shift horizon
    ) -> None:
        # Set TW data before super().__init__ so it is present for any masking;
        # _reset_state (overridden below) only needs batch_size/device.
        self.time_windows = time_windows
        self.service_time = service_time
        self.shift_end = float(shift_end)
        super().__init__(coords, demand, capacity)   # sets CVRP state, calls _reset_state
        self.dist = pairwise_distances(coords)        # (B, N, N) travel time (variant A)
        self.time: Tensor  # (B,) declared for clarity; set in _reset_state

    def _reset_state(self) -> None:
        super()._reset_state()                        # visited / current_node / capacity
        self.time = torch.zeros(self.batch_size, device=self.coords.device)  # (B,)

    def _tw_mask(self, base_mask: Tensor) -> Tensor:
        """CVRP base mask + hard TW conditions at the current clock/node. (B, N)."""
        idx = torch.arange(self.batch_size, device=self.coords.device)
        travel_from_current = self.dist[idx, self.current_node]   # (B, N)
        travel_to_depot = self.dist[:, :, 0]                       # (B, N)
        return tw_feasibility_mask(
            base_mask, self.time, travel_from_current, travel_to_depot,
            self.time_windows, self.service_time, self.shift_end,
        )

    def reset(self, force_first: Tensor | None = None) -> Tensor:
        """Initialise state; return the initial (B, N) feasibility mask."""
        self._reset_state()
        base = feasibility_mask(
            self.visited, self.demand, self.remaining_capacity, self.current_node
        )
        mask = self._tw_mask(base)
        if force_first is not None:
            forced = torch.ones_like(mask)                       # all infeasible
            forced.scatter_(1, force_first.view(-1, 1), False)   # only forced feasible
            mask = forced
        return mask

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one step: update clock, then CVRP capacity/visited, then mask."""
        idx = torch.arange(self.batch_size, device=self.coords.device)
        # Clock uses the OLD current node (super().step() will move current_node).
        travel = self.dist[idx, self.current_node, action]        # (B,) old_current -> action
        arrival = self.time + travel                              # (B,)
        earliest_a = self.time_windows[idx, action, 0]            # (B,)
        service_a = self.service_time[idx, action]                # (B,)
        new_time = torch.maximum(arrival, earliest_a) + service_a  # (B,) wait if early, then serve
        is_depot = action == 0
        self.time = torch.where(is_depot, torch.zeros_like(new_time), new_time)  # reset at depot

        base_mask, done = super().step(action)                   # CVRP update + base mask + done
        mask = self._tw_mask(base_mask)
        return mask, done

    def normalized_time(self) -> Tensor:
        """Clock in [0, 1] for the decoder context. Shape (B, 1).

        Bounded by masking: an action is only feasible if it can be served and
        returned within the shift, so self.time never exceeds shift_end.
        """
        return (self.time / self.shift_end).unsqueeze(1)