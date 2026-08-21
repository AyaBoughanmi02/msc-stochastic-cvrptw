"""CVRP evaluation: best-of-N greedy decoding, optional POMO 8x augmentation,
and optimality gap against an independent reference.

POMO's reported score is the best of the N multi-start greedy rollouts per
instance (not the mean). Instance augmentation applies the 8 dihedral symmetries
of the unit square; these preserve pairwise distances, so tour costs are
comparable across augmentations and we simply take the best candidate overall.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.envs.cvrp_env import CVRPEnv
from src.models.policy import CVRPPolicy
from src.train.pomo import build_starts


@torch.no_grad()
def multistart_greedy_cost(
    policy: CVRPPolicy,
    coords: Tensor,             # (B, N, 2)
    demand: Tensor,             # (B, N)
    capacity: Tensor | float,
    n_starts: int | None = None,
) -> Tensor:
    """Best cost over N greedy multi-start rollouts per instance. Returns (B,)."""
    b, n, _ = coords.shape
    n_customers = n - 1
    if n_starts is None:
        n_starts = n_customers

    coords_r = coords.repeat_interleave(n_starts, dim=0)   # (B*S, N, 2)
    demand_r = demand.repeat_interleave(n_starts, dim=0)   # (B*S, N)
    capacity_r = (
        capacity.repeat_interleave(n_starts, dim=0)
        if isinstance(capacity, Tensor)
        else capacity
    )
    starts = build_starts(b, n_starts, coords.device)      # (B*S,)

    env = CVRPEnv(coords_r, demand_r, capacity_r)
    _, _, cost = policy(env, decode_type="greedy", force_first=starts)  # (B*S,)
    return cost.reshape(b, n_starts).min(dim=1).values      # (B,) best of N


def augment_coords_8(coords: Tensor) -> Tensor:
    """Return the 8 dihedral symmetries of unit-square coords.

    coords: (B, N, 2) -> (8, B, N, 2). Distances are preserved by each map.
    """
    x, y = coords[..., 0], coords[..., 1]                  # (B, N) each
    variants = [
        (x, y), (y, x),
        (1 - x, y), (1 - y, x),
        (x, 1 - y), (y, 1 - x),
        (1 - x, 1 - y), (1 - y, 1 - x),
    ]
    return torch.stack([torch.stack([a, b], dim=-1) for a, b in variants], dim=0)


@torch.no_grad()
def evaluate(
    policy: CVRPPolicy,
    coords: Tensor,             # (B, N, 2)
    demand: Tensor,             # (B, N)
    capacity: Tensor | float,
    n_starts: int | None = None,
    augment: bool = False,
) -> Tensor:
    """Per-instance best cost. Returns (B,).

    With augment=True, runs best-of-N greedy on all 8 symmetries and takes the
    overall best (POMO's strongest inference mode).
    """
    policy.eval()
    if not augment:
        return multistart_greedy_cost(policy, coords, demand, capacity, n_starts)

    b, n, _ = coords.shape
    aug = augment_coords_8(coords)                          # (8, B, N, 2)
    aug_flat = aug.reshape(8 * b, n, 2)                     # (8B, N, 2)
    demand_rep = demand.unsqueeze(0).expand(8, -1, -1).reshape(8 * b, n)
    cost = multistart_greedy_cost(policy, aug_flat, demand_rep, capacity, n_starts)
    return cost.reshape(8, b).min(dim=0).values            # (B,) best over augments


def optimality_gap(cost: Tensor, reference: Tensor) -> Tensor:
    """Percentage gap of `cost` above `reference`, per instance. Returns (B,).

    gap_i = 100 * (cost_i - reference_i) / reference_i
    """
    if cost.shape != reference.shape:
        raise ValueError(f"shape mismatch: cost {cost.shape} vs ref {reference.shape}")
    return 100.0 * (cost - reference) / reference