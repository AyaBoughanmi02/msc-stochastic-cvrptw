"""POMO training for CVRPTW (variant A).

Mirrors the CVRP POMO path (src/train/pomo.py) for the CVRPTW env/policy. The only
CVRPTW-specific step is replicating the window/service tensors alongside coords and
demand when building the n_starts multi-start batch; cost is still tour distance,
so pomo_loss and the shared baseline are reused unchanged.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.envs.cvrptw_env import CVRPTWEnv
from src.models.cvrptw_policy import CVRPTWPolicy
from src.train.pomo import build_starts, pomo_loss


def _replicate_env(data: CVRPTWData, n_starts: int) -> CVRPTWEnv:
    """Build a CVRPTWEnv with each instance replicated CONTIGUOUSLY n_starts times."""
    coords = data.coords.repeat_interleave(n_starts, dim=0)        # (B*S, N, 2)
    demand = data.demand.repeat_interleave(n_starts, dim=0)        # (B*S, N)
    time_windows = data.time_windows.repeat_interleave(n_starts, dim=0)  # (B*S, N, 2)
    service = data.service_time.repeat_interleave(n_starts, dim=0)       # (B*S, N)
    return CVRPTWEnv(coords, demand, data.capacity, time_windows, service, data.shift_end)


def pomo_rollout_cvrptw(
    policy: CVRPTWPolicy,
    data: CVRPTWData,
    n_starts: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Run N multi-start sampled trajectories per instance. Returns cost, logp (B, S)."""
    b, n, _ = data.coords.shape
    n_customers = n - 1
    if n_starts is None:
        n_starts = n_customers
    if not 1 <= n_starts <= n_customers:
        raise ValueError(f"n_starts must be in [1, {n_customers}], got {n_starts}")

    env = _replicate_env(data, n_starts)
    starts = build_starts(b, n_starts, data.coords.device)          # (B*S,)
    _, sum_logp, cost = policy(env, decode_type="sampling", force_first=starts)
    return cost.reshape(b, n_starts), sum_logp.reshape(b, n_starts)


def train_step_cvrptw(
    policy: CVRPTWPolicy,
    optimizer: torch.optim.Optimizer,
    sample_batch: Callable[[], CVRPTWData],
    accum_steps: int = 1,
    n_starts: int | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """One POMO optimizer step over accum_steps CVRPTW mini-batches.

    Each mini-batch loss is scaled by 1/accum_steps so the accumulated gradient is
    the mean of the mini-batch gradients (same convention as the CVRP path).
    """
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    policy.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    cost_sum = 0.0
    for _ in range(accum_steps):
        data = sample_batch()
        cost, sum_logp = pomo_rollout_cvrptw(policy, data, n_starts)
        loss, mean_cost = pomo_loss(cost, sum_logp)                # reused unchanged
        (loss / accum_steps).backward()
        loss_sum += float(loss.detach())
        cost_sum += float(mean_cost.detach())
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    else:
        grad_norm = torch.tensor(float("nan"))
    optimizer.step()
    return {"loss": loss_sum / accum_steps, "mean_cost": cost_sum / accum_steps,
            "grad_norm": float(grad_norm)}


@torch.no_grad()
def multistart_greedy_cost_cvrptw(
    policy: CVRPTWPolicy,
    data: CVRPTWData,
    n_starts: int | None = None,
) -> Tensor:
    """Best cost over N greedy multi-start rollouts per instance. Returns (B,)."""
    policy.eval()
    b, n, _ = data.coords.shape
    n_customers = n - 1
    if n_starts is None:
        n_starts = n_customers
    env = _replicate_env(data, n_starts)
    starts = build_starts(b, n_starts, data.coords.device)
    _, _, cost = policy(env, decode_type="greedy", force_first=starts)
    return cost.reshape(b, n_starts).min(dim=1).values             # (B,) best of N