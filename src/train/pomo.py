"""POMO training (Kwon et al. 2020), critic-free shared-baseline REINFORCE.

For each instance we roll out N trajectories, each forced to start at a
different customer (POMO multi-start). The shared baseline is the per-instance
mean cost over those N trajectories; the advantage is each trajectory's cost
minus that mean. Confirmed against Kwon et al. 2020 sec. 5.2: for CVRP, ALL
customer nodes are used as starting nodes (no subsampling), trajectories are
sampled during training, and the baseline is the mean tour length over the N
trajectories.

Sign convention: cost = tour length (to be minimised). advantage = cost -
baseline; loss = mean(advantage.detach() * sum_logp). Minimising this loss
increases the log-prob of below-average-cost trajectories.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from src.envs.cvrp_env import CVRPEnv
from src.models.policy import CVRPPolicy


def build_starts(b: int, n_starts: int, device: torch.device | str = "cpu") -> Tensor:
    """Forced first-customer index per replicated trajectory.

    Returns (b * n_starts,) = [1, 2, ..., n_starts] tiled over the b instances,
    matching the contiguous instance replication used by pomo_rollout so a later
    reshape to (b, n_starts) groups trajectories by instance. Distinct starts are
    what give the shared baseline its diversity; an off-by-one here silently
    collapses it.
    """
    return torch.arange(1, n_starts + 1, device=device).repeat(b)  # (b*n_starts,)


def pomo_rollout(
    policy: CVRPPolicy,
    coords: Tensor,             # (B, N, 2)
    demand: Tensor,             # (B, N)
    capacity: Tensor | float,
    n_starts: int | None = None,  # default: all customers (N - 1)
) -> tuple[Tensor, Tensor]:
    """Run N multi-start sampled trajectories per instance.

    Returns cost and summed log-prob, each shaped (B, n_starts).
    """
    b, n, _ = coords.shape
    n_customers = n - 1
    if n_starts is None:
        n_starts = n_customers
    if not 1 <= n_starts <= n_customers:
        raise ValueError(f"n_starts must be in [1, {n_customers}], got {n_starts}")

    dev = coords.device
    # Replicate each instance n_starts times, CONTIGUOUSLY (instance b -> rows
    # [b*n_starts : (b+1)*n_starts]), so a later reshape to (B, n_starts) groups
    # trajectories by instance.
    coords_r = coords.repeat_interleave(n_starts, dim=0)   # (B*S, N, 2)
    demand_r = demand.repeat_interleave(n_starts, dim=0)   # (B*S, N)
    if isinstance(capacity, Tensor):
        capacity_r: Tensor | float = capacity.repeat_interleave(n_starts, dim=0)
    else:
        capacity_r = capacity

    # Force customer k as the first visit for the k-th trajectory of each
    # instance: [1, 2, ..., n_starts] tiled over the B instances.
    starts = build_starts(b, n_starts, dev)  # (B*S,)

    env = CVRPEnv(coords_r, demand_r, capacity_r)
    _, sum_logp, cost = policy(env, decode_type="sampling", force_first=starts)

    return cost.reshape(b, n_starts), sum_logp.reshape(b, n_starts)


def pomo_loss(cost: Tensor, sum_logp: Tensor) -> tuple[Tensor, Tensor]:
    """Shared-baseline REINFORCE loss.

    cost, sum_logp: (B, n_starts). Returns (loss, mean_cost).
    """
    baseline = cost.mean(dim=1, keepdim=True)          # (B, 1) shared baseline
    advantage = cost - baseline                        # (B, n_starts)
    loss = (advantage.detach() * sum_logp).mean()      # scalar
    return loss, cost.mean()


def train_step(
    policy: CVRPPolicy,
    optimizer: torch.optim.Optimizer,
    coords: Tensor,
    demand: Tensor,
    capacity: Tensor | float,
    n_starts: int | None = None,
    grad_clip: float | None = None,  # Kwon uses no clipping; enable if unstable
) -> dict[str, float]:
    """One POMO gradient step. Returns {'loss', 'mean_cost', 'grad_norm'}."""
    policy.train()
    cost, sum_logp = pomo_rollout(policy, coords, demand, capacity, n_starts)
    loss, mean_cost = pomo_loss(cost, sum_logp)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    else:
        grad_norm = torch.tensor(float("nan"))
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "mean_cost": float(mean_cost.detach()),
        "grad_norm": float(grad_norm),
    }


def train_step_accum(
        policy: CVRPPolicy,
        optimizer: torch.optim.Optimizer,
        sample_batch: Callable[[], tuple[Tensor, Tensor, float | Tensor]],
        accum_steps: int = 1,
        n_starts: int | None = None,
        grad_clip: float | None = None,
) -> dict[str, float]:
    """One optimizer step over `accum_steps` accumulated POMO mini-batches.

    Each mini-batch's loss is scaled by 1/accum_steps before backward, so the
    accumulated gradient is the MEAN of the mini-batch gradients -- a
    lower-variance estimate equivalent to a batch accum_steps x larger, at the
    memory cost of a single mini-batch. (Exact equality to one large batch holds
    up to BatchNorm's batch-coupling, a negligible approximation here.) The POMO
    shared baseline is per-instance (over each instance's own starts), so it is
    unaffected by accumulation.

    accum_steps=1 reproduces a single-batch step exactly.
    """
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    policy.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    cost_sum = 0.0
    for _ in range(accum_steps):
        coords, demand, capacity = sample_batch()
        cost, sum_logp = pomo_rollout(policy, coords, demand, capacity, n_starts)
        loss, mean_cost = pomo_loss(cost, sum_logp)
        (loss / accum_steps).backward()  # accumulate mean gradient
        loss_sum += float(loss.detach())
        cost_sum += float(mean_cost.detach())
    if grad_clip is not None:  # clip the accumulated gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    else:
        grad_norm = torch.tensor(float("nan"))
    optimizer.step()
    return {"loss": loss_sum / accum_steps, "mean_cost": cost_sum / accum_steps,
            "grad_norm": float(grad_norm)}


def train(
    policy: CVRPPolicy,
    sample_batch: Callable[[], tuple[Tensor, Tensor, float]],
    n_steps: int,
    lr: float = 1e-4,               # Kwon: Adam lr 1e-4
    weight_decay: float = 1e-6,     # Kwon: weight_decay 1e-6
    n_starts: int | None = None,
    grad_clip: float | None = None,  # Kwon: no clipping
    log_every: int = 10,
) -> list[dict[str, float]]:
    """Minimal POMO training loop over freshly sampled instances.

    sample_batch: callable returning (coords, demand, capacity) for one batch.
    Returns the list of per-step metric dicts.
    """
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=lr, weight_decay=weight_decay
    )
    history: list[dict[str, float]] = []
    for step in range(n_steps):
        coords, demand, capacity = sample_batch()
        metrics = train_step(
            policy, optimizer, coords, demand, capacity, n_starts, grad_clip
        )
        history.append(metrics)
        if log_every and (step % log_every == 0 or step == n_steps - 1):
            print(
                f"step {step:4d} | loss {metrics['loss']:+.4f} "
                f"| mean_cost {metrics['mean_cost']:.4f} "
                f"| grad_norm {metrics['grad_norm']:.3f}"
            )
    return history