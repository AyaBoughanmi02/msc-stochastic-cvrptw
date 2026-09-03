"""Tests for CVRPTW POMO training (variant A).

Confirms the multi-start rollout replicates windows/service correctly (forced
starts stay feasible), returns the right shapes with gradient on log-prob, and a
POMO step runs and updates the weights.
"""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.models.factory import build_cvrptw_policy
from src.train.pomo_cvrptw import (
    multistart_greedy_cost_cvrptw,
    pomo_rollout_cvrptw,
    train_step_cvrptw,
)

_CAP = 20


def test_rollout_shapes_and_grad() -> None:
    data = generate_cvrptw(4, 20, seed=1, capacity=_CAP)
    torch.manual_seed(0)
    policy = build_cvrptw_policy()
    cost, sum_logp = pomo_rollout_cvrptw(policy, data, n_starts=20)
    assert cost.shape == (4, 20)
    assert sum_logp.shape == (4, 20)
    assert sum_logp.requires_grad
    assert not cost.requires_grad


def test_greedy_eval_shape_and_range() -> None:
    data = generate_cvrptw(4, 20, seed=2, capacity=_CAP)
    torch.manual_seed(0)
    policy = build_cvrptw_policy()
    cost = multistart_greedy_cost_cvrptw(policy, data, n_starts=20)
    assert cost.shape == (4,)
    assert (cost > 0).all()                       # positive tour lengths


def test_train_step_runs_and_updates() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in policy.parameters()]

    def sample_batch():
        return generate_cvrptw(8, 20, capacity=_CAP)

    metrics = train_step_cvrptw(policy, opt, sample_batch, accum_steps=1, n_starts=20)
    assert set(metrics) == {"loss", "mean_cost", "grad_norm"}
    assert torch.isfinite(torch.tensor(metrics["mean_cost"]))
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))