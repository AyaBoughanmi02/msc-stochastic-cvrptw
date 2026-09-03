"""Tests for CVRPTWPolicy (variant A).

Checks the full policy rollout end to end: produced tours are feasible (windows +
shift respected, independently re-simulated) and complete, both greedy and
sampling; POMO forced starts are respected; and the policy is trainable (sum_logp
carries gradient, cost does not, and a gradient step updates weights).
"""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.envs.cvrptw_env import CVRPTWEnv
from src.models.factory import build_cvrptw_policy

_CAP = 20  # calibrated CVRPTW capacity


def _env(data) -> CVRPTWEnv:
    return CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )


def _assert_tours_feasible(data, dist, actions) -> None:
    b_size, n = data.coords.shape[0], data.coords.shape[1]
    earliest, latest = data.time_windows[..., 0], data.time_windows[..., 1]
    service, shift, eps = data.service_time, data.shift_end, 1e-4
    for b in range(b_size):
        clock, current = 0.0, 0
        for a in actions[b].tolist():
            arrival = clock + dist[b, current, a].item()
            if a == 0:
                assert arrival <= shift + eps, f"shift violated on return (inst {b})"
                clock, current = 0.0, 0
                continue
            assert arrival <= latest[b, a].item() + eps, f"window violated (inst {b}, node {a})"
            clock = max(arrival, earliest[b, a].item()) + service[b, a].item()
            current = a
        assert clock + dist[b, current, 0].item() <= shift + eps, f"final return (inst {b})"
        served = {x for x in actions[b].tolist() if x != 0}
        assert served == set(range(1, n)), f"not all customers served (inst {b})"


def test_rollout_feasible_and_complete_both_decode_types() -> None:
    data = generate_cvrptw(32, 20, seed=1, capacity=_CAP)
    torch.manual_seed(0)
    policy = build_cvrptw_policy()
    for decode in ("greedy", "sampling"):
        env = _env(data)
        actions, sum_logp, cost = policy(env, decode_type=decode)
        assert actions.shape[0] == 32 and cost.shape == (32,)
        _assert_tours_feasible(data, env.dist, actions)


def test_forced_first_respected_and_feasible() -> None:
    data = generate_cvrptw(4, 20, seed=3, capacity=_CAP)
    env = _env(data)
    torch.manual_seed(0)
    policy = build_cvrptw_policy()
    force = torch.tensor([1, 2, 3, 4])                     # force customer k for instance k
    actions, _, _ = policy(env, decode_type="greedy", force_first=force)
    assert torch.equal(actions[:, 0], force)               # first visit is the forced customer
    _assert_tours_feasible(data, env.dist, actions)


def test_policy_is_trainable() -> None:
    data = generate_cvrptw(8, 20, seed=2, capacity=_CAP)
    env = _env(data)
    torch.manual_seed(0)
    policy = build_cvrptw_policy()
    actions, sum_logp, cost = policy(env, decode_type="sampling")
    assert sum_logp.requires_grad          # gradient path for the policy gradient
    assert not cost.requires_grad          # cost is a reward signal, detached
    # A REINFORCE-style step must run and move the weights (not real POMO baseline).
    loss = (cost.detach() * sum_logp).mean()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in policy.parameters()]
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
