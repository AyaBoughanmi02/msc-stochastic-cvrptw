"""Tests for the CVRPTW calibration diagnostics.

The key test pins the decomposition to the real mask: the OR of the four reasons
over the CUSTOMER columns must equal the mask's customer columns, so the diagnostic
can never silently disagree with the masking it explains.
"""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.envs.cvrptw_diagnostics import (
    calibration_stats,
    decompose_infeasibility,
    greedy_feasible_rollout,
)
from src.envs.cvrptw_env import CVRPTWEnv

_COORDS = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])


def _two_route_env() -> CVRPTWEnv:
    demand = torch.tensor([[0.0, 0.3, 0.3]])
    tw = torch.tensor([[[0.0, 4.0], [1.0, 1.5], [1.0, 2.0]]])
    service = torch.tensor([[0.0, 0.2, 0.2]])
    return CVRPTWEnv(_COORDS.clone(), demand, 1.0, tw, service, shift_end=4.0)


def test_decomposition_matches_real_mask_customers() -> None:
    # Drive to a mid-route state, then check the OR of reasons over customers
    # equals the mask returned by the env (which is what the policy actually sees).
    env = _two_route_env()
    env.reset()
    mask, done = env.step(torch.tensor([1]))          # serve c1 -> mid-route, not done
    assert not bool(done.all())
    r = decompose_infeasibility(env)
    combined = r.visited | r.over_capacity | r.window_closed | r.cant_return
    assert torch.equal(combined[:, 1:], mask[:, 1:])   # customer columns agree


def test_decomposition_matches_on_generated_batch() -> None:
    data = generate_cvrptw(16, 20, seed=9)
    env = CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    mask = env.reset()
    idx = torch.arange(env.batch_size)
    for _ in range(10):                                # a few steps into rollouts
        r = decompose_infeasibility(env)
        combined = r.visited | r.over_capacity | r.window_closed | r.cant_return
        # where not done, customer columns must match the real mask
        not_done = ~env.visited[:, 1:].all(dim=1)
        assert torch.equal(combined[not_done, 1:], mask[not_done, 1:])
        d = env.dist[idx, env.current_node].masked_fill(mask, float("inf"))
        mask, done = env.step(d.argmin(dim=1))
        if bool(done.all()):
            break


def test_rollout_serves_all_customers() -> None:
    data = generate_cvrptw(8, 20, seed=4)
    env = CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    acts = greedy_feasible_rollout(env)
    for b in range(8):
        served = {x for x in acts[b].tolist() if x != 0}
        assert served == set(range(1, 21))


def test_calibration_stats_keys_and_ranges() -> None:
    data = generate_cvrptw(32, 20, seed=1)
    s = calibration_stats(data)
    for k in ("time_only_rate", "capacity_only_rate", "both_rate", "feasible_rate"):
        assert 0.0 <= s[k] <= 1.0
    assert s["avg_routes_per_instance"] >= 1.0