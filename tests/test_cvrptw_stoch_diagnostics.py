"""Tests for the stochastic-CVRPTW calibration diagnostics.

Load-bearing tests:
  - `test_decomposition_agrees_with_the_real_mask`: the diagnostic explains the mask by
    recomputing it, so if it read a different travel-time source than the mask does it
    would report confident nonsense. Checked for BOTH envs, since the shared
    `decompose_infeasibility` now reads through the seam hooks.
  - `test_variant_a_greedy_rollout_is_unchanged`: rewiring the diagnostics through the
    hooks must not have changed a single action on the certified deterministic path.
  - `test_sweep_arithmetic_is_exact`: the whole point of the sweep is that it is cheap,
    exact arithmetic over one rollout rather than a retraining loop.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig, build_edge_bank, draw_noise
from src.envs.cvrptw_diagnostics import decompose_infeasibility, greedy_feasible_rollout
from src.envs.cvrptw_env import CVRPTWEnv
from src.envs.cvrptw_stoch_diagnostics import (
    DEFAULT_WEIGHTS,
    TardinessProfile,
    policy_tardiness_stats,
    recommend_tardiness_weight,
    stoch_calibration_stats,
    tardiness_weight_sweep,
    weight_grid_saturated,
)
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.utils.masking import tour_length

_SHIFT = 4.0
_SERVICE = 0.2
_CAPACITY = 20        # the locked CVRPTW calibration


def _data(batch: int = 8, n_customers: int = 20, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


def _det_env(data: CVRPTWData) -> CVRPTWEnv:
    return CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )


def _stoch_env(data: CVRPTWData, cfg: SimulatorConfig, seed: int = 0) -> CVRPTWStochEnv:
    bank = build_edge_bank(data.coords, cfg)
    noise = draw_noise(
        bank.shape[:-1], device="cpu", generator=torch.Generator().manual_seed(seed)
    )
    return CVRPTWStochEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
        bank=bank, sim_cfg=cfg, noise=noise, tardiness_weight=0.0,
    )


# ------------------------------------------------- the diagnostics still explain the mask


@pytest.mark.parametrize("stochastic", [False, True])
def test_decomposition_agrees_with_the_real_mask(stochastic: bool) -> None:
    """OR of the four reasons == the real mask, on the customer columns, at every step."""
    data = _data(batch=6, n_customers=10, seed=1)
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.2, eta=0.6, shift_end=_SHIFT)
    env = _stoch_env(data, cfg, seed=1) if stochastic else _det_env(data)

    mask = env.reset()
    cust = slice(1, None)
    for _ in range(2 * env.num_nodes):
        reasons = decompose_infeasibility(env)
        rebuilt = (
            reasons.visited | reasons.over_capacity
            | reasons.window_closed | reasons.cant_return
        )
        assert torch.equal(rebuilt[:, cust], mask[:, cust])
        action = env._travel_from_current().masked_fill(mask, float("inf")).argmin(dim=1)
        mask, done = env.step(action)
        if bool(done.all()):
            break


def test_variant_a_greedy_rollout_is_unchanged() -> None:
    """The seam rewiring must not change a single action on the deterministic path."""
    data = _data(batch=8, n_customers=20, seed=2)

    env = _det_env(data)
    via_hooks = greedy_feasible_rollout(env)

    # Explicit re-implementation of the pre-rewiring behaviour, reading env.dist.
    ref = _det_env(data)
    mask = ref.reset()
    idx = torch.arange(ref.batch_size)
    actions: list[Tensor] = []
    for _ in range(3 * ref.num_nodes):
        d = ref.dist[idx, ref.current_node].masked_fill(mask, float("inf"))
        action = d.argmin(dim=1)
        actions.append(action)
        mask, done = ref.step(action)
        if bool(done.all()):
            break
    assert torch.equal(via_hooks, torch.stack(actions, dim=1))


# ------------------------------------------------------------------ the sweep is exact


def test_sweep_arithmetic_is_exact() -> None:
    components = {
        "distance": torch.tensor([10.0, 6.0]),
        "tardiness": torch.tensor([0.5, 0.0]),
        "late_count": torch.tensor([1.0, 0.0]),
    }
    profiles = tardiness_weight_sweep(components, [0.0, 2.0])
    assert profiles[0] == TardinessProfile(0.0, 0.0, 8.0, 0.0)
    # violations = [1.5, 0.0]; w=2 -> penalty [3.0, 0.0]; totals 16 distance + 3 penalty
    assert profiles[1].mean_penalty == pytest.approx(1.5)
    assert profiles[1].mean_cost == pytest.approx(8.0 + 1.5)
    assert profiles[1].penalty_share == pytest.approx(3.0 / 19.0)


def test_penalty_share_is_monotone_in_the_weight() -> None:
    components = {
        "distance": torch.tensor([9.0, 9.0]),
        "tardiness": torch.tensor([0.3, 0.1]),
        "late_count": torch.tensor([1.0, 1.0]),
    }
    shares = [p.penalty_share for p in tardiness_weight_sweep(components, DEFAULT_WEIGHTS)]
    assert shares[0] == 0.0
    assert all(b > a for a, b in zip(shares, shares[1:]))


def test_zero_violations_gives_zero_share_at_every_weight() -> None:
    components = {
        "distance": torch.tensor([9.0]),
        "tardiness": torch.zeros(1),
        "late_count": torch.zeros(1),
    }
    assert all(p.penalty_share == 0.0 for p in tardiness_weight_sweep(components))


def test_grid_saturation_is_flagged() -> None:
    """A recommendation pinned to an end of the grid must be reported as unusable."""
    below = [TardinessProfile(w, 0.0, 0.0, s) for w, s in [(1.0, 0.01), (8.0, 0.05)]]
    above = [TardinessProfile(w, 0.0, 0.0, s) for w, s in [(1.0, 0.55), (8.0, 0.90)]]
    spanning = [TardinessProfile(w, 0.0, 0.0, s) for w, s in [(1.0, 0.08), (8.0, 0.44)]]
    assert weight_grid_saturated(below, 0.2)      # even the largest weight is too small
    assert weight_grid_saturated(above, 0.2)      # even the smallest weight is too big
    assert not weight_grid_saturated(spanning, 0.2)


def test_recommend_picks_the_closest_share() -> None:
    profiles = [
        TardinessProfile(0.5, 0.0, 0.0, 0.05),
        TardinessProfile(1.0, 0.0, 0.0, 0.18),
        TardinessProfile(4.0, 0.0, 0.0, 0.60),
    ]
    assert recommend_tardiness_weight(profiles, target_share=0.2) == 1.0
    assert recommend_tardiness_weight(profiles, target_share=0.5) == 4.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: tardiness_weight_sweep({"distance": torch.zeros(1)}),
        lambda: tardiness_weight_sweep(
            {"distance": torch.zeros(1), "tardiness": torch.zeros(1),
             "late_count": torch.zeros(1)}, weights=[],
        ),
        lambda: tardiness_weight_sweep(
            {"distance": torch.zeros(1), "tardiness": torch.zeros(1),
             "late_count": torch.zeros(1)}, weights=[-1.0],
        ),
        lambda: recommend_tardiness_weight([]),
        lambda: recommend_tardiness_weight(
            [TardinessProfile(1.0, 0.0, 0.0, 0.1)], target_share=1.0
        ),
    ],
)
def test_sweep_guards_raise(call) -> None:
    with pytest.raises(ValueError):
        call()


# ------------------------------------------------------------------ level 1 end to end


def test_stoch_calibration_stats_reports_binding_and_tardiness() -> None:
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.15, eta=0.6, shift_end=_SHIFT)
    stats = stoch_calibration_stats(_data(batch=16, seed=3), cfg, seed=3)

    assert set(stats) >= {
        "time_only_rate", "capacity_only_rate", "both_rate", "mean_distance",
        "mean_distance_per_customer", "mean_tardiness", "mean_late_count",
        "late_rate", "profiles", "recommended_weight",
    }
    # Stage 2 measured time_only ~11x capacity_only; the ripple must not invert that.
    assert stats["time_only_rate"] > stats["capacity_only_rate"]
    assert stats["mean_tardiness"] > 0.0            # the proxy does incur lateness
    assert 0.0 <= stats["late_rate"] <= 1.0
    assert stats["recommended_weight"] in DEFAULT_WEIGHTS
    assert isinstance(stats["weight_grid_saturated"], bool)


def test_level_1_reduces_to_variant_a() -> None:
    """alpha = 0, beta_var = 0: same greedy tour and no tardiness at all."""
    cfg = SimulatorConfig(alpha=0.0, beta_var=0.0, shift_end=_SHIFT)
    data = _data(batch=8, seed=4)
    stats = stoch_calibration_stats(data, cfg, seed=4)

    env = _det_env(data)
    reference = float(tour_length(env.coords, greedy_feasible_rollout(env)).mean())
    assert stats["mean_distance"] == pytest.approx(reference, abs=1e-5)
    assert stats["mean_tardiness"] == 0.0
    assert stats["mean_late_count"] == 0.0


def test_more_variance_needs_a_smaller_weight_for_the_same_share() -> None:
    """Sanity on the dial: more lateness => the same share is reached sooner."""
    data = _data(batch=16, seed=5)
    low = stoch_calibration_stats(
        data, SimulatorConfig(alpha=0.3, beta_var=0.05, shift_end=_SHIFT), seed=5
    )
    high = stoch_calibration_stats(
        data, SimulatorConfig(alpha=0.3, beta_var=0.6, shift_end=_SHIFT), seed=5
    )
    assert high["mean_tardiness"] > low["mean_tardiness"]
    assert high["recommended_weight"] <= low["recommended_weight"]


# ------------------------------------------------------------------ level 2 end to end


class _RandomFeasiblePolicy(nn.Module):
    """Stand-in with CVRPTWPolicy's call signature (the network is not under test here)."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.seed = seed

    def forward(self, env, decode_type="sampling", force_first=None):
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        mask = env.reset(force_first)
        actions: list[Tensor] = []
        for _ in range(2 * env.num_nodes):
            feasible = (~mask).to(torch.float32)
            action = torch.multinomial(feasible, 1, generator=gen).squeeze(1)
            actions.append(action)
            mask, done = env.step(action)
            if bool(done.all()):
                break
        actions_t = torch.stack(actions, dim=1)
        sum_logp = self.scale.expand(env.batch_size) * float(actions_t.size(1))
        return actions_t, sum_logp, env.cost(actions_t)


def test_policy_tardiness_stats_prices_a_policys_own_routing() -> None:
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.2, shift_end=_SHIFT)
    stats = policy_tardiness_stats(
        _RandomFeasiblePolicy(), _data(batch=4, n_customers=10, seed=6), cfg,
        n_starts=4, seed=6,
    )
    assert stats["mean_distance"] > 0.0
    assert stats["mean_tardiness"] >= 0.0
    assert stats["recommended_weight"] in DEFAULT_WEIGHTS
    assert len(stats["profiles"]) == len(DEFAULT_WEIGHTS)