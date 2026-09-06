"""Tests for the stochastic POMO batch assembly and rollout glue.

The network is not what is under test here, so these use a deterministic stand-in policy
with the same call signature as CVRPTWPolicy (`policy(env, decode_type, force_first) ->
(actions, sum_logp, cost)`). That isolates the parts this module owns: bank/noise
assembly, the instance_idx map, the CRN guarantee, the (B, S) reshapes, and the cost
decomposition. An end-to-end test with the real policy belongs with variant B, where the
encoder actually consumes the (mu, sigma^2) edge features.

Load-bearing tests:
  - `test_bank_and_noise_are_not_replicated`: the memory optimisation is what makes this
    fit on 4 GB at n=50; if it silently regresses, training OOMs late and expensively.
  - `test_common_random_numbers_are_shared_across_starts`: the POMO baseline is only a
    comparison of decisions if every start of an instance faces the same day.
  - `test_reduces_to_variant_a`: alpha = 0, beta_var = 0 must give zero tardiness and a
    penalised cost equal to the plain tour distance.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.train.pomo_stoch import (
    build_stoch_env,
    multistart_greedy_cost_stoch,
    pomo_rollout_stoch,
    train_step_stoch,
)

_SHIFT = 4.0
_SERVICE = 0.2
_CAPACITY = 20        # the locked CVRPTW calibration


class _RandomFeasiblePolicy(nn.Module):
    """Stand-in for CVRPTWPolicy: picks a uniformly random FEASIBLE action each step.

    Carries one real parameter so `loss.backward()` and the optimizer step exercise the
    same code path a real policy would.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.seed = seed

    def forward(
        self,
        env: CVRPTWStochEnv,
        decode_type: str = "sampling",
        force_first: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        mask = env.reset(force_first)                              # (B*S, N)
        actions: list[Tensor] = []
        for _ in range(2 * env.num_nodes):
            feasible = (~mask).to(torch.float32)                   # (B*S, N)
            if not bool((feasible.sum(dim=1) > 0).all()):
                raise RuntimeError("a trajectory had no feasible node")
            action = torch.multinomial(feasible, 1, generator=gen).squeeze(1)
            actions.append(action)
            mask, done = env.step(action)
            if bool(done.all()):
                break
        actions_t = torch.stack(actions, dim=1)                    # (B*S, T)
        sum_logp = self.scale.expand(env.batch_size) * float(actions_t.size(1))
        return actions_t, sum_logp, env.cost(actions_t)


def _data(batch: int = 4, n_customers: int = 10, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


# ------------------------------------------------------------------ batch assembly


def test_bank_and_noise_are_not_replicated() -> None:
    """The bank stays at instance granularity; only the geometry is replicated."""
    data = _data(batch=4, n_customers=10, seed=1)
    cfg = SimulatorConfig(shift_end=_SHIFT)
    n_starts = 5
    env = build_stoch_env(
        data, cfg, n_starts, tardiness_weight=1.0,
        generator=torch.Generator().manual_seed(1),
    )
    b, n = 4, 11
    assert env.batch_size == b * n_starts
    assert env.coords.shape == (b * n_starts, n, 2)
    assert env.bank.shape == (b, n, n, cfg.n_bins, 2)               # NOT (B*S, ...)
    assert env.noise is not None and env.noise.shape == (b, n, n, cfg.n_bins)
    # instance_idx must map contiguous blocks of starts to their instance.
    expected = torch.arange(b).repeat_interleave(n_starts)
    assert torch.equal(env.instance_idx, expected)


def test_no_crn_draws_fresh_noise() -> None:
    env = build_stoch_env(
        _data(), SimulatorConfig(shift_end=_SHIFT), 3, tardiness_weight=1.0,
        generator=torch.Generator().manual_seed(2), common_random_numbers=False,
    )
    assert env.noise is None
    assert env.generator is not None


def test_common_random_numbers_are_shared_across_starts() -> None:
    """All starts of one instance must traverse a given edge at a given bin identically."""
    data = _data(batch=2, n_customers=8, seed=3)
    cfg = SimulatorConfig(beta_var=0.5, shift_end=_SHIFT)
    n_starts = 4
    env = build_stoch_env(
        data, cfg, n_starts, tardiness_weight=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    env.reset()
    travel = env._travel_time(torch.full((env.batch_size,), 2, dtype=torch.long))
    per_instance = travel.reshape(2, n_starts)                     # (B, S)
    assert torch.equal(per_instance, per_instance[:, :1].expand(2, n_starts))
    assert not torch.equal(per_instance[0, 0], per_instance[1, 0])  # instances differ


# ---------------------------------------------------------------------- rollouts


def test_rollout_shapes_and_decomposition() -> None:
    data = _data(batch=3, n_customers=10, seed=4)
    cfg = SimulatorConfig(beta_var=0.2, eta=0.6, shift_end=_SHIFT)
    out = pomo_rollout_stoch(
        _RandomFeasiblePolicy(), data, cfg, n_starts=6, tardiness_weight=2.0,
        generator=torch.Generator().manual_seed(4),
    )
    for field in (out.cost, out.sum_logp, out.distance, out.tardiness, out.late_count):
        assert field.shape == (3, 6)
    assert torch.allclose(
        out.cost, out.distance + 2.0 * (out.late_count + out.tardiness), atol=1e-5
    )
    assert (out.tardiness >= 0.0).all()
    assert float(out.tardiness.sum()) > 0.0        # the uncertainty must actually bite


def test_default_n_starts_is_every_customer() -> None:
    data = _data(batch=2, n_customers=10, seed=5)
    out = pomo_rollout_stoch(
        _RandomFeasiblePolicy(), data, SimulatorConfig(shift_end=_SHIFT),
        generator=torch.Generator().manual_seed(5),
    )
    assert out.cost.shape == (2, 10)               # POMO: n_starts == n_customers


@pytest.mark.parametrize("bad", [0, 11, -1])
def test_invalid_n_starts_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        pomo_rollout_stoch(
            _RandomFeasiblePolicy(), _data(n_customers=10),
            SimulatorConfig(shift_end=_SHIFT), n_starts=bad,
        )


def test_reduces_to_variant_a() -> None:
    """alpha = 0, beta_var = 0: no tardiness, so the penalised cost IS the tour distance."""
    cfg = SimulatorConfig(alpha=0.0, beta_var=0.0, shift_end=_SHIFT)
    out = pomo_rollout_stoch(
        _RandomFeasiblePolicy(), _data(batch=4, n_customers=10, seed=6), cfg,
        n_starts=4, tardiness_weight=5.0,
        generator=torch.Generator().manual_seed(6),
    )
    assert float(out.tardiness.sum()) == 0.0
    assert float(out.late_count.sum()) == 0.0
    assert torch.equal(out.cost, out.distance)


def test_variance_shift_increases_tardiness() -> None:
    """The OOD knob must move the quantity it is supposed to move."""
    data = _data(batch=8, n_customers=10, seed=7)
    policy = _RandomFeasiblePolicy()
    tardiness = []
    for beta_var in (0.02, 0.4):
        out = pomo_rollout_stoch(
            policy, data, SimulatorConfig(beta_var=beta_var, shift_end=_SHIFT),
            n_starts=5, generator=torch.Generator().manual_seed(7),
        )
        tardiness.append(float(out.tardiness.mean()))
    assert tardiness[1] > tardiness[0]


def test_rollout_is_reproducible_given_a_seed() -> None:
    data = _data(batch=3, n_customers=10, seed=8)
    cfg = SimulatorConfig(beta_var=0.3, shift_end=_SHIFT)
    runs = [
        pomo_rollout_stoch(
            _RandomFeasiblePolicy(), data, cfg, n_starts=4,
            generator=torch.Generator().manual_seed(99),
        ).cost
        for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])


# ------------------------------------------------------------------ train / eval


def test_train_step_reports_parts_and_updates_parameters() -> None:
    policy = _RandomFeasiblePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    before = policy.scale.detach().clone()
    stats = train_step_stoch(
        policy, optimizer, lambda: _data(batch=3, n_customers=10, seed=9),
        SimulatorConfig(beta_var=0.25, shift_end=_SHIFT),
        accum_steps=2, n_starts=4, tardiness_weight=1.5,
        grad_clip=1.0, generator=torch.Generator().manual_seed(9),
    )
    assert set(stats) == {
        "loss", "mean_cost", "mean_distance", "mean_tardiness", "late_rate", "grad_norm"
    }
    assert stats["mean_cost"] >= stats["mean_distance"]     # the penalty is non-negative
    assert 0.0 <= stats["late_rate"] <= 1.0
    assert not torch.equal(before, policy.scale.detach())   # the step actually stepped


def test_train_step_rejects_bad_accum() -> None:
    policy = _RandomFeasiblePolicy()
    with pytest.raises(ValueError):
        train_step_stoch(
            policy, torch.optim.Adam(policy.parameters()), lambda: _data(),
            SimulatorConfig(shift_end=_SHIFT), accum_steps=0,
        )


def test_greedy_cost_is_best_of_n_and_paired_across_seeds() -> None:
    data = _data(batch=4, n_customers=10, seed=10)
    cfg = SimulatorConfig(beta_var=0.2, shift_end=_SHIFT)
    kwargs = dict(n_starts=5, tardiness_weight=1.0)

    best = multistart_greedy_cost_stoch(
        _RandomFeasiblePolicy(), data, cfg,
        generator=torch.Generator().manual_seed(11), **kwargs,
    )
    full = pomo_rollout_stoch(
        _RandomFeasiblePolicy(), data, cfg, decode_type="greedy",
        generator=torch.Generator().manual_seed(11), **kwargs,
    )
    assert best.shape == (4,)
    assert torch.equal(best, full.cost.min(dim=1).values)
    # Same seed => same days => a paired comparison between two models.
    again = multistart_greedy_cost_stoch(
        _RandomFeasiblePolicy(), data, cfg,
        generator=torch.Generator().manual_seed(11), **kwargs,
    )
    assert torch.equal(best, again)


def test_cost_mismatch_is_detected() -> None:
    """The guard must fire if a policy returns something other than the env's cost."""

    class _WrongCostPolicy(_RandomFeasiblePolicy):
        def forward(self, env, decode_type="sampling", force_first=None):
            actions, logp, cost = super().forward(env, decode_type, force_first)
            return actions, logp, cost + 1.0

    with pytest.raises(RuntimeError):
        pomo_rollout_stoch(
            _WrongCostPolicy(), _data(batch=2, n_customers=10, seed=12),
            SimulatorConfig(shift_end=_SHIFT), n_starts=3,
        )