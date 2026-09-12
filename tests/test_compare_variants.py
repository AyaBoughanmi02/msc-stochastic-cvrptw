"""Tests for the paired B-vs-D comparison and the inference-time gate sweep.

Load-bearing tests:
  - `test_interval_straddling_zero_is_not_significant`: the entire reason this tool exists
    is to stop a difference of ~0.06 being reported as a finding when it could be noise.
    If `significant` were ever True for an interval containing zero, the tool would launder
    noise into a result.
  - `test_forced_gate_restores_the_learned_value`: the sweep must never write back to the
    policy -- every row of the sweep has to start from the same learned weights.
"""

from __future__ import annotations

import math

import pytest
import torch

from scripts.compare_variants import forced_gate, paired_difference
from src.models.factory import build_cvrptw_gated_policy, build_cvrptw_stoch_policy


# ------------------------------------------------------------------- paired statistics


def test_paired_difference_matches_hand_computation() -> None:
    cand = torch.tensor([1.0, 2.0, 3.0, 4.0])
    base = torch.tensor([1.5, 2.5, 3.5, 4.5])
    p = paired_difference(cand, base)
    assert p["mean"] == pytest.approx(-0.5)
    # Every difference is identical, so the spread -- and the interval -- is zero.
    assert p["se"] == pytest.approx(0.0, abs=1e-12)
    assert p["significant"]


def test_interval_straddling_zero_is_not_significant() -> None:
    """A large per-instance spread must swamp a small mean difference."""
    torch.manual_seed(0)
    base = torch.randn(200) * 3.0
    cand = base + torch.randn(200) * 3.0 + 0.01      # a tiny, noise-swamped edge
    p = paired_difference(cand, base)
    assert p["lo"] < 0.0 < p["hi"]
    assert not p["significant"]


def test_pairing_detects_what_unpaired_means_would_miss() -> None:
    """The point of pairing: instance difficulty cancels, so a small edge stays visible."""
    torch.manual_seed(1)
    difficulty = torch.randn(256) * 5.0              # huge instance-to-instance spread
    base = difficulty + torch.randn(256) * 0.05
    cand = difficulty + torch.randn(256) * 0.05 - 0.20   # consistently 0.2 cheaper
    p = paired_difference(cand, base)
    assert p["mean"] == pytest.approx(-0.2, abs=0.02)
    assert p["significant"]
    # Unpaired, the same data would be hopeless: the naive SE is ~50x larger.
    naive_se = float(base.std(unbiased=True)) / math.sqrt(256)
    assert naive_se > 10 * p["se"]


def test_sign_convention_is_candidate_minus_baseline() -> None:
    cheaper = paired_difference(torch.ones(10), torch.full((10,), 2.0))
    assert cheaper["mean"] < 0.0                     # negative = candidate cheaper


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="align"):
        paired_difference(torch.ones(5), torch.ones(6))


# ------------------------------------------------------------------ the gate override


def test_forced_gate_overrides_then_restores() -> None:
    policy = build_cvrptw_gated_policy()
    with torch.no_grad():
        policy.gate_w.fill_(1.3)
    learned = float(policy.gate_strength().detach())

    with forced_gate(policy, 4.0):
        assert float(policy.gate_strength().detach()) == pytest.approx(4.0)
    assert float(policy.gate_strength().detach()) == pytest.approx(learned)


def test_forced_gate_reaches_exactly_zero() -> None:
    """softplus has no finite pre-image of 0, so the METHOD is patched, not the parameter."""
    policy = build_cvrptw_gated_policy()
    with forced_gate(policy, 0.0):
        assert float(policy.gate_strength().detach()) == 0.0
    assert float(policy.gate_strength().detach()) > 0.0


def test_forced_gate_restores_the_learned_value_even_on_error() -> None:
    policy = build_cvrptw_gated_policy()
    before = float(policy.gate_strength().detach())
    with pytest.raises(RuntimeError):
        with forced_gate(policy, 9.0):
            raise RuntimeError("boom")
    assert float(policy.gate_strength().detach()) == pytest.approx(before)


def test_forced_gate_actually_changes_the_penalties() -> None:
    """The override must reach both stages, not just the reported scalar."""
    from src.data.cvrptw_generator import generate_cvrptw
    from src.data.simulator import SimulatorConfig
    from src.train.pomo_stoch import build_stoch_env

    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    data = generate_cvrptw(2, 8, capacity=20, seed=1, shift_end=4.0, service_time=0.2)
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.3, eta=0.6, shift_end=4.0)
    env = build_stoch_env(data, cfg, 3, 1.0, reward_mode="analytic")
    env.reset()
    enc = policy._encoder_inputs(env)
    feat = policy._decoder_edge_feat(env)

    with forced_gate(policy, 0.0):
        assert float(policy._encoder_gate_penalty(env, enc).detach().abs().max()) == 0.0
        assert float(policy._gate_penalty(env, feat).detach().abs().max()) == 0.0
    assert float(policy._encoder_gate_penalty(env, enc).detach().abs().max()) > 0.0


def test_gateless_policy_has_no_gate_to_sweep() -> None:
    assert not hasattr(build_cvrptw_stoch_policy(), "gate_w")


# ------------------------------------------------- domain randomisation over beta_var


def test_train_step_accepts_a_resampled_simulator_config() -> None:
    """Domain randomisation: sim_cfg may be a callable, resampled per mini-batch."""
    from src.data.cvrptw_generator import generate_cvrptw
    from src.data.simulator import SimulatorConfig
    from src.train.pomo_stoch import build_optimizer, train_step_stoch

    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    seen: list[float] = []

    def sampler() -> SimulatorConfig:
        beta = 0.15 + 0.45 * float(torch.rand(1))
        seen.append(beta)
        return SimulatorConfig(alpha=0.3, beta_var=beta, eta=0.6, shift_end=4.0)

    def batch():
        return generate_cvrptw(2, 10, capacity=20, seed=None, shift_end=4.0,
                               service_time=0.2)

    stats = train_step_stoch(
        policy, opt, batch, sampler, accum_steps=3, n_starts=3,
        tardiness_weight=2.0, reward_mode="analytic",
    )
    assert len(seen) == 3                      # resampled once per mini-batch
    assert len(set(seen)) == 3                 # genuinely different levels
    assert all(0.15 <= b <= 0.60 for b in seen)
    assert stats["mean_cost"] == stats["mean_cost"]     # not NaN


def test_a_fixed_config_still_works_unchanged() -> None:
    from src.data.cvrptw_generator import generate_cvrptw
    from src.data.simulator import SimulatorConfig
    from src.train.pomo_stoch import build_optimizer, train_step_stoch

    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    stats = train_step_stoch(
        policy, opt,
        lambda: generate_cvrptw(2, 10, capacity=20, shift_end=4.0, service_time=0.2),
        SimulatorConfig(alpha=0.3, beta_var=0.3, eta=0.6, shift_end=4.0),
        n_starts=3, tardiness_weight=2.0, reward_mode="analytic",
    )
    assert "mean_cost" in stats


# ------------------------------------------------------------- multi-seed aggregation


def test_seed_interval_widens_with_fewer_seeds() -> None:
    """A t-interval on 2-3 numbers must be honestly wide, not falsely tight."""
    from scripts.seed_study import _interval

    values = [-0.40, -0.55, -0.48]
    mean3, half3, sd3 = _interval(values)
    assert mean3 == pytest.approx(-0.4767, abs=1e-3)
    mean2, half2, _ = _interval(values[:2])
    assert half2 > half3                       # fewer seeds => wider, never tighter
    _, half1, _ = _interval(values[:1])
    assert half1 == float("inf")               # one seed says nothing about seed spread


def test_seed_interval_uses_the_t_distribution_not_normal() -> None:
    """With 3 samples the normal approximation would understate the interval by ~60%."""
    from scripts.seed_study import _interval

    values = [1.0, 2.0, 3.0]
    _, half, sd = _interval(values)
    normal_half = 1.959964 * sd / math.sqrt(3)
    assert half > 1.5 * normal_half