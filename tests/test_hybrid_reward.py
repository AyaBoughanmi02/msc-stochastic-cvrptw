"""Tests for reward_mode='hybrid' in CVRPTWStochEnv.

hybrid = REAL (sampled) clock + each arc charged its EXPECTED lateness given the departure
clock (closed form for the lognormal travel time). It must be:
  1. exact:      the closed form equals a Monte-Carlo estimate of the same quantity;
  2. unbiased:   on a fixed route, its mean over many days equals the 'sampled' mean;
  3. less noisy: its spread over days is clearly smaller than 'sampled';
  4. on the real clock: the clock path is identical to 'sampled' for the same noise.
"""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig, apply_noise, build_edge_bank
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv, _lognormal_lateness

SIM = SimulatorConfig(alpha=0.3, beta_var=0.6, eta=0.6, omega=1.0)
ROUTE = [3, 5, 0, 1, 2, 0, 4, 6, 7, 0, 8, 9, 10, 0]      # all 10 customers, 4 trips


def _rollout(mode: str, days: int, seed: int = 0) -> tuple[dict[str, torch.Tensor], list]:
    """Drive ROUTE on `days` independent days in one batch. Returns costs and clock path."""
    data = generate_cvrptw(1, 10, 20, seed=3)
    rep = lambda t: t.repeat(days, *([1] * (t.dim() - 1)))       # noqa: E731
    bank = build_edge_bank(data.coords, SIM).repeat(days, 1, 1, 1, 1)
    noise = torch.randn(days, 11, 11, SIM.n_bins,
                        generator=torch.Generator().manual_seed(seed))
    env = CVRPTWStochEnv(rep(data.coords), rep(data.demand), data.capacity,
                         rep(data.time_windows), rep(data.service_time), data.shift_end,
                         bank=bank, sim_cfg=SIM, noise=noise, tardiness_weight=2.0,
                         reward_mode=mode)
    actions = torch.tensor([ROUTE]).repeat(days, 1)
    env.reset()
    clocks = []
    for t in range(actions.size(1)):
        env.step(actions[:, t])
        clocks.append(env.time.clone())
    return env.cost_components(actions), clocks


def test_closed_form_matches_monte_carlo() -> None:
    mu = torch.tensor([0.3, 0.3, 0.5, 0.2, 0.4, 0.0, 0.3])
    s2 = torch.tensor([0.02, 0.05, 0.1, 0.0, 0.03, 0.0, 0.04])
    slack = torch.tensor([0.3, 0.5, 0.2, 0.1, -0.1, 0.5, 0.0])
    exp_late, p_late = _lognormal_lateness(mu, s2, slack)
    z = torch.randn(400_000, 1, generator=torch.Generator().manual_seed(0))
    x = apply_noise(mu, s2, z)                                    # (400000, 7)
    assert torch.allclose(exp_late, (x - slack).clamp_min(0).mean(0), atol=3e-3)
    assert torch.allclose(p_late, (x > slack).float().mean(0), atol=3e-3)


def test_degenerate_cases() -> None:
    exp_late, p_late = _lognormal_lateness(
        torch.tensor([0.2, 0.0, 0.3]), torch.tensor([0.0, 0.0, 0.05]),
        torch.tensor([0.1, 1.0, -0.2]))
    assert torch.allclose(exp_late, torch.tensor([0.1, 0.0, 0.5]), atol=1e-6)
    assert torch.allclose(p_late, torch.tensor([1.0, 0.0, 1.0]))


def test_hybrid_follows_the_real_clock() -> None:
    _, c_sampled = _rollout("sampled", days=64)
    _, c_hybrid = _rollout("hybrid", days=64)
    for a, b in zip(c_sampled, c_hybrid):
        assert torch.equal(a, b)


def test_hybrid_is_unbiased_and_less_noisy() -> None:
    sampled, _ = _rollout("sampled", days=20_000)
    hybrid, _ = _rollout("hybrid", days=20_000)
    for key in ("tardiness", "late_count"):
        s, h = sampled[key], hybrid[key]
        se = (s.var() / s.numel() + h.var() / h.numel()).sqrt()
        assert abs(float(s.mean() - h.mean())) < 4 * float(se), key   # same expectation
        assert float(h.std()) < 0.75 * float(s.std()), key            # clearly less noise
    assert torch.equal(sampled["distance"], hybrid["distance"])
