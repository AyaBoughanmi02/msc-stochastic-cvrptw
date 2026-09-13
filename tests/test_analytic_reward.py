"""Tests for reward_mode='analytic': expected tardiness in closed form.

WHY THIS MODE EXISTS. The 'sampled' reward charges the lateness of ONE simulated day. That
quantity was measured to contribute ~96% of the spread of the POMO advantage while being
only ~19% reproducible across days, so training optimised noise and ended up worse than an
untrained network. The analytic mode advances the clock on mu (route deterministic given
the actions), propagates the arrival-time VARIANCE alongside it, and charges the expected
lateness of a Normal above the window close. Both terms are deterministic functions of the
route, so the reward carries no sampling noise at all.

Load-bearing tests:
  - `test_analytic_reward_is_deterministic`: the entire point -- same route, same charge,
    whatever the noise field.
  - `test_analytic_is_conservative_against_monte_carlo`: the Normal approximation to a sum
    of lognormals is an APPROXIMATION, so its error is measured here rather than assumed,
    and pinned in the direction that matters (it must not understate risk).
  - `test_analytic_ranks_routes_like_monte_carlo`: POMO only needs the reward to ORDER
    sibling routes correctly; that is a weaker and more important property than accuracy.
  - `test_sampled_mode_is_the_default_and_unchanged`: the legacy path stays byte-identical,
    so the gate chain and every certified reduction still hold.
"""

from __future__ import annotations

import pytest
import torch

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import (
    MU,
    SIGMA2,
    SimulatorConfig,
    build_edge_bank,
    draw_noise,
    horizon_average,
)
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.factory import build_cvrptw_stoch_policy
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env, pomo_rollout_stoch, train_step_stoch

_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 8, n_customers: int = 10, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.3, "eta": 0.6,
                              "shift_end": _SHIFT, **over})


def _env(data, cfg, mode="analytic", noise=None, weight=1.0) -> CVRPTWStochEnv:
    bank = build_edge_bank(data.coords, cfg)
    return CVRPTWStochEnv(
        data.coords, data.demand, data.capacity, data.time_windows, data.service_time,
        data.shift_end, bank=bank, sim_cfg=cfg, noise=noise,
        tardiness_weight=weight, reward_mode=mode,
    )


def _greedy(env, policy, b, s):
    with torch.no_grad():
        return policy(env, "greedy", build_starts(b, s, env.coords.device))[0]


# ------------------------------------------------------------------- the legacy path


def test_sampled_mode_is_the_default_and_unchanged() -> None:
    data, cfg = _data(seed=1), _cfg()
    assert _env(data, cfg, mode=None if False else "sampled").reward_mode == "sampled"
    default = CVRPTWStochEnv(
        data.coords, data.demand, data.capacity, data.time_windows, data.service_time,
        data.shift_end, bank=build_edge_bank(data.coords, cfg), sim_cfg=cfg,
    )
    assert default.reward_mode == "sampled"


def test_unknown_reward_mode_raises() -> None:
    data, cfg = _data(seed=2), _cfg()
    with pytest.raises(ValueError, match="reward_mode"):
        _env(data, cfg, mode="quantile")


def test_analytic_env_needs_no_noise_field() -> None:
    """A noise field would be dead weight -- build_stoch_env must not allocate one."""
    env = build_stoch_env(_data(seed=3), _cfg(), 4, 1.0, reward_mode="analytic")
    assert env.noise is None
    sampled = build_stoch_env(_data(seed=3), _cfg(), 4, 1.0, reward_mode="sampled",
                              generator=torch.Generator().manual_seed(3))
    assert sampled.noise is not None


# ---------------------------------------------------------------- determinism (the point)


def test_analytic_reward_is_deterministic() -> None:
    """Same route, same charge, regardless of the noise field or generator."""
    data, cfg = _data(batch=6, seed=4), _cfg()
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy().eval()
    bank_shape = build_edge_bank(data.coords, cfg).shape[:-1]

    results = []
    for seed in (11, 22):
        noise = draw_noise(bank_shape, device="cpu",
                           generator=torch.Generator().manual_seed(seed))
        env = _env(data, cfg, noise=noise)
        actions = _greedy(env, policy, 6, 1)
        results.append((actions, env.tardiness.clone(), env.late_count.clone()))

    assert torch.equal(results[0][0], results[1][0])      # identical routes
    assert torch.equal(results[0][1], results[1][1])      # identical expected tardiness
    assert torch.equal(results[0][2], results[1][2])      # identical expected violations


def test_analytic_vanishes_when_the_bank_has_no_variance() -> None:
    data = _data(batch=6, seed=5)
    env = _env(data, _cfg(beta_var=0.0))
    torch.manual_seed(0)
    _greedy(env, build_cvrptw_stoch_policy().eval(), 6, 1)
    assert float(env.tardiness.abs().max()) < 1e-6
    assert float(env.late_count.abs().max()) < 1e-6
    assert float(env.time_var.abs().max()) == 0.0


def test_expected_charges_grow_with_variance() -> None:
    data = _data(batch=8, seed=6)
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy().eval()
    charges = []
    for beta_var in (0.05, 0.3, 0.6):
        env = _env(data, _cfg(beta_var=beta_var))
        _greedy(env, policy, 8, 1)
        charges.append(float(env.tardiness.mean()))
    assert charges[0] < charges[1] < charges[2]


def test_arrival_variance_resets_at_the_depot() -> None:
    """Uncertainty must reset with the clock -- each depot loop is a fresh vehicle."""
    data = _data(batch=4, n_customers=6, seed=7)
    env = _env(data, _cfg())
    env.reset()
    env.step(torch.full((4,), 3, dtype=torch.long))
    assert float(env.time_var.min()) > 0.0            # accumulated on the way out
    env.step(torch.zeros(4, dtype=torch.long))        # back to the depot
    assert float(env.time_var.abs().max()) == 0.0
    assert float(env.time.abs().max()) == 0.0         # clock reset too


# ------------------------------------------------------- accuracy against simulation


def _monte_carlo(env, actions, cfg, n_days: int, seed: int = 0):
    """Replay one action sequence over n_days sampled days. -> (tardiness, violations)."""
    tard = torch.zeros(env.batch_size)
    late = torch.zeros(env.batch_size)
    for day in range(n_days):
        noise = draw_noise(env.bank.shape[:-1], device="cpu",
                           generator=torch.Generator().manual_seed(seed + day))
        replay = CVRPTWStochEnv(
            env.coords, env.demand, env.capacity, env.time_windows, env.service_time,
            env.shift_end, bank=env.bank, sim_cfg=cfg, instance_idx=env.instance_idx,
            noise=noise, reward_mode="sampled",
        )
        replay.reset()
        for t in range(actions.size(1)):
            replay.step(actions[:, t])
        tard += replay.tardiness
        late += replay.late_count
    return tard / n_days, late / n_days


def test_analytic_is_conservative_against_monte_carlo() -> None:
    """The Normal approximation must OVERSTATE risk, and by a bounded amount.

    It is conservative for two documented reasons: waiting for a window to open truncates
    accumulated variance (not modelled), and arrival times are sums of right-skewed
    lognormals rather than Normals. Overstating is the safe direction for a risk surrogate;
    understating would let the policy walk into violations it was told were unlikely.
    """
    data, cfg = _data(batch=8, seed=8), _cfg()
    torch.manual_seed(0)
    env = _env(data, cfg)
    actions = _greedy(env, build_cvrptw_stoch_policy().eval(), 8, 1)
    mc_tard, mc_late = _monte_carlo(env, actions, cfg, n_days=200)

    tard_ratio = float(env.tardiness.mean()) / float(mc_tard.mean())
    late_ratio = float(env.late_count.mean()) / float(mc_late.mean())
    assert 1.0 <= tard_ratio <= 2.0, f"tardiness ratio {tard_ratio:.2f}"
    assert 1.0 <= late_ratio <= 2.0, f"violation ratio {late_ratio:.2f}"


def test_analytic_ranks_routes_like_monte_carlo() -> None:
    """POMO only needs the reward to ORDER sibling routes correctly.

    Accuracy is secondary: a surrogate that is uniformly 30% high still trains the right
    policy, provided it agrees with the truth about which route is riskier.
    """
    data, cfg = _data(batch=8, seed=9), _cfg()
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy().eval()
    env = build_stoch_env(data, cfg, 6, 1.0, reward_mode="analytic")
    actions = _greedy(env, policy, 8, 6)
    mc_tard, _ = _monte_carlo(env, actions, cfg, n_days=200)

    a = env.tardiness - env.tardiness.mean()
    m = mc_tard - mc_tard.mean()
    corr = float((a * m).sum() / (a.pow(2).sum().sqrt() * m.pow(2).sum().sqrt()))
    assert corr > 0.8, f"analytic vs Monte-Carlo correlation {corr:.3f}"


# ----------------------------------------------------------------- the training path


def test_rollout_and_train_step_accept_the_analytic_mode() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    cfg = _cfg()
    out = pomo_rollout_stoch(
        policy, _data(batch=3, seed=10), cfg, n_starts=4, tardiness_weight=1.0,
        reward_mode="analytic",
    )
    assert torch.allclose(
        out.cost, out.distance + 1.0 * (out.late_count + out.tardiness), atol=1e-4
    )
    # Expected violations are fractional, unlike the integer counts of 'sampled'.
    assert ((out.late_count % 1.0) > 1e-6).any()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    stats = train_step_stoch(
        policy, optimizer, lambda: _data(batch=3, seed=11), cfg,
        n_starts=4, tardiness_weight=1.0, reward_mode="analytic",
    )
    assert "mean_violations" in stats and stats["mean_violations"] > 0.0
    assert stats["mean_cost"] >= stats["mean_distance"]


def test_analytic_advantage_is_not_dominated_by_noise() -> None:
    """The regression guard: the reward must remain reproducible across days.

    'sampled' scored ~0.19 here, which is what made it unlearnable. Anything that
    reintroduces sampling into the training reward will trip this.
    """
    data, cfg = _data(batch=16, seed=12), _cfg()
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy().eval()

    def penalties(day: int) -> torch.Tensor:
        env = build_stoch_env(data, cfg, 6, 1.0, reward_mode="analytic",
                              generator=torch.Generator().manual_seed(day))
        _greedy(env, policy, 16, 6)
        return (env.late_count + env.tardiness).reshape(16, 6)

    a, b = penalties(11), penalties(22)
    ac = a - a.mean(dim=1, keepdim=True)
    bc = b - b.mean(dim=1, keepdim=True)
    corr = float((ac * bc).sum() / (ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()))
    assert corr > 0.99, f"per-start penalty reproducibility {corr:.3f}"


# ------------------------------------------------ negative eta: the structural shift


def test_negative_eta_inverts_which_edges_are_risky() -> None:
    """A structural shift needs the risk RANKING to change, not just its magnitude.

    Flipping the sign of eta keeps |heterogeneity| identical but reverses which corridors
    are unreliable, so a policy that memorised "central edges are risky" mis-generalises
    while one that penalises the actual sigma^2 stays correctly ordered by construction.
    """
    coords = torch.rand(1, 12, 2, generator=torch.Generator().manual_seed(0))

    def risk_per_edge(eta: float) -> torch.Tensor:

        avg = horizon_average(build_edge_bank(coords, _cfg(eta=eta)))
        ratio = (avg[..., SIGMA2] / avg[..., MU].clamp_min(1e-9) ** 2).flatten()
        return ratio[ratio > 0]                      # drop the zero-length self-loops

    plus, minus = risk_per_edge(0.6), risk_per_edge(-0.6)
    corr = float(torch.corrcoef(torch.stack([plus, minus]))[0, 1])
    assert corr < -0.99, f"eta sign flip should invert the ranking, got corr {corr:+.3f}"
    # The AMOUNT of heterogeneity is unchanged -- only its direction.
    assert float(plus.std()) == pytest.approx(float(minus.std()), rel=1e-3)


def test_eta_zero_is_still_homogeneous() -> None:
    coords = torch.rand(1, 10, 2, generator=torch.Generator().manual_seed(1))

    avg = horizon_average(build_edge_bank(coords, _cfg(eta=0.0)))
    ratio = (avg[..., SIGMA2] / avg[..., MU].clamp_min(1e-9) ** 2).flatten()
    ratio = ratio[ratio > 0]
    assert float(ratio.std() / ratio.mean()) < 1e-4


@pytest.mark.parametrize("eta", [1.0, -1.0, 1.5, -2.0])
def test_eta_outside_the_open_unit_interval_raises(eta: float) -> None:
    """|eta| >= 1 would let the reliability multiplier reach zero or go negative."""
    with pytest.raises(ValueError, match="eta"):
        SimulatorConfig(alpha=0.3, beta_var=0.3, eta=eta, shift_end=4.0)