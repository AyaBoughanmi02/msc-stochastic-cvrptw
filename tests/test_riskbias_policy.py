"""Tests for variant VGT-R (lateness-risk bias in the decoder).

Load-bearing claims:
  - p_late for the chosen action EQUALS the env's own late_count increment for that step
    (analytic reward). This is the test that checks the arrival-time convention
    (arrival = clock + mean travel time at the departure bin) against the environment,
    instead of assuming it;
  - p_late is in [0, 0.5] for every candidate the mask allows;
  - forcing the training gate to 0 (as every evaluation does) leaves the risk bias in place;
  - with both strengths at 0 the policy computes exactly what D-prime at lambda 0 computes;
  - risk_w is in the high-lr gate optimizer group, trains, and survives a checkpoint
    round trip under the label VGT-R.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from scripts.compare_variants import forced_gate
from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.checkpoints import load_stoch_policy
from src.models.cvrptw_riskbias_policy import (
    RISK_PARAM_NAME,
    CVRPTWRiskBiasPolicy,
    lateness_probability,
)
from src.models.factory import (
    DEFAULT_CVRPTW_RISKBIAS_CONFIG,
    build_cvrptw_gateonly_policy,
    build_cvrptw_riskbias_policy,
)
from src.train.pomo import build_starts
from src.train.pomo_stoch import (
    GATE_PARAM_NAME,
    build_optimizer,
    build_stoch_env,
    is_gate_param,
    train_step_stoch,
)

# Tight windows and a long service time so lateness is actually likely.
_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 3, n_customers: int = 8, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(batch, n_customers, capacity=_CAPACITY, seed=seed,
                           shift_end=_SHIFT, service_time=_SERVICE)


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.6, "eta": 0.6, "omega": 1.0,
                              "shift_end": _SHIFT, **over})


# ------------------------------------------------------- the feature against the reward


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p_late_equals_the_rewards_late_count_increment(seed: int) -> None:
    """Walk several steps (customers AND depot returns) and compare at every step."""
    env = build_stoch_env(_data(seed=seed), _cfg(), 1, 2.0, reward_mode="analytic")
    mask = env.reset()                                               # (B, N) True=infeasible
    rows = torch.arange(env.batch_size)
    checked_nonzero = False
    for _ in range(6):
        p = lateness_probability(env)                                # (B, N)
        action = (~mask).float().argmax(dim=1)                       # first feasible node
        before = env.late_count.clone()                              # (B,)
        mask, done = env.step(action)
        increment = env.late_count - before                          # (B,)
        torch.testing.assert_close(increment, p[rows, action], rtol=1e-5, atol=1e-6)
        checked_nonzero |= bool((increment > 1e-4).any())
        if bool(done.all()):
            break
    assert checked_nonzero, "test instances never risked lateness; tighten the setup"


def test_p_late_is_at_most_one_half_where_the_mask_allows() -> None:
    env = build_stoch_env(_data(seed=4), _cfg(), 1, 2.0, reward_mode="analytic")
    mask = env.reset()
    for _ in range(5):
        p = lateness_probability(env)                                # (B, N)
        assert float(p.min()) >= 0.0
        assert float(p[~mask].max()) <= 0.5 + 1e-5                   # planned arrival on time
        mask, done = env.step((~mask).float().argmax(dim=1))
        if bool(done.all()):
            break


def test_p_late_is_detached_and_shaped_per_candidate() -> None:
    env = build_stoch_env(_data(), _cfg(), 1, 2.0, reward_mode="analytic")
    env.reset()
    p = lateness_probability(env)
    assert p.shape == (env.batch_size, env.num_nodes)
    assert not p.requires_grad


# --------------------------------------------------------------------- the penalty slot


def test_forcing_the_training_gate_off_keeps_the_risk_bias() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy().eval()
    env = build_stoch_env(_data(seed=5), _cfg(), 1, 2.0, reward_mode="sampled")
    env.reset()
    with forced_gate(policy, 0.0):
        penalty = policy._gate_penalty(env, policy._decoder_edge_feat(env))   # (B, N)
    expected = policy.risk_strength() * lateness_probability(env)
    torch.testing.assert_close(penalty, expected)


def test_with_both_strengths_zero_it_is_exactly_dprime_at_lambda_zero() -> None:
    torch.manual_seed(0)
    dp = build_cvrptw_gateonly_policy().eval()
    with torch.no_grad():
        dp.decoder.w_edge.weight.normal_()
    vr = CVRPTWRiskBiasPolicy(dp.encoder, dp.decoder).eval()
    zero = torch.tensor(0.0)
    dp.gate_strength = lambda: zero
    vr.gate_strength = lambda: zero
    vr.risk_strength = lambda: zero
    data, cfg = _data(seed=6), _cfg()
    outs = []
    for policy in (dp, vr):
        env = build_stoch_env(data, cfg, 4, 2.0, reward_mode="analytic")
        with torch.no_grad():
            outs.append(policy(env, "greedy", build_starts(data.coords.size(0), 4,
                                                           data.coords.device)))
    assert torch.equal(outs[0][0], outs[1][0])                       # same actions
    assert torch.equal(outs[0][2], outs[1][2])                       # same cost


def test_the_risk_bias_changes_decisions_when_on() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy({"risk_init": 3.0}).eval()   # alpha ~ 3.05
    data, cfg = _data(batch=8, seed=7), _cfg(beta_var=0.9)
    starts = build_starts(data.coords.size(0), 4, data.coords.device)
    with torch.no_grad(), forced_gate(policy, 0.0):
        on = policy(build_stoch_env(data, cfg, 4, 2.0, reward_mode="analytic"),
                    "greedy", starts)
        policy.risk_strength = lambda: torch.tensor(0.0)
        off = policy(build_stoch_env(data, cfg, 4, 2.0, reward_mode="analytic"),
                     "greedy", starts)
    assert not torch.equal(on[0], off[0])


# ------------------------------------------------------- training, optimizer, loading


def test_risk_param_name_matches_the_optimizer_and_sits_in_the_gate_group() -> None:
    assert is_gate_param(RISK_PARAM_NAME)
    assert is_gate_param(GATE_PARAM_NAME)
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    gate_group = next(g for g in opt.param_groups if g["name"] == "gate")
    assert any(p is policy.risk_w for p in gate_group["params"])
    assert any(p is policy.gate_w for p in gate_group["params"])
    assert gate_group["weight_decay"] == 0.0


def test_vgtr_trains_and_both_scalars_move() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    before = (float(policy.gate_w.detach()), float(policy.risk_w.detach()))
    for seed in (8, 9):
        train_step_stoch(policy, opt, lambda: _data(batch=2, n_customers=8, seed=seed),
                         _cfg(), n_starts=3, tardiness_weight=2.0, reward_mode="analytic")
    after = (float(policy.gate_w.detach()), float(policy.risk_w.detach()))
    assert abs(after[0] - before[0]) > 1e-3
    assert abs(after[1] - before[1]) > 1e-3
    assert "risk_alpha" in policy.policy_metrics()
    assert "softplus_w" in policy.policy_metrics()


def test_loader_rebuilds_vgtr(tmp_path) -> None:
    torch.manual_seed(0)
    config = dict(DEFAULT_CVRPTW_RISKBIAS_CONFIG)
    policy = build_cvrptw_riskbias_policy(config)
    with torch.no_grad():
        policy.risk_w.fill_(1.234)
    path = tmp_path / "VGTR_last.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": config,
                "sim_config": asdict(_cfg()), "tardiness_weight": 2.0}, path)
    loaded = load_stoch_policy(str(path))
    assert loaded.variant == "VGT-R"
    assert isinstance(loaded.policy, CVRPTWRiskBiasPolicy)
    assert float(loaded.policy.risk_w) == pytest.approx(1.234)
    assert hasattr(loaded.policy, "gate_w")                          # seed_study can force it


def test_alpha_starts_at_one() -> None:
    policy = build_cvrptw_riskbias_policy()
    assert float(policy.risk_strength()) == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------------ the controls


def test_tracker_reproduces_env_time_var_with_the_true_bank() -> None:
    """The control path's own variance bookkeeping equals the env's, step by step."""
    from src.models.cvrptw_riskbias_policy import track_accumulated_variance

    data = _data(seed=10)
    env = build_stoch_env(data, _cfg(), 3, 2.0, reward_mode="analytic")
    mask = env.reset(build_starts(data.coords.size(0), 3, data.coords.device))  # forced start
    for _ in range(8):
        acc = track_accumulated_variance(env, env.bank)
        torch.testing.assert_close(acc, env.time_var, rtol=1e-6, atol=1e-8)
        assert torch.equal(track_accumulated_variance(env, env.bank), acc)   # idempotent
        mask, done = env.step((~mask).float().argmax(dim=1))
        if bool(done.all()):
            break


@pytest.mark.parametrize("signal", ["const", "shuffle"])
def test_control_p_late_differs_from_the_true_one_but_stays_a_probability(signal) -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy({"risk_signal": signal}).eval()
    data = _data(seed=11)
    env = build_stoch_env(data, _cfg(), 1, 2.0, reward_mode="analytic")
    mask = env.reset()
    differs = False
    for _ in range(4):
        ctrl = policy.risk_probability(env)
        true = lateness_probability(env)
        assert ctrl.shape == true.shape
        assert float(ctrl.min()) >= 0.0 and float(ctrl.max()) <= 1.0
        differs |= not torch.allclose(ctrl, true)
        mask, done = env.step((~mask).float().argmax(dim=1))
    assert differs


def test_control_uses_zero_accumulated_variance_in_sampled_mode() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy({"risk_signal": "const"}).eval()
    env = build_stoch_env(_data(seed=12), _cfg(), 1, 2.0, reward_mode="sampled")
    mask = env.reset()
    env.step((~mask).float().argmax(dim=1))
    rbank = policy._risk_bank(env)
    expected = lateness_probability(env, rbank, torch.zeros_like(env.time_var))
    torch.testing.assert_close(policy.risk_probability(env), expected)


def test_shuffle_control_leaves_the_global_rng_untouched() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy({"risk_signal": "shuffle"})
    env = build_stoch_env(_data(seed=13), _cfg(), 1, 2.0, reward_mode="analytic")
    env.reset()
    state = torch.get_rng_state()
    policy.risk_probability(env)
    assert torch.equal(torch.get_rng_state(), state)


def test_unknown_risk_signal_is_refused() -> None:
    with pytest.raises(ValueError, match="risk_signal"):
        build_cvrptw_riskbias_policy({"risk_signal": "noise"})


def test_loader_labels_a_control(tmp_path) -> None:
    torch.manual_seed(0)
    config = {**DEFAULT_CVRPTW_RISKBIAS_CONFIG, "risk_signal": "const"}
    policy = build_cvrptw_riskbias_policy(config)
    path = tmp_path / "VGTRc_last.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": config,
                "sim_config": asdict(_cfg()), "tardiness_weight": 2.0}, path)
    loaded = load_stoch_policy(str(path))
    assert loaded.variant == "VGT-R-const"
    assert loaded.policy.risk_signal == "const"


def test_old_vgtr_checkpoints_without_risk_signal_still_load_as_sigma2(tmp_path) -> None:
    torch.manual_seed(0)
    config = {k: v for k, v in DEFAULT_CVRPTW_RISKBIAS_CONFIG.items() if k != "risk_signal"}
    policy = build_cvrptw_riskbias_policy(config)
    path = tmp_path / "VGTR_old_last.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": config,
                "sim_config": asdict(_cfg()), "tardiness_weight": 2.0}, path)
    loaded = load_stoch_policy(str(path))
    assert loaded.variant == "VGT-R"
    assert loaded.policy.risk_signal == "sigma2"


def test_fixed_control_does_not_depend_on_the_days_variance_level() -> None:
    """Same instances, two variance levels: the 'fixed' risk bank must be identical."""
    torch.manual_seed(0)
    policy = build_cvrptw_riskbias_policy({"risk_signal": "fixed"}).eval()
    data = _data(seed=14)
    banks = []
    for beta in (0.2, 0.6):
        env = build_stoch_env(data, _cfg(beta_var=beta), 1, 2.0, reward_mode="analytic")
        env.reset()
        banks.append(policy._risk_bank(env)[..., 1])                   # control variance
    torch.testing.assert_close(banks[0], banks[1], rtol=1e-5, atol=1e-7)
    const = build_cvrptw_riskbias_policy({"risk_signal": "const"}).eval()
    env = build_stoch_env(data, _cfg(beta_var=0.6), 1, 2.0, reward_mode="analytic")
    env.reset()
    assert not torch.allclose(const._risk_bank(env)[..., 1], banks[1])   # const does