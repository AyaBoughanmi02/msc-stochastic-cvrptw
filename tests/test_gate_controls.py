"""Tests for the training-time gate controls and the gate schedule.

Load-bearing claims:
  - every control keeps the per-instance MAGNITUDE of sigma^2 and zero self-loops, so the
    ladder compares PATTERNS of penalty, not amounts;
  - with the gate OFF (lambda = 0) a control computes exactly what D-prime computes: the
    controls differ from M only in what happened during TRAINING;
  - shuffling does not consume the global RNG (same training-instance stream as D-prime);
  - the loader rebuilds a control from its checkpoint, labelled D'-<signal>;
  - --gate-fixed is exact and the schedule turns the gate fully off.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import MU, SIGMA2, SimulatorConfig, build_edge_bank
from src.eval.checkpoints import load_stoch_policy
from src.models.cvrptw_gate_controls import (
    CONTROL_SIGNALS,
    CVRPTWGateControlPolicy,
    gate_signal_tensor,
    softplus_inverse,
)
from src.models.factory import (
    DEFAULT_CVRPTW_GATEONLY_CONFIG,
    build_cvrptw_gate_control_policy,
    build_cvrptw_gateonly_policy,
)
from src.train.gate_schedule import gate_scale
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_optimizer, build_stoch_env, train_step_stoch

_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 3, n_customers: int = 8, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(batch, n_customers, capacity=_CAPACITY, seed=seed,
                           shift_end=_SHIFT, service_time=_SERVICE)


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.3, "eta": 0.6, "omega": 1.0,
                              "shift_end": _SHIFT, **over})


def _offdiag_mean(x: torch.Tensor) -> torch.Tensor:
    n = x.size(1)
    off = (1.0 - torch.eye(n)).view(1, n, n, 1)
    return (x * off).sum(dim=(1, 2, 3)) / (off.sum() * x.size(-1))


# ------------------------------------------------------------- the signals themselves


@pytest.mark.parametrize("omega", [0.0, 1.0])
@pytest.mark.parametrize("signal", CONTROL_SIGNALS)
def test_controls_keep_magnitude_symmetry_and_zero_diagonal(signal: str, omega: float) -> None:
    bank = build_edge_bank(_data().coords, _cfg(omega=omega))         # (B, N, N, K, 2)
    sigma2 = bank[..., SIGMA2]
    gate = gate_signal_tensor(bank, signal, torch.Generator().manual_seed(0))
    assert gate.shape == sigma2.shape
    torch.testing.assert_close(_offdiag_mean(gate), _offdiag_mean(sigma2),
                               rtol=1e-5, atol=1e-7)
    assert torch.equal(gate, gate.transpose(1, 2))                    # symmetric
    n = gate.size(1)
    assert float(gate[:, torch.arange(n), torch.arange(n)].abs().max()) == 0.0
    assert not torch.equal(gate, sigma2)                              # it IS a control


def test_shuffle_permutes_the_instances_own_values() -> None:
    bank = build_edge_bank(_data().coords, _cfg())
    sigma2 = bank[..., SIGMA2]
    gate = gate_signal_tensor(bank, "shuffle", torch.Generator().manual_seed(0))
    iu = torch.triu_indices(sigma2.size(1), sigma2.size(1), offset=1)
    for b in range(sigma2.size(0)):                                   # same multiset
        assert torch.equal(gate[b, iu[0], iu[1]].sort(dim=0).values,
                           sigma2[b, iu[0], iu[1]].sort(dim=0).values)


def test_const_has_no_contrast_and_dist_follows_length() -> None:
    bank = build_edge_bank(_data().coords, _cfg())
    n = bank.size(1)
    off = ~torch.eye(n, dtype=torch.bool)
    const = gate_signal_tensor(bank, "const")
    for b in range(const.size(0)):
        vals = const[b][off]
        assert torch.allclose(vals, vals[0].expand_as(vals))
    dist = gate_signal_tensor(bank, "dist")
    ratio = dist[..., 0][:, off] / bank[..., 0, MU][:, off].square()   # constant per instance
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), rtol=1e-4)


def test_unknown_signal_is_refused() -> None:
    bank = build_edge_bank(_data().coords, _cfg())
    with pytest.raises(ValueError, match="unknown gate signal"):
        gate_signal_tensor(bank, "noise")
    torch.manual_seed(0)
    base = build_cvrptw_gateonly_policy()
    with pytest.raises(ValueError, match="gate_signal must be one of"):
        CVRPTWGateControlPolicy(base.encoder, base.decoder, gate_signal="sigma2")


# ---------------------------------------------------- the policy, gate OFF vs gate ON


def _greedy(policy, data: CVRPTWData, cfg: SimulatorConfig):
    env = build_stoch_env(data, cfg, 4, 2.0, reward_mode="analytic")
    with torch.no_grad():
        return policy(env, "greedy", build_starts(data.coords.size(0), 4, data.coords.device))


@pytest.mark.parametrize("signal", CONTROL_SIGNALS)
def test_with_the_gate_off_a_control_is_exactly_dprime(signal: str) -> None:
    torch.manual_seed(0)
    dp = build_cvrptw_gateonly_policy().eval()
    with torch.no_grad():
        dp.decoder.w_edge.weight.normal_()
    ctrl = CVRPTWGateControlPolicy(dp.encoder, dp.decoder, gate_signal=signal).eval()
    zero = torch.tensor(0.0)
    dp.gate_strength = lambda: zero                                   # lambda forced to 0
    ctrl.gate_strength = lambda: zero
    data, cfg = _data(seed=1), _cfg()
    a_dp, _, c_dp = _greedy(dp, data, cfg)
    a_ct, _, c_ct = _greedy(ctrl, data, cfg)
    assert torch.equal(a_dp, a_ct)
    assert torch.equal(c_dp, c_ct)


@pytest.mark.parametrize("signal", CONTROL_SIGNALS)
def test_with_the_gate_on_a_control_penalises_a_different_pattern(signal: str) -> None:
    torch.manual_seed(0)
    dp = build_cvrptw_gateonly_policy()
    ctrl = CVRPTWGateControlPolicy(dp.encoder, dp.decoder, gate_signal=signal)
    env = build_stoch_env(_data(seed=2), _cfg(), 3, 2.0, reward_mode="analytic")
    env.reset()
    enc = dp._encoder_inputs(env)
    assert not torch.equal(dp._encoder_gate_penalty(env, enc).detach(),
                           ctrl._encoder_gate_penalty(env, enc).detach())
    assert ctrl._encoder_gate_penalty(env, enc).shape == (enc.edge_index.size(1),)
    assert ctrl._gate_penalty(env, None).shape == (env.batch_size, env.num_nodes)


def test_one_rollout_sees_one_shuffle_and_the_global_rng_is_untouched() -> None:
    torch.manual_seed(0)
    base = build_cvrptw_gateonly_policy()
    ctrl = CVRPTWGateControlPolicy(base.encoder, base.decoder, gate_signal="shuffle")
    env = build_stoch_env(_data(seed=3), _cfg(), 3, 2.0, reward_mode="analytic")
    env.reset()
    state = torch.get_rng_state()
    first = ctrl._gate_bank(env)
    assert ctrl._gate_bank(env) is first                              # cached per env
    assert torch.equal(torch.get_rng_state(), state)                  # private generator
    env2 = build_stoch_env(_data(seed=3), _cfg(), 3, 2.0, reward_mode="analytic")
    assert not torch.equal(ctrl._gate_bank(env2), first)              # fresh draw


def test_a_control_trains_and_its_gate_moves() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_gate_control_policy({"gate_signal": "shuffle"})
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    before = float(policy.gate_w.detach())
    train_step_stoch(policy, opt, lambda: _data(batch=2, n_customers=8, seed=4), _cfg(),
                     n_starts=3, tardiness_weight=2.0, reward_mode="analytic")
    assert abs(float(policy.gate_w.detach()) - before) > 1e-3


def test_loader_rebuilds_a_control(tmp_path) -> None:
    torch.manual_seed(0)
    config = {**DEFAULT_CVRPTW_GATEONLY_CONFIG, "gate_signal": "const"}
    policy = build_cvrptw_gate_control_policy(config)
    path = tmp_path / "DpConst_last.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": config,
                "sim_config": asdict(_cfg()), "tardiness_weight": 2.0}, path)
    loaded = load_stoch_policy(str(path))
    assert loaded.variant == "D'-const"
    assert isinstance(loaded.policy, CVRPTWGateControlPolicy)
    assert loaded.policy.gate_signal == "const"


def test_plain_dprime_checkpoints_still_load_as_dprime(tmp_path) -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_gateonly_policy()
    path = tmp_path / "Dp_last.pt"
    torch.save({"model_state": policy.state_dict(),
                "model_config": dict(DEFAULT_CVRPTW_GATEONLY_CONFIG),
                "sim_config": asdict(_cfg()), "tardiness_weight": 2.0}, path)
    loaded = load_stoch_policy(str(path))
    assert loaded.variant == "D'"
    assert not isinstance(loaded.policy, CVRPTWGateControlPolicy)


# ------------------------------------------------------- frozen gate and the schedule


@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 4.0])
def test_softplus_inverse_round_trips(lam: float) -> None:
    w = torch.tensor(softplus_inverse(lam))
    assert float(torch.nn.functional.softplus(w)) == pytest.approx(lam, rel=1e-6)


def test_a_frozen_gate_does_not_move() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_gateonly_policy({"gate_init": softplus_inverse(2.0)})
    policy.gate_w.requires_grad_(False)
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    before = policy.gate_w.detach().clone()
    for seed in (5, 6):
        train_step_stoch(policy, opt, lambda: _data(batch=2, n_customers=8, seed=seed),
                         _cfg(), n_starts=3, tardiness_weight=2.0, reward_mode="analytic")
    assert torch.equal(policy.gate_w.detach(), before)


def test_gate_schedules() -> None:
    assert gate_scale(1, 100, "constant", None) == 1.0
    assert gate_scale(10, 100, "early", 0.33) == 1.0
    assert gate_scale(33, 100, "early", 0.33) == 0.0
    assert gate_scale(100, 100, "early", 0.33) == 0.0
    assert gate_scale(1, 100, "anneal", 0.5) == pytest.approx(0.98)
    assert gate_scale(25, 100, "anneal", 0.5) == pytest.approx(0.5)
    assert gate_scale(50, 100, "anneal", 0.5) == 0.0
    with pytest.raises(ValueError):
        gate_scale(1, 100, "early", None)
    with pytest.raises(ValueError):
        gate_scale(1, 100, "sometimes", 0.5)