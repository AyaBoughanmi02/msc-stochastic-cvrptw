"""Tests for variant M0 (src/models/cvrptw_meanonly_policy.py).

The claims the thesis makes about M0, each checked here:
  1. It has no gate parameter, and is otherwise D-prime's architecture exactly.
  2. Under the same seed it starts from D-prime's weights, bit for bit.
  3. With D-prime's weights it computes exactly what D-prime computes at lambda = 0.
  4. Its actions do not depend on beta_var at all (sigma^2 reaches no decision).
  5. It passes the gate chain, so train_stoch's periodic checks cannot kill a night run.
  6. The checkpoint loader recognises it from its weights alone.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.checkpoints import load_stoch_policy
from src.models.factory import (
    DEFAULT_CVRPTW_MEANONLY_CONFIG,
    build_cvrptw_gateonly_policy,
    build_cvrptw_meanonly_policy,
)
from src.models.gate_checks import assert_gate_chain_inert
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env

N_CUSTOMERS = 8
BATCH = 4


def _cfg(beta: float) -> SimulatorConfig:
    return SimulatorConfig(beta_var=beta, eta=0.6, omega=1.0)


def _data():
    return generate_cvrptw(BATCH, N_CUSTOMERS, 20, seed=0)


def _greedy_actions(policy, data, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    env = build_stoch_env(data, cfg, N_CUSTOMERS, 2.0, reward_mode="analytic")
    starts = build_starts(BATCH, N_CUSTOMERS, data.coords.device)
    with torch.no_grad():
        actions, _, cost = policy(env, "greedy", starts)
    return actions, cost


def test_no_gate_parameter_and_dprime_architecture() -> None:
    m0 = build_cvrptw_meanonly_policy()
    dp = build_cvrptw_gateonly_policy()
    assert not hasattr(m0, "gate_w")
    assert set(m0.state_dict()) == set(dp.state_dict()) - {"gate_w"}
    for key, tensor in m0.state_dict().items():
        assert tensor.shape == dp.state_dict()[key].shape, key


def test_same_seed_same_initial_weights_as_dprime() -> None:
    torch.manual_seed(0)
    m0 = build_cvrptw_meanonly_policy()
    torch.manual_seed(0)
    dp = build_cvrptw_gateonly_policy()
    for key, tensor in m0.state_dict().items():
        assert torch.equal(tensor, dp.state_dict()[key]), key


def test_equals_dprime_at_lambda_zero() -> None:
    torch.manual_seed(0)
    dp = build_cvrptw_gateonly_policy().eval()
    with torch.no_grad():
        dp.decoder.w_edge.weight.normal_()               # non-trivial, as after training
    m0 = build_cvrptw_meanonly_policy().eval()
    m0.load_state_dict({k: v for k, v in dp.state_dict().items() if k != "gate_w"})
    zero = torch.tensor(0.0)
    dp.gate_strength = lambda: zero                       # lambda forced to exactly 0
    data, cfg = _data(), _cfg(0.6)
    a_dp, c_dp = _greedy_actions(dp, data, cfg)
    a_m0, c_m0 = _greedy_actions(m0, data, cfg)
    assert torch.equal(a_dp, a_m0)
    assert torch.equal(c_dp, c_m0)


def test_actions_invariant_to_beta_var() -> None:
    torch.manual_seed(0)
    m0 = build_cvrptw_meanonly_policy().eval()
    data = _data()
    a_lo, _ = _greedy_actions(m0, data, _cfg(0.15))
    a_hi, _ = _greedy_actions(m0, data, _cfg(0.9))
    assert torch.equal(a_lo, a_hi)


def test_passes_gate_chain() -> None:
    torch.manual_seed(0)
    m0 = build_cvrptw_meanonly_policy()
    assert_gate_chain_inert(m0, _data(), _cfg(0.3), seed=1)


def test_loader_recognises_m0(tmp_path) -> None:
    torch.manual_seed(0)
    m0 = build_cvrptw_meanonly_policy()
    path = tmp_path / "M0_test_last.pt"
    torch.save({
        "model_state": m0.state_dict(),
        "model_config": dict(DEFAULT_CVRPTW_MEANONLY_CONFIG),
        "sim_config": asdict(_cfg(0.3)),
        "tardiness_weight": 2.0,
    }, path)
    loaded = load_stoch_policy(str(path))
    assert loaded.variant == "M0"
    assert not loaded.policy.training
    for key, tensor in m0.state_dict().items():
        assert torch.equal(tensor, loaded.policy.state_dict()[key]), key