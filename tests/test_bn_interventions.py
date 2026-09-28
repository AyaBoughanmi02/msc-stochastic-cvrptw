"""Tests for src/eval/bn_interventions.py (Stage-5 mechanism diagnostics).

What must hold for the diagnostic to mean anything:
  1. batch_stat_edge_norm reproduces TRAIN-MODE BatchNorm on the calibration batch.
  2. Both interventions restore the model bit-exactly, including after an exception.
  3. scaled_decoder_variance touches the sigma^2 column only; factor 1.0 is a no-op on a
     full rollout.
  4. A mean-only (D') network's drift cannot depend on beta_var -- the placebo is exact.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import MU, SIGMA2, SimulatorConfig
from src.eval.bn_interventions import (
    batch_stat_edge_norm,
    edge_norm_stats,
    normalisation_drift,
    pre_norm_edge_features,
    scaled_decoder_variance,
)
from src.models.factory import build_cvrptw_gateonly_policy, build_cvrptw_stoch_policy
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env

N_CUSTOMERS = 8
BATCH = 4


def _cfg(beta: float) -> SimulatorConfig:
    return SimulatorConfig(beta_var=beta, eta=0.6, omega=1.0)


@pytest.fixture
def data():
    return generate_cvrptw(BATCH, N_CUSTOMERS, 20, seed=0)


def _policy_b():
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    # Non-trivial running statistics, as a trained model would have.
    with torch.no_grad():
        policy.encoder.bn_edge.running_mean.normal_()
        policy.encoder.bn_edge.running_var.uniform_(0.5, 2.0)
        policy.encoder.bn_edge.weight.uniform_(0.5, 1.5)
        policy.decoder.w_edge.weight.normal_()        # zero at init would hide scaling
    return policy.eval()


def test_batch_stats_reproduce_train_mode_bn(data) -> None:
    policy = _policy_b()
    bn = policy.encoder.bn_edge
    cfg = _cfg(0.3)
    x = pre_norm_edge_features(policy, data, cfg)                       # (E, C)
    reference = F.batch_norm(x, None, None, bn.weight, bn.bias, training=True,
                             momentum=0.0, eps=bn.eps)                  # (E, C)
    with torch.no_grad(), batch_stat_edge_norm(policy, edge_norm_stats(policy, data, cfg)):
        got = bn(x)
    torch.testing.assert_close(got, reference, atol=1e-5, rtol=1e-5)


def test_batch_stat_edge_norm_restores_exactly_even_on_error(data) -> None:
    policy = _policy_b()
    bn = policy.encoder.bn_edge
    before = (bn.running_mean.clone(), bn.running_var.clone())
    with pytest.raises(RuntimeError, match="boom"):
        with batch_stat_edge_norm(policy, edge_norm_stats(policy, data, _cfg(0.9))):
            assert not torch.equal(bn.running_mean, before[0])
            raise RuntimeError("boom")
    assert torch.equal(bn.running_mean, before[0])
    assert torch.equal(bn.running_var, before[1])


def test_batch_stat_edge_norm_refuses_train_mode(data) -> None:
    policy = _policy_b()
    stats = edge_norm_stats(policy, data, _cfg(0.3))
    policy.train()
    with pytest.raises(RuntimeError, match="train mode"):
        with batch_stat_edge_norm(policy, stats):
            pass


def test_scaled_decoder_variance_touches_sigma2_only(data) -> None:
    policy = _policy_b()
    env = build_stoch_env(data, _cfg(0.3), N_CUSTOMERS, 2.0, reward_mode="analytic")
    env.reset()
    ref = policy._decoder_edge_feat(env)                                # (B*S, N, 2)
    with scaled_decoder_variance(policy, 2.5):
        got = policy._decoder_edge_feat(env)
    assert torch.equal(got[..., MU], ref[..., MU])
    assert torch.equal(got[..., SIGMA2], ref[..., SIGMA2] * 2.5)
    assert torch.equal(policy._decoder_edge_feat(env), ref)            # restored


def test_scaled_decoder_variance_factor_one_is_identity_on_rollout(data) -> None:
    policy = _policy_b()
    starts = build_starts(BATCH, N_CUSTOMERS, data.coords.device)

    def rollout():
        env = build_stoch_env(data, _cfg(0.6), N_CUSTOMERS, 2.0, reward_mode="analytic")
        with torch.no_grad():
            actions, _, cost = policy(env, "greedy", starts)
        return actions, cost

    a0, c0 = rollout()
    with scaled_decoder_variance(policy, 1.0):
        a1, c1 = rollout()
    assert torch.equal(a0, a1) and torch.equal(c0, c1)


def test_scaled_decoder_variance_refuses_mean_only_decoder() -> None:
    policy = build_cvrptw_gateonly_policy().eval()
    with pytest.raises(ValueError, match="no sigma"):
        with scaled_decoder_variance(policy, 2.0):
            pass


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
def test_scaled_decoder_variance_rejects_bad_factor(factor: float) -> None:
    with pytest.raises(ValueError):
        with scaled_decoder_variance(_policy_b(), factor):
            pass


def test_mean_only_drift_is_exactly_beta_invariant(data) -> None:
    """The placebo: M's edge feature is mu_bar, which beta_var never touches."""
    torch.manual_seed(0)
    policy = build_cvrptw_gateonly_policy().eval()
    assert normalisation_drift(policy, data, _cfg(0.15)) == \
        normalisation_drift(policy, data, _cfg(0.9))


def test_variant_b_drift_depends_on_beta(data) -> None:
    policy = _policy_b()
    assert normalisation_drift(policy, data, _cfg(0.15))["drift"] != \
        normalisation_drift(policy, data, _cfg(0.9))["drift"]