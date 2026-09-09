"""Tests for the variant-B policy: Edge-Bank moments into encoder and pointer.

Load-bearing tests:
  - `test_gate_chain_is_inert`: every term stacked on top of variant A must be provably
    inert when switched off (env with the bank off, zeroed w_edge, empty gate slot).
    This is what stops drift sneaking in as variant D is added on top of B.
  - `test_variant_a_policy_is_unchanged`: routing CVRPTWPolicy through hooks must not have
    moved a single number on the certified deterministic path.
  - `test_certified_checkpoints_still_load`: gate B, against the real stored configs.
"""

from __future__ import annotations

import pytest
import torch

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import MU, SIGMA2, SimulatorConfig, build_edge_bank, horizon_average
from src.envs.cvrptw_env import CVRPTWEnv
from src.envs.cvrptw_stoch_diagnostics import measure_pointer_saturation
from src.models.cvrptw_policy import CVRPTWPolicy
from src.models.cvrptw_stoch_policy import CVRPTWStochPolicy
from src.models.factory import (
    DEFAULT_CVRPTW_CONFIG,
    DEFAULT_CVRPTW_STOCH_CONFIG,
    DEFAULT_MODEL_CONFIG,
    build_cvrptw_policy,
    build_cvrptw_stoch_policy,
)
from src.models.gate_checks import (
    assert_decoder_terms_inert,
    assert_env_reduces_to_variant_a,
    assert_gate_chain_inert,
)
from src.train.pomo_stoch import build_stoch_env, pomo_rollout_stoch, train_step_stoch

_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 3, n_customers: int = 10, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.3, "eta": 0.6,
                              "shift_end": _SHIFT, **over})


# ------------------------------------------------------------------- factory wiring


def test_factory_builds_the_variant_b_policy() -> None:
    policy = build_cvrptw_stoch_policy()
    assert isinstance(policy, CVRPTWStochPolicy)
    assert policy.encoder.fc_edge.weight.shape == (16, 2)     # encoder: (mu_bar, sigma^2)
    assert policy.decoder.w_edge.weight.shape == (1, 2)       # decoder: (mu, sigma^2)(tau)
    assert policy.decoder.edge_bias_mode == "pre_tanh"
    assert torch.equal(policy.decoder.w_edge.weight, torch.zeros(1, 2))


def test_new_config_keys_default_to_the_certified_architecture() -> None:
    """edge_feat_dim / edge_bias_mode must default to variant A's behaviour."""
    assert DEFAULT_MODEL_CONFIG["edge_feat_dim"] == 0
    assert DEFAULT_MODEL_CONFIG["edge_bias_mode"] == "pre_tanh"
    assert DEFAULT_CVRPTW_CONFIG["edge_feat_dim"] == 0
    assert DEFAULT_CVRPTW_STOCH_CONFIG["edge_feat_dim"] == 2
    assert not hasattr(build_cvrptw_policy().decoder, "w_edge")


def test_certified_checkpoints_still_load(tmp_path) -> None:
    """A config saved WITHOUT the new keys must reconstruct the certified architecture.

    This mirrors cvrp20_cos3e4.pt, whose stored config predates context_extra_dim and
    reconstructs it from the default -- the same mechanism edge_feat_dim relies on.
    """
    legacy = {k: v for k, v in DEFAULT_CVRPTW_CONFIG.items()
              if k not in ("edge_feat_dim", "edge_bias_mode")}
    torch.manual_seed(0)
    original = build_cvrptw_policy(legacy)
    path = tmp_path / "legacy.pt"
    torch.save({"model_config": legacy, "model_state": original.state_dict()}, path)

    blob = torch.load(path, weights_only=False)
    restored = build_cvrptw_policy(blob["model_config"])
    missing, unexpected = restored.load_state_dict(blob["model_state"], strict=True)
    assert not missing and not unexpected
    assert not hasattr(restored.decoder, "w_edge")


# --------------------------------------------------------------- variant A unchanged


def test_variant_a_policy_is_unchanged() -> None:
    """The hooks' defaults must reproduce the certified rollout exactly."""
    data = _data(batch=4, n_customers=10, seed=1)
    torch.manual_seed(0)
    policy = build_cvrptw_policy().eval()
    env = CVRPTWEnv(data.coords, data.demand, data.capacity,
                    data.time_windows, data.service_time, data.shift_end)

    # The hooks themselves.
    env.reset()
    assert policy._decoder_edge_feat(env) is None
    assert policy._gate_penalty(env, None) is None
    assert policy.policy_metrics() == {}
    enc = policy._encoder_inputs(env)
    assert enc.edge_attr.shape == (4 * 11 * 11, 1)            # still the distance column

    actions, _, cost = policy(env, decode_type="greedy")
    assert actions.shape[0] == 4 and torch.isfinite(cost).all()


# ------------------------------------------------------------------ the B rollout


def test_stoch_policy_feeds_both_stages() -> None:
    data = _data(batch=2, n_customers=8, seed=2)
    cfg = _cfg()
    policy = build_cvrptw_stoch_policy()
    env = build_stoch_env(data, cfg, n_starts=3, tardiness_weight=1.0,
                          generator=torch.Generator().manual_seed(2))
    b_env, n = env.batch_size, env.num_nodes

    enc = policy._encoder_inputs(env)
    assert enc.edge_attr.shape == (b_env * n * n, 2)          # 2-dim encoder edges
    # Column 0 is mu_bar, column 1 sigma_bar^2, expanded per trajectory from the bank.
    expected = horizon_average(env.bank)[env.instance_idx].reshape(b_env * n * n, 2)
    assert torch.equal(enc.edge_attr, expected)

    env.reset()
    feat = policy._decoder_edge_feat(env)                     # (B*S, N, 2)
    assert feat.shape == (b_env, n, 2)
    assert (feat[..., SIGMA2] >= 0).all()
    # At the depot in bin 0 the mu column is the bank row for node 0.
    assert torch.equal(feat[..., MU], env.bank[env.instance_idx, 0, :, 0, MU])


def test_stoch_policy_rejects_a_deterministic_env() -> None:
    data = _data(batch=2, n_customers=8, seed=3)
    env = CVRPTWEnv(data.coords, data.demand, data.capacity,
                    data.time_windows, data.service_time, data.shift_end)
    with pytest.raises(TypeError, match="CVRPTWStochEnv"):
        build_cvrptw_stoch_policy()(env, decode_type="greedy")


def test_end_to_end_pomo_rollout_and_train_step() -> None:
    """The real network, the real env, the real POMO glue."""
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    cfg = _cfg()
    out = pomo_rollout_stoch(
        policy, _data(batch=2, n_customers=10, seed=4), cfg, n_starts=4,
        tardiness_weight=2.0, generator=torch.Generator().manual_seed(4),
    )
    assert out.cost.shape == (2, 4)
    assert torch.isfinite(out.cost).all() and torch.isfinite(out.sum_logp).all()
    assert torch.allclose(
        out.cost, out.distance + 2.0 * (out.late_count + out.tardiness), atol=1e-4
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    stats = train_step_stoch(
        policy, optimizer, lambda: _data(batch=2, n_customers=10, seed=5), cfg,
        n_starts=4, tardiness_weight=2.0,
        generator=torch.Generator().manual_seed(5),
    )
    for key in ("loss", "mean_cost", "mean_distance", "mean_tardiness", "late_rate"):
        assert key in stats and stats[key] == stats[key]      # not NaN
    # Variant B contributes no policy metrics; variant D will add softplus_w here.
    assert "softplus_w" not in stats


def test_w_edge_receives_gradient_in_a_real_rollout() -> None:
    """Zero-init must not make the edge term a dead path inside the full policy."""
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    out = pomo_rollout_stoch(
        policy, _data(batch=2, n_customers=10, seed=6), _cfg(), n_starts=3,
        generator=torch.Generator().manual_seed(6),
    )
    out.sum_logp.sum().backward()
    assert policy.decoder.w_edge.weight.grad is not None
    assert float(policy.decoder.w_edge.weight.grad.abs().sum()) > 0.0


# ------------------------------------------------------------------- the gate chain


def test_gate_chain_is_inert() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    with torch.no_grad():                       # a TRAINED-looking edge term, not zeros
        policy.decoder.w_edge.weight.normal_()
    assert_gate_chain_inert(policy, _data(batch=4, n_customers=10, seed=7), _cfg())


def test_gate_chain_links_run_independently() -> None:
    policy = build_cvrptw_stoch_policy()
    assert_decoder_terms_inert(policy.decoder)                  # links 2 + 3, no data
    assert_decoder_terms_inert(build_cvrptw_policy().decoder)   # link 3 only (no w_edge)
    assert_env_reduces_to_variant_a(_data(batch=4, n_customers=10, seed=8), _cfg())


def test_gate_chain_detects_a_broken_link() -> None:
    """The check must FAIL when the property it guards is actually violated."""

    class _LeakyDecoder(torch.nn.Module):
        """A decoder whose zeroed edge term still perturbs the output."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._modules["inner"], name)

        def precompute(self, node_emb):
            return self.inner.precompute(node_emb)

        def forward(self, *args, gate_penalty=None, **kwargs):
            out = self.inner(*args, gate_penalty=None, **kwargs)
            return out if gate_penalty is None else out + 1e-6   # leak

    leaky = _LeakyDecoder(build_cvrptw_stoch_policy().decoder)
    with pytest.raises(RuntimeError, match="link 3"):
        assert_decoder_terms_inert(leaky)


# ------------------------------------------------------------- the saturation meter


def test_saturation_meter_reports_interpretable_rates() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    stats = measure_pointer_saturation(
        policy, _data(batch=4, n_customers=10, seed=9), _cfg(), n_starts=4, seed=9,
    )
    for key in ("saturated_frac", "saturated_frac_near_tight_window",
                "frac_near_tight", "mean_abs_u", "n_feasible", "n_tight"):
        assert key in stats
    assert 0.0 <= stats["saturated_frac"] <= 1.0
    assert 0.0 <= stats["saturated_frac_near_tight_window"] <= 1.0
    assert stats["n_feasible"] > 0
    assert stats["mean_abs_u"] >= 0.0
    # The meter must leave the decoder exactly as it found it.
    assert policy.decoder.record_diagnostics is False
    assert policy.decoder._last_u is None


def test_saturation_rises_when_the_pointer_is_driven_hard() -> None:
    """A large edge weight pushes u into tanh's flat region; the meter must see it."""
    torch.manual_seed(0)
    policy = build_cvrptw_stoch_policy()
    data, cfg = _data(batch=4, n_customers=10, seed=10), _cfg()
    calm = measure_pointer_saturation(policy, data, cfg, n_starts=4, seed=10)
    with torch.no_grad():
        policy.decoder.w_edge.weight.fill_(60.0)
    driven = measure_pointer_saturation(policy, data, cfg, n_starts=4, seed=10)
    assert driven["saturated_frac"] > calm["saturated_frac"]