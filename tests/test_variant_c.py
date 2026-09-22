"""Tests for variant C: the SED2AM-style learned gate (the published competitor).

Load-bearing tests:

  - `test_variant_c_reduces_to_variant_b_with_its_gates_off`: gate-chain link 5. C must be
    B plus gates, so any B-vs-C difference is the gate and nothing else.
  - `test_encoder_gate_is_post_softmax_and_not_renormalised`: this is THE mechanistic
    difference from variant D. If a refactor ever renormalised the weights, C would
    silently become a pre-softmax bias and the comparison would lose its meaning.
  - `test_sed_gate_parameters_land_in_their_own_optimizer_group`: C's gate gets the same
    dedicated learning rate as D's scalar. Without it C is handicapped exactly where D is
    protected, and "D beats C" would be an optimizer artefact.
  - `test_variants_b_and_d_are_unaffected`: adding C must not move one weight of B or D.
"""

from __future__ import annotations

import pytest
import torch

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SIGMA2, SimulatorConfig
from src.models.cvrptw_sed2am_policy import (
    DEFAULT_SED_GATE_BIAS_INIT,
    CVRPTWSED2AMPolicy,
)
from src.models.egat_encoder import EGATLayer
from src.models.factory import (
    build_cvrptw_gated_policy,
    build_cvrptw_sed2am_policy,
    build_cvrptw_stoch_policy,
)
from src.models.gate_checks import assert_gate_chain_inert, assert_sed_gate_inert
from src.train.pomo import build_starts
from src.train.pomo_stoch import (
    SED_GATE_TAG,
    build_optimizer,
    build_stoch_env,
    train_step_stoch,
)

_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 3, n_customers: int = 8, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.3, "eta": 0.6,
                              "shift_end": _SHIFT, **over})


def _paired_policies() -> tuple[CVRPTWSED2AMPolicy, torch.nn.Module]:
    """Variant C and a variant B holding C's shared weights (gates are C's only extra)."""
    torch.manual_seed(0)
    c = build_cvrptw_sed2am_policy()
    b = build_cvrptw_stoch_policy()
    shared = {k: v.detach().clone() for k, v in c.state_dict().items()
              if SED_GATE_TAG not in k}
    b.load_state_dict(shared, strict=True)
    return c.eval(), b.eval()


# ------------------------------------------------------------------ what C adds to B


def test_variant_c_is_variant_b_plus_gate_parameters_only() -> None:
    torch.manual_seed(0)
    c = build_cvrptw_sed2am_policy()
    torch.manual_seed(0)
    b = build_cvrptw_stoch_policy()

    extra = set(c.state_dict()) - set(b.state_dict())
    assert extra and all(SED_GATE_TAG in k for k in extra)
    # 4 layers x (8 heads x 16 head_dim + 8 biases) + Linear(2 -> 1) with bias = 547.
    assert sum(c.state_dict()[k].numel() for k in extra) == 547
    # Same seed => the SHARED weights are initialised identically to B's, so no
    # initialisation confound enters the B/C comparison.
    cs, bs = c.state_dict(), b.state_dict()
    assert all(torch.equal(cs[k], bs[k]) for k in bs)


def test_gate_starts_near_open_so_c_starts_as_b() -> None:
    policy = build_cvrptw_sed2am_policy()
    expected = float(torch.sigmoid(torch.tensor(DEFAULT_SED_GATE_BIAS_INIT)))
    assert policy.policy_metrics()["enc_gate_mean"] == pytest.approx(expected, abs=1e-6)
    # Edge-independent at init (zero weights) but responsive: sigmoid'(2) = 0.105.
    assert float(policy.sed_gate_ptr.weight.detach().abs().sum()) == 0.0
    assert policy.policy_metrics()["ptr_b"] == pytest.approx(DEFAULT_SED_GATE_BIAS_INIT)


# -------------------------------------------------------- link 5: gates off => B


def test_variant_c_reduces_to_variant_b_with_its_gates_off() -> None:
    c, b = _paired_policies()
    data, cfg = _data(seed=1), _cfg()
    starts = 3

    def rollout(policy) -> tuple[torch.Tensor, torch.Tensor]:
        env = build_stoch_env(data, cfg, starts, 2.0, reward_mode="analytic")
        actions, _, cost = policy(
            env, "greedy", build_starts(data.coords.size(0), starts, data.coords.device)
        )
        return actions, cost

    c.set_sed_gates(False)
    a_off, cost_off = rollout(c)
    a_b, cost_b = rollout(b)
    assert torch.equal(a_off, a_b)
    assert torch.equal(cost_off, cost_b)          # bitwise on CPU

    c.set_sed_gates(True)
    a_on, cost_on = rollout(c)
    assert not torch.equal(cost_on, cost_b)       # the gate is not decoration


def test_gate_chain_link_five_runs_and_restores_the_switch() -> None:
    torch.manual_seed(0)
    c = build_cvrptw_sed2am_policy()
    assert_sed_gate_inert(c)                      # raises if C with gates off drifts
    assert c.sed_gate_active is True              # the switch is restored
    assert all(layer.sed_gate_active for layer in c.encoder.layers)
    assert_gate_chain_inert(c, _data(seed=2), _cfg())   # the whole chain accepts C


def test_link_five_catches_a_real_leak() -> None:
    """A gate that ignores the off switch must fail the check, or it is a rubber stamp."""
    torch.manual_seed(0)
    c = build_cvrptw_sed2am_policy()
    with torch.no_grad():                      # make the gate visibly non-neutral
        c.encoder.layers[0].sed_gate_b.fill_(-2.0)

    class LeakyLayer(EGATLayer):
        pass

    c.encoder.layers[0].__class__ = LeakyLayer
    original = LeakyLayer.message

    def leaky(self, *args, **kwargs):           # ignores sed_gate_active
        self.sed_gate_active = True
        return original(self, *args, **kwargs)

    LeakyLayer.message = leaky
    with pytest.raises(RuntimeError, match="link 5"):
        assert_sed_gate_inert(c)


# --------------------------------------------------- the mechanism: post-softmax gate


def test_encoder_gate_is_post_softmax_and_not_renormalised() -> None:
    """SED2AM Eq. 11: a_ij = softmax(u_ij) * sigmoid(g_ij), weights no longer sum to 1.

    This is the contrast with variant D, whose pre-softmax penalty keeps a distribution.
    """
    torch.manual_seed(0)
    n, hidden, heads, edge_dim = 5, 16, 2, 4
    layer = EGATLayer(hidden, edge_dim, heads, sigmoid_gate=True).eval()
    idx = torch.arange(n)
    edge_index = torch.stack([idx.repeat_interleave(n), idx.repeat(n)], dim=0)
    x = torch.rand(n, hidden)
    edge_attr = torch.rand(n * n, edge_dim)

    # edge_index row 0 = source, row 1 = target, so view(source, target, head) and the
    # softmax normalises over the SOURCE axis for each target.
    _, (_, alpha) = layer(x, edge_index, edge_attr, return_attention_weights=True)
    per_target = alpha.view(n, n, heads).sum(dim=0)              # (N, H) sum over sources
    gate = float(torch.sigmoid(torch.tensor(DEFAULT_SED_GATE_BIAS_INIT)))
    assert torch.allclose(per_target, torch.full_like(per_target, gate), atol=1e-5)

    layer.sed_gate_active = False
    _, (_, plain) = layer(x, edge_index, edge_attr, return_attention_weights=True)
    assert torch.allclose(plain.view(n, n, heads).sum(dim=0),
                          torch.ones(n, heads), atol=1e-5)       # a proper distribution


def test_pointer_penalty_is_minus_log_sigmoid_of_the_learned_logit() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_sed2am_policy()
    data, cfg = _data(batch=2, n_customers=6, seed=3), _cfg()
    env = build_stoch_env(data, cfg, 2, 2.0, reward_mode="analytic")
    env.reset()
    feat = policy._decoder_edge_feat(env)                        # (B*S, N, 2)

    penalty = policy._gate_penalty(env, feat)
    logit = policy.sed_gate_ptr(feat).squeeze(-1)
    assert torch.allclose(penalty, -torch.log(torch.sigmoid(logit)), atol=1e-6)
    assert bool((penalty >= 0).all())                            # a gate can only damp

    # An unconstrained gate CAN learn the monotone behaviour variant D assumes -- or its
    # opposite. With a negative weight on sigma^2, more variance means more penalty.
    with torch.no_grad():
        policy.sed_gate_ptr.weight.copy_(torch.tensor([[0.0, -1.0]]))
    penalty = policy._gate_penalty(env, feat)
    order = feat[..., SIGMA2].flatten().argsort()
    assert bool((penalty.flatten()[order].diff() >= -1e-6).all())

    policy.set_sed_gates(False)
    assert policy._gate_penalty(env, feat) is None


# ------------------------------------------------------------------ fair training


def test_sed_gate_parameters_land_in_their_own_optimizer_group() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_sed2am_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    groups = {g["name"]: g for g in opt.param_groups}
    assert set(groups) == {"base", "gate"}
    assert groups["gate"]["lr"] == 1e-2 and groups["gate"]["weight_decay"] == 0.0

    gate_ids = {id(p) for p in groups["gate"]["params"]}
    expected = {id(p) for n, p in policy.named_parameters() if SED_GATE_TAG in n}
    assert gate_ids == expected                                   # by identity
    base_ids = {id(p) for p in groups["base"]["params"]}
    assert not (gate_ids & base_ids)
    assert len(gate_ids) + len(base_ids) == len(list(policy.parameters()))


def test_the_sed_gates_actually_move_under_training() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_sed2am_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()
              if SED_GATE_TAG in n}
    for _ in range(3):
        train_step_stoch(
            policy, opt, lambda: _data(batch=2, n_customers=6, seed=4), _cfg(),
            n_starts=3, tardiness_weight=2.0, reward_mode="analytic",
        )
    moved = {n: float((p.detach() - before[n]).abs().max())
             for n, p in policy.named_parameters() if SED_GATE_TAG in n}
    # Every gate tensor receives gradient from the rollout (is it in the base group?).
    assert all(v > 1e-4 for v in moved.values()), moved


def test_policy_metrics_report_the_learned_gate() -> None:
    policy = build_cvrptw_sed2am_policy()
    metrics = policy.policy_metrics()
    assert set(metrics) == {"ptr_w_mu", "ptr_w_sigma2", "ptr_b", "enc_gate_mean"}


# ------------------------------------------------------- B and D are left untouched


def test_variants_b_and_d_are_unaffected() -> None:
    torch.manual_seed(0)
    b = build_cvrptw_stoch_policy()
    torch.manual_seed(0)
    d = build_cvrptw_gated_policy()
    assert not any(SED_GATE_TAG in k for k in b.state_dict())
    assert not any(SED_GATE_TAG in k for k in d.state_dict())
    assert not any(getattr(l, "sigmoid_gate", False) for l in b.encoder.layers)
    # D is still B plus exactly one scalar.
    assert (sum(p.numel() for p in d.parameters())
            - sum(p.numel() for p in b.parameters()) == 1)
    opt = build_optimizer(b, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    assert [g["name"] for g in opt.param_groups] == ["base"]      # B has no gate group


def test_compare_variants_can_switch_the_gates_off() -> None:
    """The `sed_gates` context manager must restore the switch even on an exception."""
    from scripts.compare_variants import sed_gates

    torch.manual_seed(0)
    c = build_cvrptw_sed2am_policy()
    with sed_gates(c, False):
        assert c.sed_gate_active is False
        assert all(not layer.sed_gate_active for layer in c.encoder.layers)
    assert c.sed_gate_active is True
    assert all(layer.sed_gate_active for layer in c.encoder.layers)

    with pytest.raises(ValueError):
        with sed_gates(c, False):
            raise ValueError("boom")
    assert c.sed_gate_active is True
