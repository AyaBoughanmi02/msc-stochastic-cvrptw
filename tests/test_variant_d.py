"""Tests for variant D: the monotone variance gate with ONE shared scalar.

Load-bearing tests:

  - `test_gate_is_exactly_one_shared_parameter`: the ablation claim is "one scalar's worth
    of inductive bias". Two parameters kept in sync would look identical in every other
    test and quietly falsify that claim, so the parameter COUNT is asserted directly.
  - `test_gate_scalar_lands_in_its_own_optimizer_group`: the entire point of the dedicated
    gate learning rate is invisible to every other test. If a wiring mistake sweeps
    gate_w into the global lr=3e-4 group, the gate can travel only ~0.12 over 800 steps,
    D ties B, and the null result is an optimizer artefact rather than a finding.
  - `test_variant_b_equals_variant_d_with_the_gate_off`: B and D must differ by exactly one
    term, bitwise.
  - `test_encoder_gate_is_inert_when_zero`: gate-chain link 4, guarding the final edit to
    the certified encoder.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import MU, SIGMA2, SimulatorConfig
from src.models.cvrptw_gated_policy import CVRPTWGatedPolicy
from src.models.cvrptw_stoch_policy import CVRPTWStochPolicy
from src.models.egat_encoder import EGATEncoder, EGATLayer
from src.models.factory import (
    DEFAULT_CVRPTW_CONFIG,
    DEFAULT_CVRPTW_GATED_CONFIG,
    DEFAULT_CVRPTW_STOCH_CONFIG,
    build_cvrptw_gated_policy,
    build_cvrptw_policy,
    build_cvrptw_stoch_policy,
)
from src.models.gate_checks import assert_encoder_gate_inert, assert_gate_chain_inert
from src.train.pomo import build_starts
from src.train.pomo_stoch import (
    GATE_PARAM_NAME,
    build_optimizer,
    build_stoch_env,
    pomo_rollout_stoch,
    train_step_stoch,
)

_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 4, n_customers: int = 10, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.3, "eta": 0.6,
                              "shift_end": _SHIFT, **over})


def _fully_connected_edge_index(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    return torch.stack([idx.repeat_interleave(n), idx.repeat(n)], dim=0)   # (2, n*n)


# ------------------------------------------------------------- one parameter, shared


def test_gate_is_exactly_one_shared_parameter() -> None:
    torch.manual_seed(0)
    gated = build_cvrptw_gated_policy()
    ungated = build_cvrptw_stoch_policy()

    gate_names = [n for n, _ in gated.named_parameters() if "gate" in n]
    assert gate_names == [GATE_PARAM_NAME]                 # exactly one, and it is w
    assert gated.gate_w.shape == ()                        # a scalar, not a vector
    # Variant D is variant B plus ONE number. Nothing else about the architecture moves.
    assert (sum(p.numel() for p in gated.parameters())
            - sum(p.numel() for p in ungated.parameters()) == 1)
    assert set(gated.state_dict()) - set(ungated.state_dict()) == {GATE_PARAM_NAME}


def test_gate_init_is_small_but_nonzero() -> None:
    policy = build_cvrptw_gated_policy()
    assert float(policy.gate_w.detach()) == 0.0
    assert float(policy.gate_strength().detach()) == pytest.approx(0.6931, abs=1e-4)
    # On the healthy part of the curve: d softplus/dw = sigmoid(0) = 0.5, no vanishing.
    w = policy.gate_w.detach().clone().requires_grad_(True)
    F.softplus(w).backward()
    assert float(w.grad) == pytest.approx(0.5, abs=1e-6)


def test_both_stages_read_the_same_scalar() -> None:
    """Encoder and decoder gates must scale together -- one source of truth."""
    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    data, cfg = _data(batch=2, n_customers=8, seed=1), _cfg()
    env = build_stoch_env(data, cfg, 3, 1.0, reward_mode="analytic")
    env.reset()
    enc = policy._encoder_inputs(env)
    feat = policy._decoder_edge_feat(env)

    def penalties():
        return (policy._encoder_gate_penalty(env, enc).detach(),
                policy._gate_penalty(env, feat).detach())

    e0, d0 = penalties()
    with torch.no_grad():
        policy.gate_w.fill_(2.0)
    e1, d1 = penalties()
    ratio = float(F.softplus(torch.tensor(2.0)) / F.softplus(torch.tensor(0.0)))
    assert torch.allclose(e1, e0 * ratio, atol=1e-6)
    assert torch.allclose(d1, d0 * ratio, atol=1e-6)


def test_penalties_use_the_specified_variance_inputs() -> None:
    """Same scalar, different sigma^2: horizon-averaged in the encoder, clock-refreshed
    in the decoder -- and the encoder penalty comes from the very tensor it encodes."""
    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    env = build_stoch_env(_data(batch=2, n_customers=8, seed=2), _cfg(), 3, 1.0,
                          reward_mode="analytic")
    env.reset()
    enc = policy._encoder_inputs(env)
    strength = policy.gate_strength().detach()

    enc_pen = policy._encoder_gate_penalty(env, enc).detach()
    assert enc_pen.shape == (enc.edge_index.size(1),)
    assert torch.equal(enc_pen, strength * enc.edge_attr[:, SIGMA2])   # same tensor

    feat = policy._decoder_edge_feat(env)
    dec_pen = policy._gate_penalty(env, feat).detach()
    assert dec_pen.shape == (env.batch_size, env.num_nodes)
    assert torch.equal(dec_pen, strength * feat[..., SIGMA2])
    # The two variance inputs are genuinely different quantities.
    assert not torch.equal(feat[..., MU], feat[..., SIGMA2])


# ------------------------------------------------------- the gate chain (links 3 + 4)


def test_encoder_gate_is_inert_when_zero() -> None:
    """Link 4: gate_penalty=None and gate_penalty=zeros must be BITWISE identical."""
    torch.manual_seed(0)
    encoder = EGATEncoder(input_node_dim=6, input_edge_dim=2).eval()
    assert_encoder_gate_inert(encoder)                      # must not raise

    n = 6
    edge_index = _fully_connected_edge_index(n)
    x, edge_attr = torch.rand(n, 6), torch.rand(n * n, 2)
    with torch.no_grad():
        none_out, _ = encoder(x, edge_index, edge_attr)
        zero_out, _ = encoder(x, edge_index, edge_attr,
                              gate_penalty=torch.zeros(n * n))
        real_out, _ = encoder(x, edge_index, edge_attr,
                              gate_penalty=torch.full((n * n,), 0.7))
    assert torch.equal(none_out, zero_out)
    assert not torch.equal(none_out, real_out)              # a real gate does bite


def test_link_four_still_catches_a_real_leak_under_jitter() -> None:
    """The jitter tolerance must not turn link 4 into a rubber stamp.

    On CUDA the check compares against the backend's own run-to-run noise rather than
    demanding bitwise equality. A gate that genuinely leaked moves the embeddings by
    orders of magnitude more than atomics do, so it must still be caught.
    """

    class _LeakyEncoder(EGATEncoder):
        """Ignores the zero and applies a real gate anyway."""

        def forward(self, x, edge_index, edge_attr, batch=None, gate_penalty=None):
            if gate_penalty is not None:
                gate_penalty = gate_penalty + 0.5      # the leak
            return super().forward(x, edge_index, edge_attr, batch, gate_penalty)

    torch.manual_seed(0)
    leaky = _LeakyEncoder(input_node_dim=6, input_edge_dim=2).eval()
    with pytest.raises(RuntimeError, match="link 4"):
        assert_encoder_gate_inert(leaky)


def test_full_gate_chain_passes_on_a_trained_looking_policy() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    with torch.no_grad():                                   # non-trivial gate and edge term
        policy.gate_w.fill_(1.3)
        policy.decoder.w_edge.weight.normal_()
    assert_gate_chain_inert(policy, _data(batch=4, n_customers=10, seed=3), _cfg())


def test_variant_b_equals_variant_d_with_the_gate_off() -> None:
    """B and D must differ by exactly one term -- bitwise, on a full rollout."""

    class _GateOff(CVRPTWGatedPolicy):
        def _encoder_gate_penalty(self, env, enc):
            return None

        def _gate_penalty(self, env, edge_feat):
            return None

    torch.manual_seed(0)
    gated = build_cvrptw_gated_policy()
    with torch.no_grad():
        gated.gate_w.fill_(1.7)                             # a gate that would matter
    # Share the very same encoder/decoder modules, so ONLY the hooks differ.
    off = _GateOff(gated.encoder, gated.decoder).eval()
    plain = CVRPTWStochPolicy(gated.encoder, gated.decoder).eval()
    gated.eval()

    data, cfg = _data(batch=3, n_customers=10, seed=4), _cfg()

    def roll(policy):
        env = build_stoch_env(data, cfg, 4, 1.0, reward_mode="analytic")
        with torch.no_grad():
            return policy(env, "greedy", build_starts(3, 4, data.coords.device))

    a_act, _, a_cost = roll(off)
    b_act, _, b_cost = roll(plain)
    assert torch.equal(a_act, b_act)
    assert torch.equal(a_cost, b_cost)
    # ... and with the gate ON, D genuinely differs from B.
    g_act, _, _ = roll(gated)
    assert not torch.equal(g_act, b_act)


# --------------------------------------------------------------- the gate's behaviour


def test_gate_shifts_attention_away_from_high_variance_edges() -> None:
    """Monotone by construction: more penalty => strictly less attention, and softmax
    still normalises to 1 (the gate moves where the mass goes, not whether it sums)."""
    torch.manual_seed(0)
    n, heads, hidden, edge_dim = 4, 4, 32, 2
    layer = EGATLayer(hidden_dim=hidden, edge_dim=edge_dim, n_heads=heads).eval()
    edge_index = _fully_connected_edge_index(n)
    h = torch.rand(n, hidden)
    edge_attr = torch.rand(edge_index.size(1), edge_dim)

    def attention(gate: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            _, (_, alpha) = layer(h, edge_index, edge_attr,
                                  return_attention_weights=True, gate_penalty=gate)
        return alpha                                        # (E, H)

    penalty = torch.zeros(edge_index.size(1))
    penalty[5] = 2.0                                        # one "risky" edge
    base, gated = attention(None), attention(penalty)

    assert bool((gated[5] < base[5]).all())                 # less attention, every head
    target = edge_index[1]
    for alpha in (base, gated):
        summed = torch.zeros(n, heads).index_add_(0, target, alpha)
        assert torch.allclose(summed, torch.ones(n, heads), atol=1e-5)


def test_stronger_gate_means_strictly_less_attention() -> None:
    torch.manual_seed(1)
    n, edge_index = 4, _fully_connected_edge_index(4)
    layer = EGATLayer(hidden_dim=32, edge_dim=2, n_heads=4).eval()
    h, edge_attr = torch.rand(n, 32), torch.rand(edge_index.size(1), 2)
    weights = []
    for scale in (0.0, 1.0, 3.0):
        penalty = torch.zeros(edge_index.size(1))
        penalty[5] = scale
        with torch.no_grad():
            _, (_, alpha) = layer(h, edge_index, edge_attr,
                                  return_attention_weights=True, gate_penalty=penalty)
        weights.append(float(alpha[5].mean()))
    assert weights[0] > weights[1] > weights[2]


# ------------------------------------------------------------------------- guards


def test_channel_attention_refuses_the_gate() -> None:
    """Channel mode has no scalar pre-softmax logit to subtract from."""
    layer = EGATLayer(hidden_dim=32, edge_dim=2, n_heads=4, attn="channel")
    edge_index = _fully_connected_edge_index(4)
    with pytest.raises(ValueError, match="scalar"):
        layer(torch.rand(4, 32), edge_index, torch.rand(edge_index.size(1), 2),
              gate_penalty=torch.zeros(edge_index.size(1)))


def test_encoder_gate_shape_guard_raises() -> None:
    encoder = EGATEncoder(input_node_dim=6, input_edge_dim=2)
    edge_index = _fully_connected_edge_index(5)
    args = (torch.rand(5, 6), edge_index, torch.rand(edge_index.size(1), 2))
    with pytest.raises(ValueError, match="one scalar per edge"):
        encoder(*args, gate_penalty=torch.zeros(edge_index.size(1), 2))
    with pytest.raises(ValueError, match="one scalar per edge"):
        encoder(*args, gate_penalty=torch.zeros(3))


def test_gate_needs_two_dim_edge_features() -> None:
    """A variant-D policy built on 1-dim edges must fail loudly, not silently degrade."""
    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy({"input_edge_dim": 1})
    env = build_stoch_env(_data(batch=2, n_customers=8, seed=5), _cfg(), 2, 1.0,
                          reward_mode="analytic")
    env.reset()
    from src.data.graph_build import build_graph_tw

    enc = build_graph_tw(env.coords, env.demand, env.time_windows, env.service_time)
    with pytest.raises(ValueError, match="2-dim edge features"):
        policy._encoder_gate_penalty(env, enc)


# ------------------------------------------- the optimizer group (protects the design)


def test_gate_scalar_lands_in_its_own_optimizer_group() -> None:
    """The dedicated gate lr is invisible to every other test -- assert the wiring.

    If gate_w were swept into the global lr=3e-4 group it could travel only ~0.12 over
    800 steps, the gate would stay near its initialisation whatever the data said, and
    "D ties B" would be an optimizer artefact rather than a result.
    """
    policy = build_cvrptw_gated_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)

    groups = {g["name"]: g for g in opt.param_groups}
    assert set(groups) == {"base", "gate"}
    assert len(groups["gate"]["params"]) == 1
    assert groups["gate"]["params"][0] is policy.gate_w      # identity, not a copy
    assert groups["gate"]["lr"] == 1e-2
    assert groups["gate"]["weight_decay"] == 0.0             # no w -> 0 attractor
    assert groups["base"]["lr"] == 3e-4
    assert groups["base"]["weight_decay"] == 1e-6
    # And it must NOT also be in the base group.
    assert all(p is not policy.gate_w for p in groups["base"]["params"])
    # Every parameter is accounted for exactly once.
    assert (len(groups["base"]["params"]) + 1
            == sum(1 for _ in policy.parameters()))


def test_gateless_variants_get_no_gate_group() -> None:
    opt = build_optimizer(build_cvrptw_stoch_policy(), lr=3e-4, weight_decay=1e-6,
                          gate_lr=1e-2)
    assert [g["name"] for g in opt.param_groups] == ["base"]


def test_the_gate_actually_moves_under_its_own_learning_rate() -> None:
    """A step must change softplus(w) by far more than the base lr would allow."""
    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    before = float(policy.gate_w.detach())
    train_step_stoch(
        policy, opt, lambda: _data(batch=2, n_customers=10, seed=6), _cfg(),
        n_starts=3, tardiness_weight=2.0, reward_mode="analytic",
    )
    moved = abs(float(policy.gate_w.detach()) - before)
    assert moved > 1e-3, f"gate moved only {moved:.2e} -- is it in the base group?"


# ------------------------------------------------------------------ training path


def test_gradient_reaches_the_gate_from_both_stages() -> None:
    """Encoder-only and decoder-only paths must each feed the shared scalar."""

    class _DecoderOnly(CVRPTWGatedPolicy):
        def _encoder_gate_penalty(self, env, enc):
            return None

    class _EncoderOnly(CVRPTWGatedPolicy):
        def _gate_penalty(self, env, edge_feat):
            return None

    for cls in (_DecoderOnly, _EncoderOnly):
        torch.manual_seed(0)
        base = build_cvrptw_gated_policy()
        policy = cls(base.encoder, base.decoder)
        out = pomo_rollout_stoch(
            policy, _data(batch=2, n_customers=10, seed=7), _cfg(), n_starts=3,
            tardiness_weight=2.0, reward_mode="analytic",
        )
        out.sum_logp.sum().backward()
        assert policy.gate_w.grad is not None, cls.__name__
        assert float(policy.gate_w.grad.abs()) > 0.0, cls.__name__


def test_softplus_w_is_reported_every_step() -> None:
    torch.manual_seed(0)
    policy = build_cvrptw_gated_policy()
    assert policy.policy_metrics()["softplus_w"] == pytest.approx(0.6931, abs=1e-4)
    opt = build_optimizer(policy, lr=3e-4, weight_decay=1e-6, gate_lr=1e-2)
    stats = train_step_stoch(
        policy, opt, lambda: _data(batch=2, n_customers=10, seed=8), _cfg(),
        n_starts=3, tardiness_weight=2.0, reward_mode="analytic",
    )
    assert "softplus_w" in stats and stats["softplus_w"] > 0.0


def test_variant_b_reports_no_gate_metric() -> None:
    assert build_cvrptw_stoch_policy().policy_metrics() == {}


# ------------------------------------------------------------------------- gate B


def test_certified_configs_are_unaffected_by_variant_d(tmp_path) -> None:
    """Adding the gate must not disturb the checkpointed architectures."""
    assert "gate_init" not in DEFAULT_CVRPTW_CONFIG
    assert "gate_init" not in DEFAULT_CVRPTW_STOCH_CONFIG
    assert DEFAULT_CVRPTW_GATED_CONFIG["gate_init"] == 0.0
    differing = {
        k for k in DEFAULT_CVRPTW_STOCH_CONFIG
        if DEFAULT_CVRPTW_STOCH_CONFIG[k] != DEFAULT_CVRPTW_GATED_CONFIG[k]
    }
    assert differing == set()            # D adds a key; it changes none

    torch.manual_seed(0)
    original = build_cvrptw_policy(DEFAULT_CVRPTW_CONFIG)
    path = tmp_path / "a.pt"
    torch.save({"model_config": DEFAULT_CVRPTW_CONFIG,
                "model_state": original.state_dict()}, path)
    blob = torch.load(path, weights_only=False)
    restored = build_cvrptw_policy(blob["model_config"])
    missing, unexpected = restored.load_state_dict(blob["model_state"], strict=True)
    assert not missing and not unexpected
    assert not hasattr(restored, GATE_PARAM_NAME)