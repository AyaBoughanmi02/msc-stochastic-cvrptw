"""Tests for src/models/gate_location.py (WHERE the training-time gate acts).

What must hold for the encoder / decoder split to be interpretable:
  1. the switched-off path really receives no gate, the other one still does;
  2. the gate scalar still gets a gradient from the remaining path;
  3. at lambda = 0 all three locations are bit-identical (evaluation is location-free);
  4. the CONSTANT control is constant over the open candidates at every decode step, so
     in the decoder it can only act through the tanh (temperature), never as a ranking;
  5. a constant shift is inert after the tanh and NOT inert before it;
  6. the checkpoint loader re-applies the location.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.checkpoints import load_stoch_policy
from src.models.factory import (
    DEFAULT_CVRPTW_GATEONLY_CONFIG,
    DEFAULT_CVRPTW_MEANONLY_CONFIG,
    DEFAULT_CVRPTW_RISKBIAS_CONFIG,
    build_cvrptw_gate_control_policy,
    build_cvrptw_gateonly_policy,
    build_cvrptw_meanonly_policy,
    build_cvrptw_riskbias_policy,
)
from src.models.gate_location import GATE_LOCATIONS, gate_location_of, restrict_gate
from src.models.pointer_decoder import PointerDecoder
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env

SIM = SimulatorConfig(alpha=0.3, beta_var=0.6, eta=0.6, omega=1.0)
N_CUST, N_INST = 10, 3


def _const_policy(location: str, seed: int = 0) -> torch.nn.Module:
    """D-prime with the CONSTANT gate (the DpConst control), restricted to `location`."""
    torch.manual_seed(seed)                       # same seed -> same initial weights
    cfg = {**DEFAULT_CVRPTW_GATEONLY_CONFIG, "gate_signal": "const"}
    policy = build_cvrptw_gate_control_policy(cfg)
    restrict_gate(policy, location)
    return policy.eval()


def _env_and_starts():
    data = generate_cvrptw(N_INST, N_CUST, 20, seed=3)
    env = build_stoch_env(data, SIM, N_CUST, 2.0, reward_mode="analytic")
    starts = build_starts(N_INST, N_CUST, data.coords.device)        # (B*S,)
    return env, starts


def _capture_decoder_calls(policy: torch.nn.Module) -> tuple[list, object]:
    """Record (mask, gate_penalty) at every decode step. Returns (records, hook handle)."""
    records: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def hook(module, args, kwargs, output):
        mask = args[4] if len(args) > 4 else kwargs["mask"]
        gate = kwargs.get("gate_penalty")
        records.append((mask.clone(), None if gate is None else gate.detach().clone()))

    handle = policy.decoder.register_forward_hook(hook, with_kwargs=True)
    return records, handle


def test_rejects_unknown_location() -> None:
    with pytest.raises(ValueError):
        _const_policy("nowhere")


def test_rejects_policies_without_a_plain_gate() -> None:
    with pytest.raises(TypeError):
        restrict_gate(build_cvrptw_meanonly_policy(DEFAULT_CVRPTW_MEANONLY_CONFIG), "encoder")
    with pytest.raises(TypeError):
        restrict_gate(build_cvrptw_riskbias_policy(DEFAULT_CVRPTW_RISKBIAS_CONFIG), "encoder")


def test_refuses_to_stack_restrictions() -> None:
    policy = _const_policy("encoder")
    with pytest.raises(RuntimeError):
        restrict_gate(policy, "decoder")


@pytest.mark.parametrize("location", GATE_LOCATIONS)
def test_only_the_chosen_path_is_gated(location: str) -> None:
    policy = _const_policy(location)
    env, starts = _env_and_starts()
    enc_gate = policy._encoder_gate_penalty(env, policy._encoder_inputs(env))
    assert (enc_gate is None) == (location == "decoder")

    env, starts = _env_and_starts()
    records, handle = _capture_decoder_calls(policy)
    with torch.no_grad():
        policy(env, decode_type="greedy", force_first=starts)
    handle.remove()
    assert records, "the decoder was never called"
    dec_gated = [gate is not None for _, gate in records]
    assert all(dec_gated) if location != "encoder" else not any(dec_gated)


@pytest.mark.parametrize("location", GATE_LOCATIONS)
def test_gate_still_learns_from_the_remaining_path(location: str) -> None:
    policy = _const_policy(location).train()
    env, starts = _env_and_starts()
    torch.manual_seed(1)
    _, sum_logp, _ = policy(env, decode_type="sampling", force_first=starts)
    sum_logp.sum().backward()
    grad = policy.gate_w.grad
    assert grad is not None and float(grad.abs()) > 0.0


def test_lambda_zero_makes_the_location_irrelevant() -> None:
    outs = []
    for location in GATE_LOCATIONS:
        policy = _const_policy(location, seed=0)
        policy.gate_strength = lambda: torch.zeros(())       # lambda forced to exactly 0
        env, starts = _env_and_starts()
        with torch.no_grad():
            actions, _, cost = policy(env, decode_type="greedy", force_first=starts)
        outs.append((actions, cost))
    for actions, cost in outs[1:]:
        assert torch.equal(actions, outs[0][0])
        assert torch.equal(cost, outs[0][1])


def test_const_decoder_gate_is_constant_over_open_candidates() -> None:
    policy = _const_policy("decoder")
    env, starts = _env_and_starts()
    records, handle = _capture_decoder_calls(policy)
    with torch.no_grad():
        policy(env, decode_type="greedy", force_first=starts)
    handle.remove()
    checked = 0
    for mask, gate in records:
        open_ = ~mask                                             # (B*S, N)
        rows = open_.sum(dim=1) >= 2
        if not bool(rows.any()):
            continue
        hi = gate.masked_fill(mask, float("-inf")).amax(dim=1)
        lo = gate.masked_fill(mask, float("inf")).amin(dim=1)
        assert torch.allclose(hi[rows], lo[rows], atol=1e-6), "constant gate varies"
        checked += int(rows.sum())
    assert checked > 0


@pytest.mark.parametrize("mode", ["pre_tanh", "post_tanh"])
def test_constant_shift_inert_after_tanh_only(mode: str) -> None:
    torch.manual_seed(0)
    b, n, h = 4, 7, 16
    dec = PointerDecoder(hidden_dim=h, n_heads=2, edge_feat_dim=1, edge_bias_mode=mode)
    cache = dec.precompute(torch.randn(b, n, h) * 3.0)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, 0] = True                                             # one closed node per row
    inputs = (cache, torch.randn(b, h), torch.randn(b, h), torch.rand(b, 1), mask)
    edge_feat = torch.rand(b, n, 1)
    base = dec(*inputs, edge_feat=edge_feat, gate_penalty=None)
    shifted = dec(*inputs, edge_feat=edge_feat, gate_penalty=torch.full((b, n), 0.7))
    open_ = ~mask
    diff = (base[open_] - shifted[open_]).abs().max()
    if mode == "post_tanh":
        assert float(diff) < 1e-5                # softmax ignores a common shift
    else:
        assert float(diff) > 1e-3                # the tanh turns it into a temperature


def test_loader_reapplies_the_location(tmp_path) -> None:
    policy = _const_policy("decoder")
    cfg = {**DEFAULT_CVRPTW_GATEONLY_CONFIG, "gate_signal": "const",
           "gate_location": "decoder"}
    path = tmp_path / "ck.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": cfg,
                "sim_config": asdict(SIM), "tardiness_weight": 2.0}, path)
    loaded = load_stoch_policy(str(path))
    assert gate_location_of(loaded.policy) == "decoder"
    assert loaded.variant == "D'-const[decoder]"
    env, _ = _env_and_starts()
    assert loaded.policy._encoder_gate_penalty(env, loaded.policy._encoder_inputs(env)) is None