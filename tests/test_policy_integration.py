"""End-to-end integration test: encoder + decoder + env + policy.

Runs a full rollout on a small random batch and checks that every tour is valid
(all customers served), costs are finite and positive, greedy is deterministic,
POMO force-first is honoured, and gradients flow from the loss back into both
the encoder and the decoder.
"""

from __future__ import annotations

import torch

from src.envs.cvrp_env import CVRPEnv
from src.models.egat_encoder import EGATEncoder
from src.models.policy import CVRPPolicy
from src.models.pointer_decoder import PointerDecoder

_HID = 32


def _make(b: int = 3, n: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    coords = torch.rand(b, n, 2)
    demand = torch.rand(b, n) * 0.2 + 0.1   # customers in [0.1, 0.3]
    demand[:, 0] = 0.0                       # depot
    env = CVRPEnv(coords, demand, capacity=1.0)
    encoder = EGATEncoder(
        input_node_dim=3, input_edge_dim=1, hidden_dim=_HID, hidden_edge_dim=8,
        n_layers=2, n_heads=4,
    )
    decoder = PointerDecoder(hidden_dim=_HID, n_heads=4)
    return env, CVRPPolicy(encoder, decoder), n


def _all_customers_served(actions: torch.Tensor, n: int) -> bool:
    for row in actions:
        served = set(row.tolist()) - {0}
        if served != set(range(1, n)):
            return False
    return True


def test_greedy_rollout_produces_valid_tours() -> None:
    env, policy, n = _make()
    policy.eval()
    actions, sum_logp, cost = policy(env, decode_type="greedy")
    assert _all_customers_served(actions, n)
    assert torch.isfinite(cost).all() and (cost > 0).all()
    assert sum_logp.shape == (env.batch_size,)


def test_greedy_is_deterministic() -> None:
    env, policy, _ = _make()
    policy.eval()
    a1, _, c1 = policy(env, decode_type="greedy")
    a2, _, c2 = policy(env, decode_type="greedy")
    assert torch.equal(a1, a2)
    assert torch.allclose(c1, c2)


def test_force_first_is_honoured() -> None:
    env, policy, n = _make()
    policy.eval()
    forced = torch.tensor([1, 2, 3])
    actions, _, _ = policy(env, decode_type="greedy", force_first=forced)
    assert torch.equal(actions[:, 0], forced)  # first visit is the forced customer


def test_gradients_flow_to_encoder_and_decoder() -> None:
    env, policy, _ = _make()
    policy.train()
    _, sum_logp, cost = policy(env, decode_type="sampling")
    # REINFORCE-style surrogate: cost is a constant, sum_logp carries grad.
    loss = (cost.detach() * sum_logp).mean()
    loss.backward()

    enc_grads = [p.grad for p in policy.encoder.parameters() if p.requires_grad]
    dec_grads = [p.grad for p in policy.decoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in enc_grads)
    assert any(g is not None and g.abs().sum() > 0 for g in dec_grads)


def test_sampling_varies_across_calls() -> None:
    env, policy, _ = _make()
    policy.train()
    torch.manual_seed(1)
    a1, _, _ = policy(env, decode_type="sampling")
    env.reset()
    torch.manual_seed(2)
    a2, _, _ = policy(env, decode_type="sampling")
    # Different seeds should (very likely) give different sampled tours.
    assert not torch.equal(a1, a2)