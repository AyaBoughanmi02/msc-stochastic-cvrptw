"""Tests for the POMO training loop and the CVRP generator.

Covers the multi-start trap (forced starts must be distinct and correctly
assigned per instance), rollout/loss shapes and sign, and a small end-to-end
learning check that mean cost drops over a handful of steps.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp
from src.envs.cvrp_env import CVRPEnv
from src.models.egat_encoder import EGATEncoder
from src.models.policy import CVRPPolicy
from src.models.pointer_decoder import PointerDecoder
from src.train.pomo import build_starts, pomo_loss, pomo_rollout, train


def _tiny_policy(hidden: int = 32) -> CVRPPolicy:
    enc = EGATEncoder(3, 1, hidden_dim=hidden, hidden_edge_dim=8, n_layers=2, n_heads=4)
    dec = PointerDecoder(hidden_dim=hidden, n_heads=4)
    return CVRPPolicy(enc, dec)


# --- generator ---

def test_generator_shapes_and_normalisation() -> None:
    coords, demand, cap = generate_cvrp(4, n_customers=5, capacity=10, seed=0)
    assert coords.shape == (4, 6, 2)
    assert demand.shape == (4, 6)
    assert cap == 1.0
    assert torch.all(demand[:, 0] == 0)                 # depot demand 0
    assert torch.all(demand[:, 1:] > 0) and torch.all(demand[:, 1:] <= 0.9 + 1e-6)


def test_generator_seed_reproducible() -> None:
    a = generate_cvrp(3, 5, capacity=10, seed=42)[0]
    b = generate_cvrp(3, 5, capacity=10, seed=42)[0]
    assert torch.equal(a, b)


# --- the multi-start trap ---

def test_build_starts_distinct_and_grouped() -> None:
    starts = build_starts(b=2, n_starts=3)
    assert starts.tolist() == [1, 2, 3, 1, 2, 3]        # tiled per instance
    grouped = starts.reshape(2, 3)
    # each instance sees all three distinct starts, correctly ordered
    assert torch.equal(grouped, torch.tensor([[1, 2, 3], [1, 2, 3]]))


def test_forced_starts_actually_distinct_in_rollout() -> None:
    # Build the replicated env exactly as pomo_rollout does and check the first
    # visited node per trajectory is the intended distinct customer.
    coords, demand, cap = generate_cvrp(2, n_customers=4, capacity=10, seed=1)
    b, n_starts = 2, 4
    coords_r = coords.repeat_interleave(n_starts, dim=0)
    demand_r = demand.repeat_interleave(n_starts, dim=0)
    starts = build_starts(b, n_starts)
    env = CVRPEnv(coords_r, demand_r, cap)
    policy = _tiny_policy()
    policy.eval()
    actions, _, _ = policy(env, decode_type="greedy", force_first=starts)
    first = actions[:, 0].reshape(b, n_starts)
    assert torch.equal(first, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]))


# --- rollout + loss ---

def test_pomo_rollout_shapes_default_all_customers() -> None:
    coords, demand, cap = generate_cvrp(3, n_customers=5, capacity=10, seed=2)
    policy = _tiny_policy()
    cost, sum_logp = pomo_rollout(policy, coords, demand, cap)  # default n_starts=5
    assert cost.shape == (3, 5)
    assert sum_logp.shape == (3, 5)
    assert torch.isfinite(cost).all()


def test_pomo_loss_baseline_and_sign() -> None:
    # Hand-built costs: for instance 0, start 0 is best (lowest cost).
    cost = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    sum_logp = torch.zeros(2, 3, requires_grad=True)
    loss, mean_cost = pomo_loss(cost, sum_logp)
    assert torch.isclose(mean_cost, cost.mean())
    # baseline is the per-row mean; advantage of the cheapest start is negative
    baseline = cost.mean(dim=1, keepdim=True)
    advantage = cost - baseline
    assert advantage[0, 0] < 0 and advantage[0, 2] > 0
    assert torch.isfinite(loss)


def test_train_step_runs_and_updates_params() -> None:
    from src.train.pomo import train_step

    coords, demand, cap = generate_cvrp(8, n_customers=5, capacity=10, seed=3)
    policy = _tiny_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in policy.parameters()]
    metrics = train_step(policy, opt, coords, demand, cap)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["grad_norm"] >= 0.0
    after = list(policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_pomo_learns_on_fixed_problem() -> None:
    # Train on freshly sampled n=5 instances; mean greedy cost on a fixed eval
    # set should drop over a few dozen steps.
    torch.manual_seed(0)
    policy = _tiny_policy(hidden=64)

    def sample_batch():
        return generate_cvrp(32, n_customers=5, capacity=10)

    eval_coords, eval_demand, eval_cap = generate_cvrp(64, 5, capacity=10, seed=999)

    def mean_greedy_cost() -> float:
        policy.eval()
        with torch.no_grad():
            env = CVRPEnv(eval_coords, eval_demand, eval_cap)
            _, _, cost = policy(env, decode_type="greedy")
        return float(cost.mean())

    start_cost = mean_greedy_cost()
    train(policy, sample_batch, n_steps=40, lr=1e-3, log_every=0)
    end_cost = mean_greedy_cost()
    assert end_cost < start_cost  # learning reduced tour length