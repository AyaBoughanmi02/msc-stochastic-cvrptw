"""Tests for the evaluation module and the model factory."""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp
from src.eval.evaluate import (
    augment_coords_8,
    evaluate,
    multistart_greedy_cost,
    optimality_gap,
)
from src.models.factory import build_policy, model_config_from


def _policy():
    return build_policy(model_config_from(hidden_dim=32, hidden_edge_dim=8,
                                          n_layers=2, n_heads=4))


def test_build_policy_from_config_runs() -> None:
    policy = _policy()
    coords, demand, cap = generate_cvrp(2, n_customers=5, capacity=10, seed=0)
    cost = multistart_greedy_cost(policy, coords, demand, cap)
    assert cost.shape == (2,)
    assert torch.isfinite(cost).all() and (cost > 0).all()


def test_greedy_best_of_n_is_deterministic() -> None:
    policy = _policy()
    coords, demand, cap = generate_cvrp(3, n_customers=5, capacity=10, seed=1)
    a = multistart_greedy_cost(policy, coords, demand, cap)
    b = multistart_greedy_cost(policy, coords, demand, cap)
    assert torch.allclose(a, b)  # greedy => deterministic


def test_augmentation_preserves_distances() -> None:
    # The 8 dihedral maps must not change pairwise distances (same problem).
    coords = torch.rand(2, 6, 2)
    aug = augment_coords_8(coords)                     # (8, 2, 6, 2)
    d0 = torch.cdist(coords, coords)                   # (2, 6, 6)
    for k in range(8):
        dk = torch.cdist(aug[k], aug[k])
        assert torch.allclose(d0, dk, atol=1e-5)


def test_augment_eval_not_worse_than_plain() -> None:
    policy = _policy()
    coords, demand, cap = generate_cvrp(4, n_customers=5, capacity=10, seed=2)
    plain = evaluate(policy, coords, demand, cap, augment=False)
    aug = evaluate(policy, coords, demand, cap, augment=True)
    # More candidates can only help (we take the min).
    assert torch.all(aug <= plain + 1e-5)


def test_optimality_gap_math() -> None:
    cost = torch.tensor([11.0, 20.0])
    ref = torch.tensor([10.0, 20.0])
    gap = optimality_gap(cost, ref)
    assert torch.allclose(gap, torch.tensor([10.0, 0.0]))