"""Tests for the OR-Tools CVRP reference solver.

Uses the unit-square instance from test_cvrp_env.py, whose CVRP optima are
hand-computable, so the reference solver is checked against known-correct numbers
(and against the model's own cost convention), not against itself.

    node 0 = depot (0, 0), customers at (1, 0), (1, 1), (0, 1).
"""

from __future__ import annotations

import math

import torch

from src.eval.reference import _int_distance_matrix, solve_reference_set

_SQUARE = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])  # (1, 4, 2)
# Rounding at scale 1e6 is ~1e-6; GLS hits the optimum on these tiny instances.
_TOL = 1e-4


def test_int_distance_matrix_known_pairs() -> None:
    m = _int_distance_matrix(_SQUARE[0], scale=1_000_000)
    assert m[0][0] == 0                                  # self-distance
    assert m[0][1] == 1_000_000                          # unit edge depot->(1,0)
    assert m[0][2] == round(1_000_000 * math.sqrt(2))    # diagonal depot->(1,1)


def test_single_route_optimum() -> None:
    # demands fit one route (cap 9); optimum = square perimeter = 4.0.
    demands = torch.tensor([[0, 1, 1, 1]])               # (1, 4)
    costs, statuses = solve_reference_set(
        _SQUARE, demands, capacity_int=9, time_limit_s=1
    )
    assert costs.shape == (1,)
    assert statuses == [1]                               # ROUTING_SUCCESS
    assert abs(costs.item() - 4.0) < _TOL


def test_capacity_forces_splits() -> None:
    # any two customers (12) exceed cap 9 -> one per route; optimum = 4 + 2*sqrt(2).
    demands = torch.tensor([[0, 6, 6, 6]])               # (1, 4)
    costs, _ = solve_reference_set(
        _SQUARE, demands, capacity_int=9, time_limit_s=1
    )
    assert abs(costs.item() - (4.0 + 2.0 * math.sqrt(2))) < _TOL


def test_batched_shape_and_values() -> None:
    # two identical instances -> identical reference costs, shape (2,).
    coords = _SQUARE.repeat(2, 1, 1)                      # (2, 4, 2)
    demands = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])  # (2, 4)
    costs, statuses = solve_reference_set(
        coords, demands, capacity_int=9, time_limit_s=1
    )
    assert costs.shape == (2,)
    assert costs.dtype == torch.float32
    assert statuses == [1, 1]
    assert torch.allclose(costs, torch.tensor([4.0, 4.0]), atol=_TOL)