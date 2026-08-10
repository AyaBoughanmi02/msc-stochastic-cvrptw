"""Tests for the CVRP environment and masking primitives.

Uses a unit-square instance whose tour lengths are hand-computable, so the
capacity/depot/masking logic is checked against known-correct numbers rather
than against itself.

    node 0 = depot   (0, 0)
    node 1           (1, 0)
    node 2           (1, 1)
    node 3           (0, 1)
"""

from __future__ import annotations

import torch

from src.envs.cvrp_env import CVRPEnv
from src.utils.masking import tour_length

_SQUARE = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])  # (1, 4, 2)


def _env(demand: list[float], capacity: float = 1.0) -> CVRPEnv:
    d = torch.tensor([demand])  # (1, 4)
    return CVRPEnv(_SQUARE.clone(), d, capacity)


def test_initial_mask_forbids_depot_allows_customers() -> None:
    env = _env([0.0, 0.3, 0.3, 0.3])
    mask = env.reset()
    # at depot: depot forbidden, all customers feasible
    assert mask.tolist() == [[True, False, False, False]]


def test_force_first_restricts_to_one_customer() -> None:
    env = _env([0.0, 0.3, 0.3, 0.3])
    mask = env.reset(force_first=torch.tensor([2]))
    assert mask.tolist() == [[True, True, False, True]]  # only customer 2 feasible


def test_visited_and_depot_feasibility_after_step() -> None:
    env = _env([0.0, 0.3, 0.3, 0.3])
    env.reset()
    mask, done = env.step(torch.tensor([1]))            # visit customer 1
    # customer 1 now visited; depot feasible (we left it); 2 and 3 fit in 0.7
    assert mask.tolist() == [[False, True, False, False]]
    assert done.tolist() == [False]
    assert torch.isclose(env.remaining_capacity, torch.tensor([0.7]))


def test_capacity_forces_depot_return() -> None:
    env = _env([0.0, 0.6, 0.6, 0.6])                    # two customers can't share a trip
    env.reset()
    mask, _ = env.step(torch.tensor([1]))              # remaining 0.4
    # customers 2,3 need 0.6 > 0.4 -> masked; only depot feasible
    assert mask.tolist() == [[False, True, True, True]]
    mask, _ = env.step(torch.tensor([0]))              # return to depot -> refill
    assert torch.isclose(env.remaining_capacity, torch.tensor([1.0]))
    # at depot now: depot masked, customers 2,3 feasible again, 1 visited
    assert mask.tolist() == [[True, True, False, False]]


def test_full_rollout_terminates_all_visited() -> None:
    env = _env([0.0, 0.3, 0.3, 0.3])
    env.reset()
    done = torch.tensor([False])
    for a in (1, 2, 3):
        _, done = env.step(torch.tensor([a]))
    assert done.tolist() == [True]


def test_tour_length_unit_square() -> None:
    # depot -> 1 -> 2 -> 3 -> depot around the unit square = 4.0
    actions = torch.tensor([[1, 2, 3]])
    assert torch.isclose(tour_length(_SQUARE, actions), torch.tensor([4.0]))


def test_tour_length_with_depot_returns() -> None:
    # depot->1->depot->2->depot->3->depot. Legs to/from node 2 are the diagonal
    # (sqrt 2); the other four legs are length 1. Total = 4 + 2*sqrt(2).
    actions = torch.tensor([[1, 0, 2, 0, 3]])
    expected = 4.0 + 2.0 * (2.0 ** 0.5)
    assert torch.isclose(tour_length(_SQUARE, actions), torch.tensor([expected]))


def test_normalized_remaining_shape_and_range() -> None:
    env = _env([0.0, 0.3, 0.3, 0.3])
    env.reset()
    env.step(torch.tensor([1]))
    nr = env.normalized_remaining()
    assert nr.shape == (1, 1)
    assert torch.isclose(nr, torch.tensor([[0.7]]))


def test_batched_instances_independent() -> None:
    coords = _SQUARE.repeat(3, 1, 1)                    # (3, 4, 2)
    demand = torch.tensor([[0.0, 0.3, 0.3, 0.3]]).repeat(3, 1)
    env = CVRPEnv(coords, demand, capacity=1.0)
    env.reset(force_first=torch.tensor([1, 2, 3]))     # each starts differently
    mask, _ = env.step(torch.tensor([1, 2, 3]))
    # each instance visited a different customer -> different visited columns
    assert env.visited.tolist() == [
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]