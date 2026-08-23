"""Tests for the generate_cvrp `return_raw` flag.

Locks the two guarantees the reference pipeline depends on:
  1. Additivity — passing return_raw does not change the sampled instances, so
     eval and the reference solver see bit-identical coords/demand.
  2. Raw contract — integer demands (depot 0, range [1, 9]) and the integer
     capacity D reconstruct the normalised demand exactly.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp


def test_return_raw_is_additive() -> None:
    # Same seed, with and without the flag -> identical coords and demand.
    c0, d0, cap0 = generate_cvrp(4, 20, seed=2024)
    c1, d1, cap1, _dem_raw, _D = generate_cvrp(4, 20, seed=2024, return_raw=True)
    assert torch.equal(c0, c1)
    assert torch.equal(d0, d1)
    assert cap0 == cap1 == 1.0


def test_raw_contract_shapes_and_range() -> None:
    _c, _d, _cap, dem_raw, D = generate_cvrp(4, 20, seed=2024, return_raw=True)
    assert dem_raw.shape == (4, 21)          # (B, N), N = n_customers + 1
    assert dem_raw.dtype == torch.long
    assert torch.equal(dem_raw[:, 0], torch.zeros(4, dtype=torch.long))  # depot 0
    cust = dem_raw[:, 1:]
    assert int(cust.min()) >= 1 and int(cust.max()) <= 9  # demand_range (1, 9)
    assert D == 30                           # _DEFAULT_CAPACITY[20]


def test_raw_reconstructs_normalised_demand() -> None:
    _c, demand, _cap, dem_raw, D = generate_cvrp(8, 50, seed=7, return_raw=True)
    assert torch.allclose(dem_raw.float() / D, demand, atol=0.0)  # exact, no round-trip


def test_capacity_override_sets_D() -> None:
    _c, _d, _cap, _dem_raw, D = generate_cvrp(2, 50, capacity=40, seed=1, return_raw=True)
    assert D == 40