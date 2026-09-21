"""Regression test: unseeded generate_cvrp must return DIFFERENT instances per call.

Guards the Stage-1 bug where an unseeded torch.Generator (fixed default seed
67280421310721) made every training batch identical.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp


def test_unseeded_batches_differ() -> None:
    torch.manual_seed(0)
    a_coords, a_dem, _ = generate_cvrp(32, 20)
    b_coords, b_dem, _ = generate_cvrp(32, 20)
    assert not torch.equal(a_coords, b_coords)
    assert not torch.equal(a_dem, b_dem)


def test_unseeded_run_is_reproducible_under_global_seed() -> None:
    torch.manual_seed(1234)
    first = [generate_cvrp(8, 20)[0] for _ in range(3)]
    torch.manual_seed(1234)
    second = [generate_cvrp(8, 20)[0] for _ in range(3)]
    for x, y in zip(first, second):
        assert torch.equal(x, y)


def test_seeded_call_unchanged() -> None:
    """Seeded sets (val/test/reference) must be bit-identical to the old generator."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2024)
    coords_old = torch.rand(4, 21, 2, generator=gen)            # old code path, verbatim
    dem_old = torch.randint(1, 10, (4, 20), generator=gen)
    coords, _, _, dem_raw, _ = generate_cvrp(4, 20, seed=2024, return_raw=True)
    assert torch.equal(coords, coords_old)
    assert torch.equal(dem_raw[:, 1:], dem_old)