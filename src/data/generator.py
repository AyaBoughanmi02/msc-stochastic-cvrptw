"""Random CVRP instance generator (standard Kool/POMO convention).

Coordinates are uniform in the unit square; integer demands in [1, 9] are
normalised by a size-dependent vehicle capacity so the env runs with capacity
1.0 and demands in (0, 1]. The depot is node 0 with demand 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Standard capacities by customer count (Kool et al. 2019 / Kwon et al. 2020).
_DEFAULT_CAPACITY: dict[int, int] = {10: 20, 20: 30, 50: 40, 100: 50}


def generate_cvrp(
    batch_size: int,
    n_customers: int,
    capacity: int | None = None,      # vehicle capacity in demand units; None -> lookup
    demand_range: tuple[int, int] = (1, 9),
    seed: int | None = None,
    device: str | torch.device = "cpu",
    return_raw: bool = False,         # also return integer demands + capacity D
) -> tuple[Tensor, Tensor, float] | tuple[Tensor, Tensor, float, Tensor, int]:
    """Sample a batch of CVRP instances.

    Returns (return_raw=False, default):
        coords:   (B, N, 2)  N = n_customers + 1, node 0 is the depot
        demand:   (B, N)     normalised to (0, 1]; demand[:, 0] = 0
        capacity: float      normalised vehicle capacity (always 1.0)

    Returns (return_raw=True), additionally:
        demand_raw: (B, N)   integer demands in [1, 9]; demand_raw[:, 0] = 0
        D:          int      integer vehicle capacity (the normalisation divisor)

    The raw values reuse the same draws as the normalised ones (no extra RNG
    calls), so instances are bit-identical regardless of `return_raw`. They exist
    for external solvers (e.g. OR-Tools) that need integer demands and capacity.
    """
    if capacity is None:
        if n_customers not in _DEFAULT_CAPACITY:
            raise ValueError(
                f"no default capacity for n_customers={n_customers}; pass capacity="
            )
        capacity = _DEFAULT_CAPACITY[n_customers]

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    n = n_customers + 1
    coords = torch.rand(batch_size, n, 2, generator=gen)          # (B, N, 2)
    low, high = demand_range
    dem_int = torch.randint(low, high + 1, (batch_size, n_customers), generator=gen)
    demand = dem_int.float() / float(capacity)                    # normalise -> (0, 1]
    demand = torch.cat([torch.zeros(batch_size, 1), demand], dim=1)  # depot demand 0

    if return_raw:
        dem_raw = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.long), dem_int], dim=1
        )  # (B, N) int64, depot column 0
        return coords.to(device), demand.to(device), 1.0, dem_raw.to(device), int(capacity)

    return coords.to(device), demand.to(device), 1.0