"""CVRPTW instance generator (deterministic variant A): CVRP + hard time windows.

Extends the Kool/POMO CVRP sampler with a service-start time window and a service
duration per customer, plus a per-route shift horizon (day length). Windows are
generated FEASIBLE BY CONSTRUCTION: every customer is individually servable from a
fresh shift (leave the depot at clock 0, reach the customer within its window,
serve it, and return before the shift ends). That invariant is what lets hard
masking never dead-end (there is always at least one servable customer).

Semantics (Solomon 1987 convention, as used by learned-VRPTW work incl. SED2AM):
  - window [earliest, latest] constrains the START OF SERVICE. Arriving early is
    allowed (wait until `earliest`); arriving after `latest` is infeasible.
  - service_time is a fixed duration; Solomon uses 10 min, normalized here to 0.2
    for coords in the unit square (the common learned-VRPTW scaling).

Travel time (variant A): time == distance, so `pairwise_distances` IS the travel
time. The stochastic variants (B/C/D) replace this with an Edge-Bank lookup behind
the same interface; nothing else in the env/masking needs to change.

shift_end and tw_half_width are the calibration dials for how time-binding the
instances are; sensible starting defaults are set here and tuned empirically with
the mask-reason diagnostic (next module).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from src.data.generator import generate_cvrp

# Service duration: Solomon (1987) 10 min, normalized (SED2AM / Kool-style) -> 0.2.
_DEFAULT_SERVICE = 0.2
# Max depot->customer distance in the unit square (opposite corners).
_MAX_RADIUS = math.sqrt(2.0)


class CVRPTWData(NamedTuple):
    """A batch of CVRPTW instances (variant A)."""

    coords: Tensor        # (B, N, 2)   node 0 is the depot
    demand: Tensor        # (B, N)      normalized (0, 1]; depot 0
    capacity: float       # normalized vehicle capacity (1.0)
    time_windows: Tensor  # (B, N, 2)   [earliest, latest] service-start; depot [0, shift_end]
    service_time: Tensor  # (B, N)      service duration; depot 0
    shift_end: float      # per-route shift horizon (a route must finish by this)


def pairwise_distances(coords: Tensor) -> Tensor:
    """Euclidean distance matrix; this is variant-A travel time (time == distance).

    coords: (B, N, 2) -> (B, N, N).  Explicit pairwise norm (not torch.cdist) so
    the diagonal is exactly 0 and there is no matmul-mode ambiguity.
    """
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 2)
    return diff.norm(dim=-1)                            # (B, N, N)


def generate_cvrptw(
    batch_size: int,
    n_customers: int,
    capacity: int | None = None,           # vehicle capacity in demand units; None -> lookup
    demand_range: tuple[int, int] = (1, 9),
    service_time: float = _DEFAULT_SERVICE,
    shift_end: float = 4.0,                 # calibration dial (see module docstring)
    tw_half_width: float = 0.5,             # calibration dial: window half-width
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> CVRPTWData:
    """Sample a batch of feasible-by-construction CVRPTW instances.

    Every customer is guaranteed individually servable from a fresh shift. Raises
    if `shift_end` is below the reachability floor 2*sqrt(2)+service_time, since a
    far-corner customer would then be unservable regardless of its window.
    """
    floor = 2.0 * _MAX_RADIUS + service_time
    if shift_end < floor:
        raise ValueError(
            f"shift_end={shift_end} is below the reachability floor {floor:.4f} "
            f"(= 2*sqrt(2)+service_time); a far-corner customer would be unservable"
        )
    if tw_half_width < 0.0:
        raise ValueError(f"tw_half_width must be >= 0, got {tw_half_width}")

    # CVRP part (coords, demand, capacity): reuse the tested sampler, on CPU.
    coords, demand, cap = generate_cvrp(
        batch_size, n_customers, capacity, demand_range=demand_range,
        seed=seed, device="cpu",
    )
    b, n, _ = coords.shape

    # Depot<->node distance = round-trip half (Euclidean is symmetric).
    r = pairwise_distances(coords)[:, 0, :]              # (B, N); r[:, 0] = 0

    # Feasible service-start interval [lo, hi] per customer:
    #   lo = earliest possible arrival (leave depot at 0)
    #   hi = latest start that still allows service + return within the shift
    # Non-empty for every customer because r <= sqrt(2) and shift_end >= floor.
    lo = r                                               # (B, N)
    hi = shift_end - service_time - r                    # (B, N)

    # Sample a window centre in [lo, hi] with an independent, seeded stream so
    # the same `seed` reproduces the same instance (windows included).
    # Unseeded torch.Generator objects all start from the same fixed default seed, so
    # an unseeded gen_tw returned the SAME window draws on every call (training batches
    # shared one window pattern). Unseeded calls now draw a seed from the global RNG;
    # seeded calls (val/test) are bit-identical to before.
    gen_tw = torch.Generator(device="cpu")
    gen_tw.manual_seed(
        seed + 1 if seed is not None else int(torch.randint(0, 2**62, (1,)).item())
    )
    u = torch.rand(b, n, generator=gen_tw)               # (B, N)
    centre = lo + u * (hi - lo).clamp_min(0.0)           # (B, N)

    earliest = (centre - tw_half_width).clamp_min(0.0)   # (B, N)
    latest = centre + tw_half_width                      # (B, N)

    # Depot: open the whole shift, no service.
    earliest[:, 0] = 0.0
    latest[:, 0] = float(shift_end)
    service = torch.full((b, n), float(service_time))    # (B, N)
    service[:, 0] = 0.0

    time_windows = torch.stack([earliest, latest], dim=-1)  # (B, N, 2)

    dev = torch.device(device)
    return CVRPTWData(
        coords=coords.to(dev),
        demand=demand.to(dev),
        capacity=float(cap),
        time_windows=time_windows.to(dev),
        service_time=service.to(dev),
        shift_end=float(shift_end),
    )