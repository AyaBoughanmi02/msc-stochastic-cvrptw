"""OR-Tools reference solver for CVRP instances.

Computes near-optimal tour lengths for the SAME instances the model is evaluated
on, so ``scripts/eval.py --ref`` can report an optimality gap. OR-Tools' routing
solver is a metaheuristic (guided local search), so these are near-optimal, NOT
proven-optimal references: a negative gap (POMO + 8x augmentation beating the
reference) is expected, not a bug.

Cost convention matches ``src/utils/masking.py::tour_length``: plain Euclidean L2
in the unit square, summed over the depot-anchored tour. Distances are scaled by
``scale`` and rounded to integers for OR-Tools, then the objective is divided
back, so reference costs are in the same [0, 1] units as the model's cost.

The first-solution strategy and metaheuristic are LOCKED (not configurable) so
the reference is consistent across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Locked search configuration (deliberately not exposed as knobs).
_FIRST_SOLUTION = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
_METAHEURISTIC = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

# routing.status() == 1 is ROUTING_SUCCESS in the OR-Tools routing enum.
_ROUTING_SUCCESS = 1

# Default distance scale: round(scale * L2). Error per arc <= 0.5/scale; over
# ~2N arcs the total is <= ~N*1e-6, negligible vs tour lengths of order 5-10.
DEFAULT_SCALE = 1_000_000


def _int_distance_matrix(coords_b: Tensor, scale: int) -> list[list[int]]:
    """Integer L2 distance matrix for one instance.

    coords_b: (N, 2) float coords in [0, 1]
    returns:  N x N nested list of ints, dist[i][j] = round(scale * ||c_i - c_j||)
    """
    c = coords_b.detach().cpu()                       # (N, 2)
    diff = c.unsqueeze(1) - c.unsqueeze(0)            # (N, N, 2)
    d = diff.norm(dim=-1)                             # (N, N) exact L2, diagonal 0
    return (d * scale).round().to(torch.int64).tolist()


def solve_cvrp_ortools(
    dist_int: list[list[int]],   # N x N integer distances
    demands_int: list[int],      # (N,) integer demands, demands_int[0] = 0 (depot)
    capacity_int: int,           # integer vehicle capacity D
    num_vehicles: int,           # upper bound on routes (n_customers is safe)
    time_limit_s: int,           # per-instance GLS wall-clock budget, seconds
) -> tuple[int | None, int]:
    """Solve one CVRP instance with OR-Tools guided local search.

    Returns (objective, status): ``objective`` is the tour length in scaled-int
    units (divide by ``scale`` for [0, 1] cost), or ``None`` if no solution was
    found. ``status`` is ``routing.status()`` (1 == ROUTING_SUCCESS), kept for
    provenance in the reference sidecar.
    """
    n = len(dist_int)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)  # depot = node 0
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_index: int, to_index: int) -> int:
        return dist_int[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(from_index: int) -> int:
        return demands_int[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                                  # no slack
        [capacity_int] * num_vehicles,      # per-vehicle capacity
        True,                               # start cumul at zero
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = _FIRST_SOLUTION
    params.local_search_metaheuristic = _METAHEURISTIC
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    status = routing.status()               # numeric; 1 == ROUTING_SUCCESS
    if solution is None:                    # authoritative "no solution" check
        return None, status
    return solution.ObjectiveValue(), status


def solve_reference_set(
    coords: Tensor,          # (B, N, 2) float in [0, 1]
    demands_int: Tensor,     # (B, N) int, depot column = 0
    capacity_int: int,       # integer vehicle capacity D
    scale: int = DEFAULT_SCALE,
    time_limit_s: int = 2,
) -> tuple[Tensor, list[int]]:
    """Solve every instance in a batch and return per-instance reference costs.

    coords:       (B, N, 2)
    demands_int:  (B, N) integer demands (depot column = 0)
    returns:      (costs, statuses); costs is (B,) float32 in [0, 1] units (same
                  convention as masking.tour_length) and statuses is the length-B
                  list of routing.status() codes (provenance for the sidecar).

    Raises RuntimeError if any instance fails to solve (naming the indices), so a
    broken reference aborts loudly instead of saving garbage costs.
    """
    b, n, _ = coords.shape
    num_vehicles = n - 1                     # = n_customers; safe upper bound
    costs = torch.empty(b, dtype=torch.float64)
    statuses: list[int] = []
    failed: list[int] = []

    for i in range(b):
        dist_int = _int_distance_matrix(coords[i], scale)
        demand_row = demands_int[i].detach().cpu().to(torch.int64).tolist()  # (N,)
        objective, status = solve_cvrp_ortools(
            dist_int, demand_row, capacity_int, num_vehicles, time_limit_s
        )
        statuses.append(status)
        if objective is None:
            failed.append(i)
            costs[i] = float("nan")
        else:
            costs[i] = objective / scale

    if failed:
        shown = failed[:10]
        raise RuntimeError(
            f"OR-Tools found no solution for {len(failed)} instance(s): "
            f"indices {shown}{'...' if len(failed) > 10 else ''}. "
            f"Increase time_limit_s or inspect the instance generator."
        )
    return costs.to(torch.float32), statuses


def load_reference(ref_path: str | Path) -> tuple[Tensor, dict | None]:
    """Load reference costs and the provenance sidecar (if it exists).

    ref_path: path to the .pt written by scripts/gen_reference.py.
    returns:  (costs, sidecar); costs is (B,) float32 on CPU, sidecar is the
              parsed JSON dict or None when the .json is missing (e.g. an old
              reference generated before sidecars existed).
    """
    ref_path = Path(ref_path)
    costs = torch.load(ref_path, map_location="cpu").float()   # (B,)
    sidecar_path = ref_path.with_suffix(".json")
    sidecar: dict | None = None
    if sidecar_path.exists():
        with sidecar_path.open(encoding="utf-8") as f:
            sidecar = json.load(f)
    return costs, sidecar


def assert_reference_alignment(
    sidecar: dict,
    n_customers: int,
    test_seed: int,
    test_size: int,
) -> None:
    """Raise if the reference was built on different instances than eval uses.

    A mismatch in (n_customers, test_seed, test_size) produces the SAME tensor
    shape but different instances, so the per-instance gap would be silently
    meaningless. This fails loudly instead. Note: vehicle capacity is NOT checked
    here (it defaults per size); if you override --capacity, regenerate the
    reference to match.
    """
    inst = sidecar.get("instances", {})
    checks = {
        "n_customers": (inst.get("n_customers"), n_customers),
        "test_seed": (inst.get("test_seed"), test_seed),
        "test_size": (inst.get("test_size"), test_size),
    }
    mismatches = [
        f"{k}: reference {ref!r} != eval {cur!r}"
        for k, (ref, cur) in checks.items()
        if ref != cur
    ]
    if mismatches:
        raise ValueError(
            "reference/eval instance mismatch -- the optimality gap would be "
            "meaningless:\n  " + "\n  ".join(mismatches)
            + "\nRegenerate the reference with matching flags, or point --ref at "
            "the correct file."
        )