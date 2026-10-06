"""Classical reference for the stochastic CVRPTW: PyVRP plans, executed on OUR days.

WHAT THIS IS -- AND IS NOT
--------------------------
PyVRP (hybrid genetic search) is a DETERMINISTIC VRPTW solver. At n = 20-25 it is
near-optimal for the deterministic problem. It does NOT solve the stochastic problem.
Two uses here:

  1. Deterministic reference (alpha = 0, beta_var = 0): travel time == distance, so
     PyVRP's plan is a near-optimal answer to exactly the problem the env scores. This
     gives a true "gap to (near-)optimal" for the neural policies.
  2. Mean-value plan ("expected value" plan in stochastic programming): PyVRP plans on
     the DAY-AVERAGE travel time. The ripple is mean-preserving (SimulatorConfig.
     is_mean_preserving), so the day-average of mu_ij is exactly d_ij and the
     mean-value plan IS the deterministic plan. The plan is then FIXED and executed on
     the same sampled days as every neural number, in the same env, under the same
     cost (distance + w * (late_count + tardiness)). It is what a practitioner without
     a stochastic model would do -- never call it "optimal" for the stochastic problem.

  X0b, the sigma-buffered plan: durations d + k * sqrt(sigma_bar^2), with sigma_bar^2
  the horizon-averaged variance. A classical, risk-aware plan. It tests whether sigma^2
  is useful information at all in this simulator.

INTEGER SCALING (PyVRP is integer-only)
---------------------------------------
Everything is multiplied by SCALE and rounded CONSERVATIVELY: durations, service and
window opening UP, window closing and shift end DOWN. A plan PyVRP calls feasible is
then feasible in the float env (tested). Distances (objective only) are rounded to
nearest; the reported distance is always recomputed by the env in float.

EXECUTION SEMANTICS
-------------------
CVRPEnv.step does not consult the mask, so a fixed plan is driven exactly as planned:
a move the mean-based mask would forbid under the ripple is still taken, and any
lateness it causes is CHARGED. That is the correct semantics for an a-priori plan.
The vehicle leaves the depot at clock 0 and waits if early (PyVRP may schedule a later
departure; with waiting allowed, leaving earlier is never worse, so feasibility is
unaffected).
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData, pairwise_distances
from src.data.simulator import (
    SIGMA2,
    SimulatorConfig,
    build_edge_bank,
    draw_noise,
    horizon_average,
)
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv

SCALE = 10_000          # integer resolution: 1e-4 of a unit-square side
_ROUND_TOL = 1e-6       # scaled units; absorbs float noise like 0.2 * 1e4 = 2000.0000000002


@dataclass(frozen=True)
class PlanResult:
    """One instance's PyVRP plan."""

    routes: tuple[tuple[int, ...], ...]   # node indices 1..N-1 per route, depot excluded
    feasible: bool                        # PyVRP's own verdict (no time warp / excess load)
    complete: bool                        # every customer is in some route
    runtime: float                        # seconds spent in PyVRP's search
    pyvrp_distance: float                 # PyVRP objective / SCALE (rounded distances)


# --------------------------------------------------------------------------- packing


def _up(x: np.ndarray) -> np.ndarray:
    """Scale and round UP (conservative for durations, service, window opening)."""
    return np.ceil(x * SCALE - _ROUND_TOL).astype(np.int64)


def _down(x: np.ndarray) -> np.ndarray:
    """Scale and round DOWN (conservative for window closing and the shift end)."""
    return np.floor(x * SCALE + _ROUND_TOL).astype(np.int64)


def demand_units(data: CVRPTWData, capacity_units: int) -> np.ndarray:
    """Integer demands in vehicle-capacity units. -> (B, N) int64.

    The generator stores demand as a fraction of the vehicle (demand / capacity in
    (0, 1]); `capacity_units` is the integer capacity the instances were generated with
    (the checkpoint's 'capacity'). Raises if the fractions are not exact multiples of
    1 / capacity_units, i.e. if the instance cannot be represented in integers.
    """
    ratio = (data.demand / data.capacity).double().cpu().numpy()          # (B, N)
    units = np.rint(ratio * capacity_units).astype(np.int64)              # (B, N)
    if not np.allclose(units / capacity_units, ratio, atol=1e-6):
        raise ValueError(
            "demand / capacity is not a multiple of 1/capacity_units; check the "
            "capacity_units value (the checkpoint's integer 'capacity')"
        )
    return units


def pack_instances(
    data: CVRPTWData,
    capacity_units: int,
    durations: Tensor | None = None,    # (B, N, N) planning travel times; None = distance
) -> list[dict[str, Any]]:
    """Integer, picklable per-instance inputs for `solve_one`. One dict per instance."""
    b, n, _ = data.coords.shape
    coords = data.coords.double().cpu().numpy()                           # (B, N, 2)
    dist = pairwise_distances(data.coords.double()).cpu().numpy()        # (B, N, N)
    dur = dist if durations is None else durations.double().cpu().numpy()
    if dur.shape != (b, n, n):
        raise ValueError(f"durations must be {(b, n, n)}, got {dur.shape}")
    if (dur < dist - 1e-9).any():
        raise ValueError("planning durations below the distance would let a plan "
                         "assume faster-than-mean travel")
    tw = data.time_windows.double().cpu().numpy()                          # (B, N, 2)
    svc = data.service_time.double().cpu().numpy()                        # (B, N)
    units = demand_units(data, capacity_units)                            # (B, N)
    depot_late = int(_down(np.array(data.shift_end)))

    out: list[dict[str, Any]] = []
    for i in range(b):
        out.append({
            "xy": np.rint(coords[i] * SCALE).astype(np.int64),            # (N, 2)
            "delivery": units[i],                                         # (N,)
            "capacity": int(capacity_units),
            "service": _up(svc[i]),                                       # (N,)
            "early": _up(tw[i, :, 0]),                                    # (N,)
            "late": _down(tw[i, :, 1]),                                   # (N,)
            "depot_late": depot_late,
            "distance": np.rint(dist[i] * SCALE).astype(np.int64),       # (N, N)
            "duration": _up(dur[i]),                                      # (N, N)
        })
    return out


# --------------------------------------------------------------------------- solving


def solve_one(inst: dict[str, Any], time_limit: float, seed: int = 0) -> PlanResult:
    """Solve one packed instance with PyVRP (0.14 API). Module-level so it pickles."""
    from pyvrp import Model                    # local: workers import it themselves
    from pyvrp.stop import MaxRuntime

    n = inst["xy"].shape[0]
    m = Model()
    locs = [m.add_location(x=int(x), y=int(y)) for x, y in inst["xy"]]
    depot = m.add_depot(locs[0], tw_early=0, tw_late=inst["depot_late"])
    m.add_vehicle_type(
        num_available=n - 1, capacity=inst["capacity"], start_depot=depot,
        end_depot=depot, tw_early=0, tw_late=inst["depot_late"],
    )
    for j in range(1, n):                      # clients added in node order: client k = node k+1
        m.add_client(
            locs[j], delivery=int(inst["delivery"][j]),
            service_duration=int(inst["service"][j]),
            tw_early=int(inst["early"][j]), tw_late=int(inst["late"][j]),
        )
    for a in range(n):
        for c in range(n):
            m.add_edge(locs[a], locs[c], distance=int(inst["distance"][a, c]),
                       duration=int(inst["duration"][a, c]))

    with warnings.catch_warnings():
        # PenaltyBoundWarning fires when an instance is hard or impossible to make feasible
        # (e.g. a large buffer). Silenced here because the verdict is RECORDED instead:
        # `feasible` below, and the fallback chain in `fallback_to_feasible`.
        warnings.simplefilter("ignore")
        res = m.solve(stop=MaxRuntime(time_limit), seed=seed, display=False)
    sol = res.best
    routes = tuple(
        tuple(int(act.idx) + 1 for act in route if act.is_client())
        for route in sol.routes()
    )
    visited = sorted(c for r in routes for c in r)
    return PlanResult(
        routes=routes,
        feasible=bool(sol.is_feasible()),
        complete=visited == list(range(1, n)),
        runtime=float(res.runtime),
        pyvrp_distance=float(sol.distance()) / SCALE,
    )


def _solve_star(args: tuple[dict[str, Any], float, int]) -> PlanResult:
    return solve_one(*args)


def solve_batch(
    instances: Sequence[dict[str, Any]],
    time_limit: float,
    seed: int = 0,
    workers: int = 1,
) -> list[PlanResult]:
    """Solve every instance; parallel over processes if workers > 1. Order preserved.

    Each solve is limited by WALL-CLOCK time, so use at most one worker per PHYSICAL
    core: oversubscribing gives each search less CPU and silently weakens the reference.
    """
    jobs = [(inst, time_limit, seed) for inst in instances]
    if workers <= 1:
        return [_solve_star(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_solve_star, jobs, chunksize=1))


# --------------------------------------------------------------------------- executing


def routes_to_actions(plans: Sequence[PlanResult], n: int, device: torch.device) -> Tensor:
    """Plans -> env action sequences [r1..., 0, r2..., 0, ...], padded with 0. (B, T) long.

    Raises on an incomplete plan: executing it would leave customers unserved and the
    cost would look artificially low.
    """
    seqs: list[list[int]] = []
    for i, p in enumerate(plans):
        if not p.complete:
            raise ValueError(f"plan {i} does not visit every customer exactly once")
        seq: list[int] = []
        for route in p.routes:
            seq.extend(route)
            seq.append(0)
        seqs.append(seq)
    t = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), t, dtype=torch.long, device=device)   # (B, T)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    if not bool((out.unsqueeze(-1) == torch.arange(1, n, device=device)).sum(1).eq(1).all()):
        raise ValueError("an action sequence visits some customer more or less than once")
    return out


def draw_test_days(
    b: int, n: int, sim_cfg: SimulatorConfig, seeds: Sequence[int],
    device: torch.device, dtype: torch.dtype,
) -> list[Tensor]:
    """The realised days, drawn EXACTLY as evaluate_stoch_apriori draws them. (B, N, N, K) each.

    Same seeds, same shape (the WHOLE test set, sliced later), same device. A CUDA and a
    CPU generator give different numbers for the same seed, so the days only match an
    earlier run made on the same device type.
    """
    return [
        draw_noise((b, n, n, sim_cfg.n_bins), device=device, dtype=dtype,
                   generator=torch.Generator(device=device).manual_seed(int(s)))
        for s in seeds
    ]


@torch.no_grad()
def execute_actions(
    data: CVRPTWData,
    actions: Tensor,                 # (B, T) long
    sim_cfg: SimulatorConfig,
    tardiness_weight: float,
    days: Sequence[Tensor],          # each (B, N, N, K)
) -> dict[str, Tensor]:
    """Drive FIXED action sequences on each day; mean over days. key -> (B,).

    Keys: distance, late_count, tardiness, penalty, total (CVRPTWStochEnv.cost_components).
    """
    if len(days) == 0:
        raise ValueError("need at least one day")
    bank = build_edge_bank(data.coords, sim_cfg)                          # (B, N, N, K, 2)
    totals: dict[str, Tensor] = {}
    for noise in days:
        env = CVRPTWStochEnv(
            data.coords, data.demand, data.capacity, data.time_windows,
            data.service_time, data.shift_end, bank=bank, sim_cfg=sim_cfg,
            noise=noise, tardiness_weight=tardiness_weight, reward_mode="sampled",
        )
        env.reset()
        for t in range(actions.size(1)):
            env.step(actions[:, t])
        if not bool(env.visited[:, 1:].all()):
            raise RuntimeError("a fixed plan left customers unserved")
        parts = env.cost_components(actions)
        for k, v in parts.items():
            totals[k] = v.clone() if k not in totals else totals[k] + v
    return {k: v / len(days) for k, v in totals.items()}


def buffered_durations(data: CVRPTWData, sim_cfg: SimulatorConfig, k: float) -> Tensor:
    """Planning travel times d + k * sqrt(sigma_bar^2). (B, N, N).

    sigma_bar^2 is the horizon-averaged variance of the Edge Bank at sim_cfg.beta_var --
    the same static quantity variant B's encoder receives. k = 0 returns the distance.
    """
    if k < 0.0:
        raise ValueError(f"k must be >= 0, got {k}")
    d = pairwise_distances(data.coords)                                   # (B, N, N)
    if k == 0.0:
        return d
    sigma2_bar = horizon_average(build_edge_bank(data.coords, sim_cfg))[..., SIGMA2]
    return d + k * sigma2_bar.clamp_min(0.0).sqrt()                       # (B, N, N)


def fallback_to_feasible(
    plans_by_k: Sequence[Sequence[PlanResult]],   # [k index][instance], k ASCENDING, k[0] == 0
) -> tuple[list[list[PlanResult]], list[int]]:
    """Replace each infeasible buffered plan by the plan of the next SMALLER buffer.

    A large buffer can make a customer unreachable even from a fresh shift, and PyVRP then
    returns a time-warped (infeasible) plan. Executing that plan would mix "risk-aware
    planning" with "planning on an impossible instance", so the instance falls back to the
    largest buffer that was still feasible. Returns (chosen plans per k, fallbacks per k).
    The k = 0 plans are kept as they are (they are the unbuffered reference).
    """
    if len(plans_by_k) == 0:
        raise ValueError("need at least the k = 0 plans")
    chosen: list[list[PlanResult]] = [list(plans_by_k[0])]
    fallbacks: list[int] = [0]
    for plans in plans_by_k[1:]:
        if len(plans) != len(chosen[0]):
            raise ValueError("every k must have one plan per instance")
        row, count = [], 0
        for i, p in enumerate(plans):
            if p.feasible and p.complete:
                row.append(p)
            else:
                row.append(chosen[-1][i])
                count += 1
        chosen.append(row)
        fallbacks.append(count)
    return chosen, fallbacks


def summarise_plans(plans: Sequence[PlanResult]) -> dict[str, float]:
    """Feasibility / completeness counts and runtime of a batch of plans."""
    return {
        "n": float(len(plans)),
        "n_feasible": float(sum(p.feasible for p in plans)),
        "n_complete": float(sum(p.complete for p in plans)),
        "mean_runtime_s": float(np.mean([p.runtime for p in plans])),
        "mean_pyvrp_distance": float(np.mean([p.pyvrp_distance for p in plans])),
    }


def wall_time(fn, device: torch.device) -> tuple[Any, float]:
    """Run fn() and return (result, seconds), synchronising CUDA around it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0


__all__ = [
    "SCALE", "PlanResult", "demand_units", "pack_instances", "solve_one", "solve_batch",
    "routes_to_actions", "draw_test_days", "execute_actions", "buffered_durations",
    "fallback_to_feasible", "summarise_plans", "wall_time",
]