"""Rolling-horizon PyVRP: an ONLINE classical re-planner for the stochastic CVRPTW.

WHY THIS EXISTS
---------------
`src/eval/pyvrp_reference.py` freezes one PyVRP plan before the day. The neural policies
re-decide at every stop with the real clock, so "neural beats PyVRP" there partly means
"online beats frozen". This module removes that confound: PyVRP re-solves the REMAINING
problem at every decision, from the vehicle's real position, real clock and real
remaining capacity, and only the NEXT move of its new plan is executed. Online vs online.

WHAT ONE DECISION LOOKS LIKE (one instance, one realised day)
-------------------------------------------------------------
State read from CVRPTWStochEnv: current node c, clock tau (ready to leave c), remaining
capacity q, unvisited customers U.

  * c == depot: a new vehicle starts at clock 0 (the env resets the clock at the depot).
    PyVRP solves U with vehicles from the depot; the next move is the first customer of
    one of its routes (any route: they all start at 0 and are independent).
  * c != depot: the CURRENT vehicle must continue from c at tau with capacity q, other
    vehicles start fresh from the depot. Modelled as:
      - a second depot D1 at c's location, a vehicle type A (1 vehicle) starting at D1
        at exactly tau (tw_early = start_late = tau), capacity q, ending at the depot;
      - a required DUMMY client at c's location with window [tau, tau] and no demand,
        which forces vehicle A to be used. Without it PyVRP could leave A unused at
        zero cost, while in reality the van at c must still drive home (d(c, depot)),
        so "return now" would be under-priced;
      - vehicle type B (fresh vehicles from the depot) uses a second routing PROFILE in
        which c's location is unreachable within the dummy's window, so only A can
        serve the dummy.
    The next move is the first real customer after the dummy on A's route, or the depot
    if A's route is [dummy] only.

PLANNING TRAVEL TIMES: static, the same matrix for every decision (default: distance =
the day-average mean, as the X0 mean-value plan; optionally the X0b buffer
d + k * sqrt(sigma_bar^2)). Integer scaling and conservative rounding as in
pyvrp_reference.pack_instances. The executed travel times are the env's realised ones.

Infeasible subproblems (already late, so a window cannot be met) are still solved;
PyVRP returns its best penalised (time-warped) plan, whose next move is executed and
the lateness charged by the env. The count of such decisions is reported.

STOPPING: MaxIterations, not wall-clock, so the result does not depend on how loaded the
machine is (e.g. a GPU training run in parallel). Deterministic for a fixed seed.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData, pairwise_distances
from src.data.simulator import SimulatorConfig, build_edge_bank
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.eval.pyvrp_reference import SCALE, _down, _up, demand_units

_UNREACHABLE_FACTOR = 10   # profile-B duration to c = factor * shift end: never in time


@dataclass(frozen=True)
class StaticInstance:
    """Per-instance integer data that does not change during the day (picklable)."""

    xy: np.ndarray          # (N, 2) int64 scaled coordinates
    delivery: np.ndarray    # (N,) int64 demand units
    capacity: int           # vehicle capacity, units
    service: np.ndarray     # (N,) int64 scaled service time (rounded up)
    early: np.ndarray       # (N,) int64 window opening (rounded up)
    late: np.ndarray        # (N,) int64 window closing (rounded down)
    depot_late: int         # shift end (rounded down)
    distance: np.ndarray    # (N, N) int64 scaled distance (objective)
    duration: np.ndarray    # (N, N) int64 scaled planning travel time (rounded up)


def pack_static(
    data: CVRPTWData,
    capacity_units: int,
    durations: Tensor | None = None,   # (B, N, N) planning travel times; None = distance
) -> list[StaticInstance]:
    """Integer per-instance data for the subproblems. One StaticInstance per instance."""
    b, n, _ = data.coords.shape
    coords = data.coords.double().cpu().numpy()                          # (B, N, 2)
    dist = pairwise_distances(data.coords.double()).cpu().numpy()       # (B, N, N)
    dur = dist if durations is None else durations.double().cpu().numpy()
    if dur.shape != (b, n, n):
        raise ValueError(f"durations must be {(b, n, n)}, got {dur.shape}")
    if (dur < dist - 1e-9).any():
        raise ValueError("planning durations below the distance would assume "
                         "faster-than-mean travel")
    tw = data.time_windows.double().cpu().numpy()                         # (B, N, 2)
    svc = data.service_time.double().cpu().numpy()                       # (B, N)
    units = demand_units(data, capacity_units)                           # (B, N)
    depot_late = int(_down(np.array(data.shift_end)))
    return [
        StaticInstance(
            xy=np.rint(coords[i] * SCALE).astype(np.int64),
            delivery=units[i], capacity=int(capacity_units),
            service=_up(svc[i]), early=_up(tw[i, :, 0]), late=_down(tw[i, :, 1]),
            depot_late=depot_late,
            distance=np.rint(dist[i] * SCALE).astype(np.int64),
            duration=_up(dur[i]),
        )
        for i in range(b)
    ]


@dataclass(frozen=True)
class Decision:
    """One subproblem's answer."""

    next_node: int      # env action: a customer index, or 0 = return to / stay at depot
    feasible: bool      # PyVRP's verdict on the whole re-plan
    iterations: int


def solve_next(
    inst: StaticInstance,
    current: int,                 # env node index of the vehicle
    tau: float,                   # env clock (ready to depart `current`)
    remaining_units: int,         # remaining capacity of the current vehicle, units
    unvisited: Sequence[int],     # env indices of unvisited customers (non-empty)
    max_iterations: int,
    seed: int = 0,
) -> Decision:
    """Re-plan the rest of the day with PyVRP and return its next move. Module-level
    so it pickles for ProcessPoolExecutor (Windows spawns worker processes)."""
    from pyvrp import Model                       # local: spawned workers import it
    from pyvrp.stop import MaxIterations

    if len(unvisited) == 0:
        return Decision(next_node=0, feasible=True, iterations=0)
    at_depot = current == 0
    tau_int = int(_up(np.array(tau)))
    m = Model()

    # Locations: 0 = depot, then one per unvisited customer, then (if en route) the
    # vehicle's current position. `node_of[k]` maps model location k -> env node index.
    node_of: list[int] = [0] + [int(c) for c in unvisited]
    if not at_depot:
        node_of.append(int(current))
    locs = [m.add_location(x=int(inst.xy[v, 0]), y=int(inst.xy[v, 1])) for v in node_of]
    cur_loc = len(node_of) - 1                    # only meaningful when en route

    # Already past the shift end (a late day): PyVRP requires every vehicle's window to
    # overlap its end depot's, so the depot window is widened to tau. The fresh vehicles
    # keep tw_late = shift end; the current vehicle's late return is time-warped
    # (penalised) by PyVRP and charged by the env.
    late_bound = max(inst.depot_late, tau_int)
    depot = m.add_depot(locs[0], tw_early=0, tw_late=late_bound)
    prof_a = m.add_profile(name="current")
    prof_b = m.add_profile(name="fresh") if not at_depot else prof_a

    # Fresh vehicles from the depot (one per unvisited customer is always enough).
    m.add_vehicle_type(
        num_available=len(unvisited), capacity=inst.capacity, start_depot=depot,
        end_depot=depot, tw_early=0, tw_late=inst.depot_late, profile=prof_b,
        name="fresh",
    )
    client_node: list[int] = []                   # client order -> env node index
    for k, v in enumerate(node_of[1:1 + len(unvisited)], start=1):
        m.add_client(
            locs[k], delivery=int(inst.delivery[v]),
            service_duration=int(inst.service[v]),
            tw_early=int(inst.early[v]), tw_late=int(inst.late[v]), name=f"n{v}",
        )
        client_node.append(v)

    if not at_depot:
        start = m.add_depot(locs[cur_loc], tw_early=0, tw_late=late_bound)
        m.add_vehicle_type(
            num_available=1, capacity=max(int(remaining_units), 0), start_depot=start,
            end_depot=depot, tw_early=tau_int, start_late=tau_int,
            tw_late=late_bound, profile=prof_a, name="current",
        )
        m.add_client(locs[cur_loc], delivery=0, service_duration=0,
                     tw_early=tau_int, tw_late=tau_int, name="dummy")
        client_node.append(-1)                    # the dummy

    big = _UNREACHABLE_FACTOR * max(late_bound, 1)
    for a in range(len(node_of)):
        for c in range(len(node_of)):
            va, vc = node_of[a], node_of[c]
            dist = int(inst.distance[va, vc])
            dur = int(inst.duration[va, vc])
            m.add_edge(locs[a], locs[c], distance=dist, duration=dur, profile=prof_a)
            if not at_depot:
                touches_cur = (a == cur_loc) != (c == cur_loc)
                m.add_edge(locs[a], locs[c], distance=dist,
                           duration=big if touches_cur else dur, profile=prof_b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # infeasibility is RECORDED below instead
        res = m.solve(stop=MaxIterations(max_iterations), seed=seed, display=False)
    sol = res.best
    def client_seq(route) -> list[int]:
        # A client activity's idx is its position among CLIENTS (checked on PyVRP 0.14:
        # depots and clients are numbered separately in a route's activities).
        return [client_node[int(act.idx)] for act in route if act.is_client()]

    routes = [(int(r.vehicle_type()), client_seq(r)) for r in sol.routes()]
    if at_depot:
        firsts = [seq[0] for _, seq in routes if seq]
        nxt = firsts[0] if firsts else 0
    else:
        cur_type = 1                               # vehicle types in insertion order
        mine = [seq for vt, seq in routes if vt == cur_type]
        seq = [v for v in (mine[0] if mine else []) if v != -1]
        if mine and -1 not in mine[0]:
            raise RuntimeError("the dummy is not on the current vehicle's route")
        nxt = seq[0] if seq else 0
    if nxt != 0 and nxt not in unvisited:
        raise RuntimeError(f"PyVRP proposed node {nxt}, which is not unvisited")
    return Decision(next_node=int(nxt), feasible=bool(sol.is_feasible()),
                    iterations=int(res.num_iterations))


def _solve_star(job: tuple) -> Decision:
    return solve_next(*job)


@torch.no_grad()
def rolling_horizon(
    data: CVRPTWData,
    static: Sequence[StaticInstance],
    sim_cfg: SimulatorConfig,
    tardiness_weight: float,
    days: Sequence[Tensor],          # each (B, N, N, K), drawn by draw_test_days
    max_iterations: int,
    workers: int = 1,
    seed: int = 0,
    log_every: int = 0,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Drive every instance with PyVRP re-planning at each step, on each day.

    Returns (components averaged over days: key -> (B,) on CPU, stats). Components are
    CVRPTWStochEnv.cost_components: distance, late_count, tardiness, penalty, total.
    """
    if len(days) == 0:
        raise ValueError("need at least one day")
    b, n, _ = data.coords.shape
    if len(static) != b:
        raise ValueError("one StaticInstance per instance")
    bank = build_edge_bank(data.coords, sim_cfg)                         # (B, N, N, K, 2)
    cap_units = static[0].capacity
    totals: dict[str, Tensor] = {}
    n_solves, n_infeasible, iters = 0, 0, 0
    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for d_idx, noise in enumerate(days):
            env = CVRPTWStochEnv(
                data.coords, data.demand, data.capacity, data.time_windows,
                data.service_time, data.shift_end, bank=bank, sim_cfg=sim_cfg,
                noise=noise, tardiness_weight=tardiness_weight, reward_mode="sampled",
            )
            env.reset()
            actions: list[Tensor] = []
            for step in range(4 * n):                                    # hard cap
                visited = env.visited.cpu()                              # (B, N)
                done = visited[:, 1:].all(dim=1)                         # (B,)
                if bool(done.all()):
                    break
                cur = env.current_node.cpu()                             # (B,)
                tau = env.time.double().cpu()                            # (B,)
                rem = env.remaining_capacity.double().cpu() / env.capacity.double().cpu()
                rem_units = torch.round(rem * cap_units).long()          # (B,)
                jobs, rows = [], []
                for i in range(b):
                    if bool(done[i]):
                        continue
                    unv = [int(j) for j in range(1, n) if not bool(visited[i, j])]
                    jobs.append((static[i], int(cur[i]), float(tau[i]), int(rem_units[i]),
                                 unv, max_iterations, seed))
                    rows.append(i)
                decisions = (list(pool.map(_solve_star, jobs, chunksize=4)) if pool
                             else [_solve_star(j) for j in jobs])
                act = torch.zeros(b, dtype=torch.long)                   # done rows: depot
                for i, dec in zip(rows, decisions):
                    act[i] = dec.next_node
                    n_infeasible += int(not dec.feasible)
                    iters += dec.iterations
                n_solves += len(jobs)
                # A vehicle AT the depot asked to "return" would loop forever: guard.
                stuck = (act == 0) & (cur == 0) & ~done
                if bool(stuck.any()):
                    raise RuntimeError(f"PyVRP returned no move at the depot for rows "
                                       f"{stuck.nonzero().flatten().tolist()}")
                act_dev = act.to(data.coords.device)
                env.step(act_dev)
                actions.append(act_dev)
                if log_every and step % log_every == 0:
                    print(f"    day {d_idx} step {step}: {int((~done).sum())} instances "
                          f"active, {n_solves} solves, {time.perf_counter() - t0:.0f} s",
                          flush=True)
            else:
                raise RuntimeError("rolling horizon did not finish within 4N steps")
            seq = torch.stack(actions + [torch.zeros_like(actions[0])], dim=1)  # (B, T)
            if not bool(env.visited[:, 1:].all()):
                raise RuntimeError("a rolling-horizon run left customers unserved")
            if not bool((env.current_node == 0).all()):
                env.step(torch.zeros_like(env.current_node))            # close last route
            parts = env.cost_components(seq)
            for k, v in parts.items():
                v = v.detach().double().cpu()
                totals[k] = v.clone() if k not in totals else totals[k] + v
    finally:
        if pool is not None:
            pool.shutdown()
    comps = {k: v / len(days) for k, v in totals.items()}
    stats = {
        "solves": float(n_solves),
        "infeasible_solves": float(n_infeasible),
        "mean_iterations": iters / max(n_solves, 1),
        "seconds": time.perf_counter() - t0,
        "seconds_per_instance_day": (time.perf_counter() - t0) / (b * len(days)),
    }
    return comps, stats


__all__ = ["StaticInstance", "Decision", "pack_static", "solve_next", "rolling_horizon"]