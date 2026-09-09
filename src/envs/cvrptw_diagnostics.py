"""CVRPTW calibration diagnostics.

Answers the calibration question the variance gate depends on: are the generated
instances genuinely TIME-BINDING, or capacity problems wearing a CVRPTW costume?
If time never actually removes an option, the policy can't learn window-aware
routing and the gate has nothing to manage.

We roll out generated instances with a greedy-feasible policy (pick the nearest
feasible node) and, at every decision, decompose why each not-yet-served customer
is infeasible: capacity vs time (window-closed or shift-exceeded). The headline is
`time_only_rate` -- customers the vehicle could still carry but cannot reach in
time. A meaningful value means time is a real constraint; near zero means loosen
the dials (`tw_half_width`, `shift_end`).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.envs.cvrptw_env import CVRPTWEnv

_EPS = 1e-6


class InfeasibilityReasons(NamedTuple):
    """Per-node boolean reasons at a given state (depot columns not meaningful)."""

    visited: Tensor        # (B, N)
    over_capacity: Tensor  # (B, N)
    window_closed: Tensor  # (B, N)  condition (a)
    cant_return: Tensor    # (B, N)  condition (b)


def decompose_infeasibility(env: CVRPTWEnv) -> InfeasibilityReasons:
    """Reason each node is (in)feasible at the env's current state.

    The OR of these four over the CUSTOMER columns equals the real mask's customer
    columns (verified in tests), so the diagnostic can't silently diverge from the
    mask it is explaining.
    """
    # Read through the env's travel-time seam, NOT env.dist: under the stochastic env
    # the mask is built from Edge-Bank means at the current time bin, and a diagnostic
    # that explained it with Euclidean distance would silently disagree with it.
    travel_from_current = env._travel_from_current()           # (B, N) planned
    travel_to_depot = env._travel_to_depot()                   # (B, N) planned
    earliest = env.time_windows[..., 0]
    latest = env.time_windows[..., 1]
    arrival = env.time.unsqueeze(1) + travel_from_current
    serve_start = torch.maximum(arrival, earliest)
    depart = serve_start + env.service_time

    visited = env.visited.clone()
    over_capacity = env.demand > env.remaining_capacity.unsqueeze(1) + _EPS
    window_closed = arrival > latest + _EPS
    cant_return = depart + travel_to_depot > env.shift_end + _EPS
    return InfeasibilityReasons(visited, over_capacity, window_closed, cant_return)


def greedy_feasible_rollout(env: CVRPTWEnv, max_steps: int | None = None) -> Tensor:
    """Roll out picking the nearest feasible node each step. Returns (B, T) actions."""
    mask = env.reset()
    n = env.num_nodes
    max_steps = 3 * n if max_steps is None else max_steps
    actions: list[Tensor] = []
    for _ in range(max_steps):
        # Nearest in PLANNED TRAVEL TIME (== distance for variant A, Edge-Bank mu at
        # the current time bin for the stochastic env).
        d = env._travel_from_current().masked_fill(mask, float("inf"))
        action = d.argmin(dim=1)
        actions.append(action)
        mask, done = env.step(action)
        if bool(done.all()):
            break
    return torch.stack(actions, dim=1)


def calibration_stats(data: CVRPTWData, max_steps: int | None = None) -> dict[str, float]:
    """Roll out `data` greedily and return time-vs-capacity binding statistics."""
    env = CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    mask = env.reset()
    b, n = env.batch_size, env.num_nodes
    max_steps = 3 * n if max_steps is None else max_steps
    cust = slice(1, None)

    tot_uv = time_only = cap_only = both = 0
    steps_time_pressure = 0            # (instance, step) slots with >=1 time-only block
    decisions = 0
    served_since_depot = torch.zeros(b, device=env.coords.device)
    routes = torch.zeros(b, device=env.coords.device)
    utils: list[float] = []

    for _ in range(max_steps):
        r = decompose_infeasibility(env)
        uv = ~r.visited[:, cust]                                   # unvisited customers
        t_block = (r.window_closed | r.cant_return)[:, cust] & uv  # time-blocked
        c_block = r.over_capacity[:, cust] & uv                    # capacity-blocked
        t_only = t_block & ~c_block
        tot_uv += int(uv.sum())
        time_only += int(t_only.sum())
        cap_only += int((c_block & ~t_block).sum())
        both += int((t_block & c_block).sum())
        steps_time_pressure += int(t_only.any(dim=1).sum())        # per (instance) this step
        decisions += b

        # record clock utilization at genuine depot returns (served >= 1 since depot)
        travel_row = env._travel_from_current()                           # (B, N) planned
        d = travel_row.masked_fill(mask, float("inf"))
        action = d.argmin(dim=1)
        is_depot = action == 0
        # PLANNED arrival: never call env._travel_time() in a diagnostic -- that draws a
        # sample and would consume RNG for a step we are not actually taking.
        arrival_depot = env.time + travel_row.gather(1, action.view(-1, 1)).squeeze(1)
        real_return = is_depot & (served_since_depot > 0)
        if bool(real_return.any()):
            routes += real_return.float()
            utils.extend((arrival_depot[real_return] / env.shift_end).tolist())
            served_since_depot = torch.where(
                real_return, torch.zeros(b, device=env.coords.device), served_since_depot
            )
        served_since_depot += (~is_depot).float()

        mask, done = env.step(action)
        if bool(done.all()):
            break

    # final routes (trajectories that ended mid-route with customers served)
    routes += (served_since_depot > 0).float()
    denom = max(tot_uv, 1)
    return {
        "time_only_rate": time_only / denom,
        "capacity_only_rate": cap_only / denom,
        "both_rate": both / denom,
        "feasible_rate": 1.0 - (time_only + cap_only + both) / denom,
        "steps_with_time_pressure_rate": steps_time_pressure / max(decisions, 1),
        "avg_routes_per_instance": float(routes.mean()),
        "avg_customers_per_route": (n - 1) / max(float(routes.mean()), 1e-9),
        "avg_route_time_utilization": float(sum(utils) / len(utils)) if utils else 0.0,
    }


@torch.no_grad()
def policy_binding_stats(policy, data: CVRPTWData) -> dict[str, float]:
    """Time-vs-capacity binding breakdown under a TRAINED policy's own routing.

    Rolls out `policy` greedily and, at each decision, decomposes why each unserved
    customer is infeasible. Answers "does the trained policy actually face time
    constraints?" (Level 2), complementing calibration_stats' instance-level view
    (Level 1). `time_only_rate` = fraction of unserved-customer slots blocked by
    time but not capacity.
    """
    from src.data.graph_build import build_graph_tw
    from src.envs.cvrptw_env import CVRPTWEnv

    policy.eval()
    env = CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    b, n = env.batch_size, env.num_nodes
    enc = build_graph_tw(env.coords, env.demand, env.time_windows, env.service_time)
    node_flat, global_emb = policy.encoder(enc.x, enc.edge_index, enc.edge_attr, enc.batch)
    node_emb = node_flat.view(b, n, -1)
    cache = policy.decoder.precompute(node_emb)

    mask = env.reset()
    idx = torch.arange(b, device=env.coords.device)
    cust = slice(1, None)
    tot_uv = time_only = cap_only = both = 0
    for _ in range(3 * n):
        r = decompose_infeasibility(env)
        uv = ~r.visited[:, cust]
        t_block = (r.window_closed | r.cant_return)[:, cust] & uv
        c_block = r.over_capacity[:, cust] & uv
        tot_uv += int(uv.sum())
        time_only += int((t_block & ~c_block).sum())
        cap_only += int((c_block & ~t_block).sum())
        both += int((t_block & c_block).sum())

        last = node_emb[idx, env.current_node]
        log_p = policy.decoder(
            cache, global_emb, last, env.normalized_remaining(), mask,
            context_extra=env.normalized_time(),
        )
        mask, done = env.step(log_p.argmax(dim=-1))
        if bool(done.all()):
            break

    d = max(tot_uv, 1)
    return {
        "time_only_rate": time_only / d,
        "capacity_only_rate": cap_only / d,
        "both_rate": both / d,
    }