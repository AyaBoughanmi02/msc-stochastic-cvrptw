"""Tests for the rolling-horizon PyVRP re-planner (src/eval/pyvrp_rolling.py).

Load-bearing claims:
  - every decision is an unvisited customer or the depot, never a visited node;
  - a vehicle with no capacity left goes home (the dummy forces the current vehicle to be
    used, and an empty current route means "return now");
  - with no noise (alpha = beta = 0) the re-planner serves everyone with zero lateness,
    and its cost is within 2% of a frozen PyVRP plan on the same instances.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("pyvrp")

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.pyvrp_reference import (
    draw_test_days,
    execute_actions,
    pack_instances,
    routes_to_actions,
    solve_batch,
)
from src.eval.pyvrp_rolling import pack_static, rolling_horizon, solve_next

_CAP, _SHIFT, _SERVICE, _W = 20, 4.0, 0.2, 2.0


@pytest.fixture(scope="module")
def setup():
    data = generate_cvrptw(4, 10, capacity=_CAP, seed=0, shift_end=_SHIFT,
                           service_time=_SERVICE)
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.6, eta=0.6, omega=1.0, shift_end=_SHIFT)
    return data, cfg, pack_static(data, _CAP)


def test_decision_is_unvisited_or_depot(setup):
    _, _, static = setup
    unvisited = [2, 5, 7]
    dec = solve_next(static[0], current=3, tau=0.8, remaining_units=_CAP,
                     unvisited=unvisited, max_iterations=200)
    assert dec.next_node in unvisited + [0]
    dec0 = solve_next(static[0], current=0, tau=0.0, remaining_units=_CAP,
                      unvisited=unvisited, max_iterations=200)
    assert dec0.next_node in unvisited             # at the depot it must leave


def test_empty_vehicle_returns_home(setup):
    _, _, static = setup
    dec = solve_next(static[0], current=3, tau=0.8, remaining_units=0,
                     unvisited=[2, 5, 7], max_iterations=200)
    assert dec.next_node == 0


def test_deterministic_matches_frozen_plan(setup):
    data, cfg, static = setup
    b, n, _ = data.coords.shape
    det = replace(cfg, alpha=0.0, beta_var=0.0)
    days = draw_test_days(b, n, det, [0], data.coords.device, data.coords.dtype)
    frozen = execute_actions(data, routes_to_actions(
        solve_batch(pack_instances(data, _CAP), 0.2), n, data.coords.device), det, _W, days)
    rh, stats = rolling_horizon(data, static, det, _W, days, max_iterations=300)
    assert float(rh["penalty"].abs().max()) < 1e-5
    assert stats["infeasible_solves"] == 0
    assert float(rh["total"].mean()) <= 1.02 * float(frozen["total"].mean())