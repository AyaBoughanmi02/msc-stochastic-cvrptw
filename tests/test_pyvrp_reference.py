"""Tests for src/eval/pyvrp_reference.py (the PyVRP classical reference, X0 / X0b).

Skipped automatically if PyVRP is not installed. Run:
    python -m pytest tests/test_pyvrp_reference.py -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

pytest.importorskip("pyvrp")

from src.data.cvrptw_generator import generate_cvrptw, pairwise_distances  # noqa: E402
from src.data.simulator import SimulatorConfig, draw_noise                 # noqa: E402
from src.eval.pyvrp_reference import (                                     # noqa: E402
    PlanResult,
    buffered_durations,
    demand_units,
    draw_test_days,
    execute_actions,
    pack_instances,
    routes_to_actions,
    solve_batch,
)
from src.utils.masking import tour_length                                  # noqa: E402

CAP = 20
SIM = SimulatorConfig(alpha=0.3, beta_var=0.3, eta=0.6, omega=1.0)      # trained setting
SIM_DET = replace(SIM, alpha=0.0, beta_var=0.0)                           # travel == distance


@pytest.fixture(scope="module")
def small():
    """8 instances, 10 customers, solved once for the whole module (0.2 s each)."""
    data = generate_cvrptw(8, 10, CAP, seed=123)
    plans = solve_batch(pack_instances(data, CAP), time_limit=0.2, seed=0)
    return data, plans


def test_demand_units_are_integers_in_range():
    data = generate_cvrptw(4, 10, CAP, seed=1)
    units = demand_units(data, CAP)
    assert units[:, 0].tolist() == [0, 0, 0, 0]
    assert (units[:, 1:] >= 1).all() and (units[:, 1:] <= CAP).all()


def test_demand_units_rejects_wrong_capacity():
    data = generate_cvrptw(4, 10, CAP, seed=1)
    with pytest.raises(ValueError):
        demand_units(data, 7)          # 1/7 does not divide the k/20 fractions


def test_plans_are_complete_and_feasible(small):
    _, plans = small
    assert all(p.complete for p in plans)
    assert all(p.feasible for p in plans)


def test_feasible_plan_has_zero_penalty_in_float_env(small):
    """Conservative integer rounding: PyVRP-feasible => feasible in the float env."""
    data, plans = small
    actions = routes_to_actions(plans, data.coords.size(1), data.coords.device)
    day = draw_test_days(8, 11, SIM_DET, [0], data.coords.device, data.coords.dtype)
    parts = execute_actions(data, actions, SIM_DET, tardiness_weight=2.0, days=day)
    assert float(parts["penalty"].abs().max()) < 1e-5
    # The env's float distance agrees with PyVRP's rounded objective (<= 0.5/SCALE per arc).
    pyvrp_d = torch.tensor([p.pyvrp_distance for p in plans], dtype=parts["distance"].dtype)
    assert torch.allclose(parts["distance"], pyvrp_d, atol=2e-3)
    assert torch.allclose(parts["distance"], tour_length(data.coords, actions))


def test_stochastic_execution_charges_lateness_but_keeps_distance(small):
    """Same fixed plan on a noisy day: distance unchanged, penalty >= 0."""
    data, plans = small
    actions = routes_to_actions(plans, 11, data.coords.device)
    sim = replace(SIM, beta_var=0.9)
    days = draw_test_days(8, 11, sim, [0, 1], data.coords.device, data.coords.dtype)
    parts = execute_actions(data, actions, sim, tardiness_weight=2.0, days=days)
    assert torch.allclose(parts["distance"], tour_length(data.coords, actions))
    assert bool((parts["penalty"] >= 0).all())
    assert torch.allclose(parts["total"], parts["distance"] + parts["penalty"])


def test_routes_to_actions_rejects_incomplete():
    bad = PlanResult(routes=((1, 2),), feasible=True, complete=False, runtime=0.0,
                     pyvrp_distance=0.0)
    with pytest.raises(ValueError):
        routes_to_actions([bad], 4, torch.device("cpu"))


def test_routes_to_actions_layout():
    p = PlanResult(routes=((2, 1), (3,)), feasible=True, complete=True, runtime=0.0,
                   pyvrp_distance=0.0)
    q = PlanResult(routes=((1, 2, 3),), feasible=True, complete=True, runtime=0.0,
                   pyvrp_distance=0.0)
    a = routes_to_actions([p, q], 4, torch.device("cpu"))
    assert a.tolist() == [[2, 1, 0, 3, 0], [1, 2, 3, 0, 0]]


def test_days_match_the_apriori_draw():
    """Same seeds, shape and device as evaluate_stoch_apriori -> identical days."""
    dev = torch.device("cpu")
    days = draw_test_days(3, 5, SIM, [0, 7], dev, torch.float32)
    for seed, day in zip([0, 7], days):
        ref = draw_noise((3, 5, 5, SIM.n_bins), device=dev, dtype=torch.float32,
                         generator=torch.Generator(device=dev).manual_seed(seed))
        assert torch.equal(day, ref)


def test_buffer_k0_is_distance_and_monotone_in_k():
    data = generate_cvrptw(2, 6, CAP, seed=5)
    d = pairwise_distances(data.coords)
    assert torch.equal(buffered_durations(data, SIM, 0.0), d)
    b1 = buffered_durations(data, SIM, 1.0)
    b2 = buffered_durations(data, SIM, 2.0)
    assert bool((b1 >= d).all()) and bool((b2 >= b1).all())
    assert bool((b1 > d)[:, 1:, 1:].any())      # the buffer actually bites off-diagonal


def test_buffered_plan_is_more_conservative_on_average():
    """A buffered plan is never shorter than the unbuffered optimum (it plans on slower roads)."""
    data = generate_cvrptw(6, 10, CAP, seed=9)
    base = solve_batch(pack_instances(data, CAP), 0.2)
    buff = solve_batch(pack_instances(data, CAP, buffered_durations(data, SIM, 1.0)), 0.2)
    ok = [(b, u) for b, u in zip(base, buff) if b.feasible and u.feasible]
    assert ok, "no instance stayed feasible under the buffer"
    mean_base = sum(b.pyvrp_distance for b, _ in ok) / len(ok)
    mean_buff = sum(u.pyvrp_distance for _, u in ok) / len(ok)
    assert mean_buff >= mean_base - 1e-3


def test_pack_rejects_durations_below_distance():
    data = generate_cvrptw(2, 5, CAP, seed=3)
    with pytest.raises(ValueError):
        pack_instances(data, CAP, pairwise_distances(data.coords) * 0.5)


def test_fallback_keeps_feasible_and_replaces_infeasible():
    def plan(tag: int, ok: bool) -> PlanResult:
        return PlanResult(routes=((tag,),), feasible=ok, complete=True, runtime=0.0,
                          pyvrp_distance=float(tag))
    from src.eval.pyvrp_reference import fallback_to_feasible
    k0 = [plan(1, True), plan(2, True)]
    k1 = [plan(3, True), plan(4, False)]
    k2 = [plan(5, False), plan(6, False)]
    chosen, fb = fallback_to_feasible([k0, k1, k2])
    assert [p.pyvrp_distance for p in chosen[1]] == [3.0, 2.0]
    assert [p.pyvrp_distance for p in chosen[2]] == [3.0, 2.0]   # chains back through k1
    assert fb == [0, 1, 2]