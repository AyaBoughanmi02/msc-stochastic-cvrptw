"""Tests for the CVRPTW environment.

Two levels:
  1. A fully hand-computed two-route scenario (the reading_notes example): time
     windows force serving A on one route and B on another, with a clock reset in
     between and a wait on the second route.
  2. An integration test: roll out many GENERATED instances with a greedy-feasible
     policy and check the feasibility invariant holds (never an all-masked state),
     the rollout terminates, and every produced tour independently satisfies all
     windows and the shift.
"""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw, pairwise_distances
from src.envs.cvrptw_env import CVRPTWEnv

# depot (0,0), A (1,0), B (0,1): depot<->A = depot<->B = 1.0, A<->B = sqrt(2).
_COORDS = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])   # (1, 3, 2)


def _two_route_env() -> CVRPTWEnv:
    demand = torch.tensor([[0.0, 0.3, 0.3]])                     # fit one trip (capacity not binding)
    tw = torch.tensor([[[0.0, 4.0], [1.0, 1.5], [1.0, 2.0]]])    # A [1,1.5], B [1,2]
    service = torch.tensor([[0.0, 0.2, 0.2]])
    return CVRPTWEnv(_COORDS.clone(), demand, 1.0, tw, service, shift_end=4.0)


def test_two_route_scenario_end_to_end() -> None:
    env = _two_route_env()

    mask = env.reset()
    assert mask.tolist() == [[True, False, False]]              # at depot: A,B both open
    assert torch.allclose(env.time, torch.tensor([0.0]))

    # Serve A: arrive 1.0, serve_start 1.0, clock -> 1.2. Now B's window (<=2.0) is
    # unreachable: arrival 1.2 + sqrt(2) = 2.614 > 2.0 -> only the depot is feasible.
    mask, done = env.step(torch.tensor([1]))
    assert mask.tolist() == [[False, True, True]]
    assert done.tolist() == [False]
    assert torch.allclose(env.time, torch.tensor([1.2]), atol=1e-5)

    # Return to depot: clock resets to 0, capacity refills, B open again.
    mask, done = env.step(torch.tensor([0]))
    assert mask.tolist() == [[True, True, False]]
    assert done.tolist() == [False]
    assert torch.allclose(env.time, torch.tensor([0.0]))

    # Serve B on the fresh route: all customers served -> done.
    mask, done = env.step(torch.tensor([2]))
    assert done.tolist() == [True]
    assert torch.allclose(env.time, torch.tensor([1.2]), atol=1e-5)

    # Cost is plain distance: depot->A->depot->B->depot = 1+1+1+1 = 4.0.
    cost = env.cost(torch.tensor([[1, 0, 2]]))
    assert torch.allclose(cost, torch.tensor([4.0]), atol=1e-5)


def _assert_tours_feasible(data, dist, actions) -> None:
    """Independently re-simulate each tour and assert all windows + shift hold."""
    b_size, n = data.coords.shape[0], data.coords.shape[1]
    earliest = data.time_windows[..., 0]
    latest = data.time_windows[..., 1]
    service = data.service_time
    shift = data.shift_end
    eps = 1e-4
    for b in range(b_size):
        clock, current = 0.0, 0
        seq = actions[b].tolist()
        for a in seq:
            travel = dist[b, current, a].item()
            arrival = clock + travel
            if a == 0:                                          # depot return
                assert arrival <= shift + eps, f"shift violated on return (inst {b})"
                clock, current = 0.0, 0
                continue
            assert arrival <= latest[b, a].item() + eps, f"window violated (inst {b}, node {a})"
            serve_start = max(arrival, earliest[b, a].item())
            clock = serve_start + service[b, a].item()
            current = a
        final_return = clock + dist[b, current, 0].item()       # tour_length adds this leg
        assert final_return <= shift + eps, f"shift violated on final return (inst {b})"
        served = {x for x in seq if x != 0}
        assert served == set(range(1, n)), f"not all customers served (inst {b})"


def test_generated_instances_roll_out_feasibly() -> None:
    data = generate_cvrptw(64, 20, seed=5, shift_end=4.0, tw_half_width=0.5)
    env = CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    mask = env.reset()
    b_size, n = env.batch_size, env.num_nodes
    idx = torch.arange(b_size)
    actions: list[torch.Tensor] = []
    done = torch.zeros(b_size, dtype=torch.bool)
    for _ in range(3 * n):
        # Feasibility invariant: at least one node feasible everywhere, always.
        assert bool((~mask).any(dim=1).all()), "an all-masked (dead-end) state occurred"
        travel_from_current = env.dist[idx, env.current_node]   # (B, N)
        d = travel_from_current.masked_fill(mask, float("inf"))
        action = d.argmin(dim=1)                                # nearest feasible node
        actions.append(action)
        mask, done = env.step(action)
        if bool(done.all()):
            break
    assert bool(done.all()), "greedy-feasible rollout did not terminate"
    acts = torch.stack(actions, dim=1)                          # (B, T)
    _assert_tours_feasible(data, env.dist, acts)


def test_normalized_time_in_unit_range() -> None:
    env = _two_route_env()
    env.reset()
    env.step(torch.tensor([1]))
    nt = env.normalized_time()
    assert nt.shape == (1, 1)
    assert (nt >= 0.0).all() and (nt <= 1.0).all()