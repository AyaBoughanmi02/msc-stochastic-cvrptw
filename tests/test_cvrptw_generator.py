"""Tests for the CVRPTW generator.

The load-bearing test is feasibility-by-construction: every generated customer
must be individually servable from a fresh shift (arrive within its window, serve,
and return before the shift ends). It is recomputed here independently of the
generator's own arithmetic.
"""

from __future__ import annotations

import math

import torch

from src.data.cvrptw_generator import (
    _MAX_RADIUS,
    generate_cvrptw,
    pairwise_distances,
)


def test_shapes_and_depot_conventions() -> None:
    data = generate_cvrptw(4, 20, seed=1)
    b, n = 4, 21
    assert data.coords.shape == (b, n, 2)
    assert data.demand.shape == (b, n)
    assert data.time_windows.shape == (b, n, 2)
    assert data.service_time.shape == (b, n)
    assert data.capacity == 1.0
    # depot: window [0, shift_end], zero service, zero demand
    assert torch.allclose(data.time_windows[:, 0, 0], torch.zeros(b))
    assert torch.allclose(data.time_windows[:, 0, 1], torch.full((b,), data.shift_end))
    assert torch.allclose(data.service_time[:, 0], torch.zeros(b))
    assert torch.allclose(data.demand[:, 0], torch.zeros(b))


def test_windows_are_valid() -> None:
    data = generate_cvrptw(8, 20, seed=2)
    earliest, latest = data.time_windows[..., 0], data.time_windows[..., 1]
    assert (earliest >= 0.0).all()
    assert (latest >= earliest).all()


def test_pairwise_distances_symmetric_zero_diagonal() -> None:
    coords = torch.rand(3, 6, 2)
    d = pairwise_distances(coords)
    assert d.shape == (3, 6, 6)
    assert torch.allclose(d, d.transpose(1, 2), atol=1e-6)          # symmetric
    assert torch.allclose(d.diagonal(dim1=1, dim2=2), torch.zeros(3, 6), atol=1e-6)


def test_feasible_by_construction() -> None:
    # Every customer must be servable from a fresh shift, for several settings.
    for shift_end, w, svc in [(4.0, 0.5, 0.2), (3.5, 0.1, 0.2), (5.0, 1.5, 0.3)]:
        data = generate_cvrptw(
            16, 20, seed=3, shift_end=shift_end, tw_half_width=w, service_time=svc
        )
        r = pairwise_distances(data.coords)[:, 0, :]               # (B, N) depot radius
        earliest = data.time_windows[..., 0]
        latest = data.time_windows[..., 1]
        service = data.service_time
        arrival = r                                                # leave depot at clock 0
        serve_start = torch.maximum(arrival, earliest)             # wait if early
        depart = serve_start + service
        # (a) can arrive within the window; (b) can serve and return within shift
        arrives_in_time = arrival <= latest + 1e-6
        returns_in_time = depart + r <= data.shift_end + 1e-6
        cust = slice(1, None)                                      # skip depot
        assert arrives_in_time[:, cust].all(), "a customer's window is unreachable"
        assert returns_in_time[:, cust].all(), "a customer can't be served and returned"


def test_reproducible_with_seed() -> None:
    a = generate_cvrptw(4, 20, seed=7)
    b = generate_cvrptw(4, 20, seed=7)
    assert torch.equal(a.coords, b.coords)
    assert torch.equal(a.demand, b.demand)
    assert torch.equal(a.time_windows, b.time_windows)
    assert torch.equal(a.service_time, b.service_time)


def test_reachability_floor_rejected() -> None:
    floor = 2.0 * _MAX_RADIUS + 0.2
    try:
        generate_cvrptw(2, 20, seed=1, shift_end=floor - 0.1, service_time=0.2)
    except ValueError:
        return
    raise AssertionError("shift_end below the reachability floor should raise")