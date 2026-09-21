"""Regression test: unseeded generate_cvrptw must return different instances AND windows."""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw


def test_unseeded_cvrptw_batches_differ() -> None:
    torch.manual_seed(0)
    a = generate_cvrptw(16, 20)
    b = generate_cvrptw(16, 20)
    assert not torch.equal(a.coords, b.coords)
    # Same coords would give the same windows only if the window draws were frozen;
    # compare the raw window-centre fraction via a fixed-coords check below instead.
    assert not torch.equal(a.time_windows, b.time_windows)


def test_window_draws_not_frozen() -> None:
    """The window stream itself must vary between unseeded calls."""
    torch.manual_seed(0)
    a = generate_cvrptw(16, 20)
    b = generate_cvrptw(16, 20)
    # Window centre fraction u = (centre - lo) / (hi - lo); frozen stream => identical u.
    def frac(d) -> torch.Tensor:
        r = (d.coords - d.coords[:, :1]).norm(dim=-1)            # (B, N) depot distance
        centre = (d.time_windows[..., 1] - 0.5)                   # latest - half_width
        lo, hi = r, d.shift_end - 0.2 - r
        return ((centre - lo) / (hi - lo))[:, 1:]                 # customers only
    assert not torch.allclose(frac(a), frac(b))


def test_seeded_cvrptw_reproducible() -> None:
    a = generate_cvrptw(8, 20, seed=9999)
    b = generate_cvrptw(8, 20, seed=9999)
    assert torch.equal(a.coords, b.coords)
    assert torch.equal(a.time_windows, b.time_windows)
