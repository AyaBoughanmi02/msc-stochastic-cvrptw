"""Tests for --days-per-instance (E1): K independent days per instance at equal compute."""

from __future__ import annotations

import pytest
import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.train.day_replication import replicate_days
from src.train.pomo_stoch import build_stoch_env

SIM = SimulatorConfig(alpha=0.3, beta_var=0.3, eta=0.6, omega=1.0)


def test_k1_returns_the_same_batch() -> None:
    data = generate_cvrptw(8, 10, 20, seed=0)
    assert replicate_days(data, 1) is data


def test_bad_k_raises() -> None:
    data = generate_cvrptw(2, 10, 20, seed=0)
    with pytest.raises(ValueError):
        replicate_days(data, 0)


def test_copies_are_contiguous_and_identical() -> None:
    base = generate_cvrptw(8, 10, 20, seed=1)
    k = 4
    rep = replicate_days(base, k)
    assert rep.coords.shape == (8 * k, 11, 2)
    assert rep.capacity == base.capacity and rep.shift_end == base.shift_end
    for i in range(8):
        rows = slice(i * k, (i + 1) * k)
        for field in ("coords", "demand", "time_windows", "service_time"):
            block = getattr(rep, field)[rows]                       # (K, ...)
            ref = getattr(base, field)[i].expand_as(block)
            assert torch.equal(block, ref), field


def test_each_copy_gets_its_own_day() -> None:
    """Same instance -> same Edge Bank; different rows -> independent noise (days)."""
    base = generate_cvrptw(2, 10, 20, seed=2)
    rep = replicate_days(base, 4)
    env = build_stoch_env(rep, SIM, n_starts=10, tardiness_weight=2.0,
                          generator=torch.Generator().manual_seed(0),
                          reward_mode="sampled")
    assert torch.equal(env.bank[0], env.bank[3])                     # same instance
    assert not torch.equal(env.bank[0], env.bank[4])                 # next instance
    assert not torch.equal(env.noise[0], env.noise[1])               # different days
