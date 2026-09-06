"""Tests for the stochastic, time-dependent CVRPTW environment.

Load-bearing tests:

1. `test_reduces_to_variant_a` -- with alpha = 0 and beta_var = 0 the stochastic env must
   reproduce CVRPTWEnv step for step (masks, clock, cost) BIT-EXACTLY, with zero
   tardiness, even when a noise field is supplied. This is what certifies that the new
   seam did not change the deterministic behaviour.
2. `test_instance_indexed_bank_matches_replicated_bank` -- the memory optimisation (one
   bank row per INSTANCE plus an instance_idx map, rather than one row per trajectory)
   must be numerically identical to the naive replication it replaces.
3. `test_rollouts_always_complete_under_the_locked_config` -- hard masking on mu must
   still leave a feasible node at every step and terminate, with the ripple switched on.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.data.cvrptw_generator import generate_cvrptw, pairwise_distances
from src.data.simulator import (
    MU,
    SIGMA2,
    SimulatorConfig,
    build_edge_bank,
    draw_noise,
    gather_edge,
)
from src.envs.cvrptw_env import CVRPTWEnv
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv

_SHIFT = 4.0
_SERVICE = 0.2
_CAPACITY = 20        # the locked CVRPTW calibration


# --------------------------------------------------------------------------- helpers


def _sample_rollout(env, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out with a uniformly-random FEASIBLE action each step. -> (actions, done)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mask = env.reset()
    done = torch.zeros(env.batch_size, dtype=torch.bool)
    actions: list[torch.Tensor] = []
    for _ in range(2 * env.num_nodes):
        feasible = (~mask).to(torch.float32)                       # (B, N)
        assert bool((feasible.sum(dim=1) > 0).all()), "a trajectory had no feasible node"
        action = torch.multinomial(feasible, 1, generator=gen).squeeze(1)  # (B,)
        actions.append(action)
        mask, done = env.step(action)
        if bool(done.all()):
            break
    return torch.stack(actions, dim=1), done                       # (B, T), (B,)


def _replay(env, actions: torch.Tensor) -> list[torch.Tensor]:
    """Replay a fixed action sequence; return the mask after each step."""
    env.reset()
    masks = []
    for t in range(actions.size(1)):
        mask, _ = env.step(actions[:, t])
        masks.append(mask)
    return masks


def _make_envs(
    cfg: SimulatorConfig,
    *,
    batch: int = 6,
    n_customers: int = 10,
    seed: int = 0,
    noise: torch.Tensor | None = None,
    tardiness_weight: float = 1.0,
) -> tuple[CVRPTWEnv, CVRPTWStochEnv]:
    """A deterministic env and a stochastic env over the SAME instances."""
    data = generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )
    det = CVRPTWEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    bank = build_edge_bank(data.coords, cfg)                       # (B, N, N, K, 2)
    stoch = CVRPTWStochEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
        bank=bank, sim_cfg=cfg, noise=noise, tardiness_weight=tardiness_weight,
    )
    return det, stoch


# ------------------------------------------------------- the reduction to variant A


def test_variant_a_hook_defaults_are_unchanged() -> None:
    """The extracted seam hooks must return exactly the expressions they replaced."""
    det, _ = _make_envs(SimulatorConfig())
    det.reset()
    det.step(torch.full((det.batch_size,), 3, dtype=torch.long))
    idx = torch.arange(det.batch_size)
    assert torch.equal(det._travel_from_current(), det.dist[idx, det.current_node])
    assert torch.equal(det._travel_to_depot(), det.dist[:, :, 0])
    action = torch.full((det.batch_size,), 5, dtype=torch.long)
    assert torch.equal(det._travel_time(action), det.dist[idx, det.current_node, action])
    assert det._on_arrival(action, det.time) is None


@pytest.mark.parametrize("with_noise", [False, True])
def test_reduces_to_variant_a(with_noise: bool) -> None:
    """alpha = 0, beta_var = 0 => identical masks, clock, cost; zero tardiness."""
    cfg = SimulatorConfig(alpha=0.0, beta_var=0.0, shift_end=_SHIFT)
    noise = None
    if with_noise:
        # Noise must be IGNORED when the variance is zero -- deliberately large draws.
        noise = 5.0 * draw_noise(
            (6, 11, 11, cfg.n_bins), device="cpu",
            generator=torch.Generator().manual_seed(4),
        )
    det, stoch = _make_envs(cfg, noise=noise)

    # The bank's planned travel time IS the distance matrix, in every bin.
    assert torch.equal(stoch.bank[..., 0, MU], pairwise_distances(det.coords))

    actions, done = _sample_rollout(det, seed=1)
    assert bool(done.all())
    det_masks = _replay(det, actions)
    stoch_masks = _replay(stoch, actions)

    for t, (a, b) in enumerate(zip(det_masks, stoch_masks)):
        assert torch.equal(a, b), f"masks diverged at step {t}"
    assert torch.equal(det.time, stoch.time)
    assert torch.equal(stoch.tardiness, torch.zeros_like(stoch.tardiness))
    assert torch.equal(stoch.late_count, torch.zeros_like(stoch.late_count))
    assert torch.equal(stoch.cost(actions), det.cost(actions))


# ------------------------------------------------------------- the instance_idx map


def test_instance_indexed_bank_matches_replicated_bank() -> None:
    """One bank row per instance + instance_idx == one bank row per trajectory."""
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.2, eta=0.6, shift_end=_SHIFT)
    n_starts = 3
    data = generate_cvrptw(
        4, 10, capacity=_CAPACITY, seed=21, shift_end=_SHIFT, service_time=_SERVICE
    )
    bank = build_edge_bank(data.coords, cfg)                       # (B, N, N, K, 2)
    noise = draw_noise(
        bank.shape[:-1], device="cpu", generator=torch.Generator().manual_seed(5)
    )                                                              # (B, N, N, K)

    def build(replicated: bool) -> CVRPTWStochEnv:
        rep = lambda t: t.repeat_interleave(n_starts, dim=0)       # noqa: E731
        return CVRPTWStochEnv(
            rep(data.coords), rep(data.demand), data.capacity,
            rep(data.time_windows), rep(data.service_time), data.shift_end,
            bank=rep(bank) if replicated else bank,
            sim_cfg=cfg,
            instance_idx=None if replicated
            else torch.arange(4).repeat_interleave(n_starts),
            noise=rep(noise) if replicated else noise,
        )

    naive, indexed = build(True), build(False)
    actions, done = _sample_rollout(naive, seed=2)
    assert bool(done.all())
    _replay(naive, actions)
    _replay(indexed, actions)

    assert torch.equal(naive.time, indexed.time)
    assert torch.equal(naive.tardiness, indexed.tardiness)
    assert torch.equal(naive.cost(actions), indexed.cost(actions))


def test_common_random_numbers_are_shared_across_starts() -> None:
    """All POMO replicas of one instance must experience the same realised travel times."""
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.5, shift_end=_SHIFT)
    n_starts = 4
    data = generate_cvrptw(
        1, 8, capacity=_CAPACITY, seed=22, shift_end=_SHIFT, service_time=_SERVICE
    )
    bank = build_edge_bank(data.coords, cfg)
    noise = draw_noise(
        bank.shape[:-1], device="cpu", generator=torch.Generator().manual_seed(6)
    )
    rep = lambda t: t.repeat_interleave(n_starts, dim=0)           # noqa: E731

    def build(with_crn: bool) -> CVRPTWStochEnv:
        return CVRPTWStochEnv(
            rep(data.coords), rep(data.demand), data.capacity,
            rep(data.time_windows), rep(data.service_time), data.shift_end,
            bank=bank, sim_cfg=cfg,
            instance_idx=torch.zeros(n_starts, dtype=torch.long),
            noise=noise if with_crn else None,
            generator=torch.Generator().manual_seed(7),
        )

    # Every replica takes the SAME first action. Assert on the sampled TRAVEL TIME, not
    # on the clock: waiting until `earliest` can hide the difference in the clock.
    action = torch.full((n_starts,), 3, dtype=torch.long)

    crn = build(True)
    crn.reset()
    travel_crn = crn._travel_time(action)                          # (S,)
    assert torch.equal(travel_crn, travel_crn[0].expand(n_starts))
    # ... and the shared draw is a real draw, not a silent fallback to the mean.
    mu = gather_edge(
        crn.bank, crn.current_node, action, crn.current_bin(), crn.instance_idx
    )[:, MU]
    assert not torch.allclose(travel_crn, mu)
    crn.step(action)
    assert torch.equal(crn.time, crn.time[0].expand(n_starts))

    fresh = build(False)
    fresh.reset()
    travel_fresh = fresh._travel_time(action)                      # (S,)
    assert not torch.equal(travel_fresh, travel_fresh[0].expand(n_starts))


# ------------------------------------------------------------ tardiness bookkeeping


def test_tardiness_arithmetic_is_exact() -> None:
    """Hand-checked penalty on a synthetic bank (sigma^2 = 0 => travel == mu exactly)."""
    n, k, w = 3, 4, 2.5
    coords = torch.tensor([[[0.0, 0.0], [0.3, 0.0], [0.0, 0.4]]])     # (1, 3, 2)
    demand = torch.tensor([[0.0, 0.1, 0.1]])                          # (1, 3)
    # Windows: customer 1 closes at 0.5, customer 2 at 3.0; depot open all shift.
    time_windows = torch.tensor([[[0.0, _SHIFT], [0.0, 0.5], [0.0, 3.0]]])  # (1, 3, 2)
    service = torch.tensor([[0.0, _SERVICE, _SERVICE]])               # (1, 3)

    bank = torch.zeros(1, n, n, k, 2)                                 # (1, N, N, K, 2)
    bank[..., MU] = 0.9          # every edge takes 0.9 in every bin ...
    bank[:, torch.arange(n), torch.arange(n), :, MU] = 0.0            # ... except self-loops
    assert bank[..., SIGMA2].abs().max() == 0.0                       # deterministic

    cfg = SimulatorConfig(n_bins=k, shift_end=_SHIFT, period=_SHIFT)
    env = CVRPTWStochEnv(
        coords, demand, 1.0, time_windows, service, _SHIFT,
        bank=bank, sim_cfg=cfg, tardiness_weight=w,
    )
    env.reset()
    env.step(torch.tensor([1]))    # depot -> 1: arrives 0.9, latest 0.5 => 0.4 late
    assert env.tardiness.item() == pytest.approx(0.4, abs=1e-6)
    assert env.late_count.item() == 1.0
    # clock = max(0.9, 0.0) + 0.2 = 1.1
    assert env.time.item() == pytest.approx(1.1, abs=1e-6)

    env.step(torch.tensor([2]))    # 1 -> 2: arrives 2.0, latest 3.0 => on time
    assert env.tardiness.item() == pytest.approx(0.4, abs=1e-6)
    assert env.late_count.item() == 1.0

    actions = torch.tensor([[1, 2]])
    parts = env.cost_components(actions)
    expected_distance = 0.3 + math.hypot(0.3, 0.4) + 0.4             # 0 ->1 ->2 ->0
    assert parts["distance"].item() == pytest.approx(expected_distance, abs=1e-6)
    assert parts["penalty"].item() == pytest.approx(w * (1.0 + 0.4), abs=1e-6)
    assert parts["total"].item() == pytest.approx(
        expected_distance + w * 1.4, abs=1e-6
    )
    assert torch.equal(env.cost(actions), parts["total"])


def test_late_depot_return_counts_as_shift_overrun() -> None:
    """The depot's window is [0, shift_end], so a late return is priced by the same rule."""
    n, k = 2, 2
    coords = torch.tensor([[[0.0, 0.0], [0.5, 0.0]]])
    demand = torch.tensor([[0.0, 0.1]])
    time_windows = torch.tensor([[[0.0, _SHIFT], [0.0, _SHIFT]]])
    service = torch.tensor([[0.0, 0.0]])
    bank = torch.zeros(1, n, n, k, 2)
    bank[..., MU] = 3.0
    bank[:, torch.arange(n), torch.arange(n), :, MU] = 0.0
    cfg = SimulatorConfig(n_bins=k, shift_end=_SHIFT, period=_SHIFT)
    env = CVRPTWStochEnv(
        coords, demand, 1.0, time_windows, service, _SHIFT, bank=bank, sim_cfg=cfg,
    )
    env.reset()
    env.step(torch.tensor([1]))                 # clock 3.0, on time
    assert env.late_count.item() == 0.0
    env.step(torch.tensor([0]))                 # arrives 6.0 > shift_end 4.0
    assert env.tardiness.item() == pytest.approx(2.0, abs=1e-6)
    assert env.late_count.item() == 1.0
    assert env.time.item() == 0.0               # clock still resets at the depot


# ------------------------------------------------------------------- integration


def test_rollouts_always_complete_under_the_locked_config() -> None:
    """Ripple on, variance on: masking on mu must still never dead-end, and lateness happens."""
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.15, eta=0.6, shift_end=_SHIFT)
    data = generate_cvrptw(
        32, 20, capacity=_CAPACITY, seed=31, shift_end=_SHIFT, service_time=_SERVICE
    )
    bank = build_edge_bank(data.coords, cfg)
    noise = draw_noise(
        bank.shape[:-1], device="cpu", generator=torch.Generator().manual_seed(8)
    )
    env = CVRPTWStochEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
        bank=bank, sim_cfg=cfg, noise=noise, tardiness_weight=1.0,
    )
    actions, done = _sample_rollout(env, seed=3)   # asserts a feasible node every step
    assert bool(done.all()), "a rollout failed to serve every customer"

    parts = env.cost_components(actions)
    assert (parts["tardiness"] >= 0.0).all()
    assert torch.allclose(parts["total"], parts["distance"] + parts["penalty"])
    # The uncertainty must actually bite, or the penalty term is untested.
    assert float(parts["tardiness"].sum()) > 0.0
    assert float(parts["late_count"].sum()) > 0.0


def test_zero_variance_ripple_never_causes_tardiness() -> None:
    """alpha > 0 but beta_var = 0: execution matches the plan, so hard masking still holds."""
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.0, shift_end=_SHIFT)
    _, stoch = _make_envs(cfg, batch=16, n_customers=20, seed=32)
    actions, done = _sample_rollout(stoch, seed=4)
    assert bool(done.all())
    assert float(stoch.tardiness.sum()) == 0.0
    assert float(stoch.late_count.sum()) == 0.0


# ------------------------------------------------------------------------ guards


def _minimal_kwargs() -> dict:
    data = generate_cvrptw(
        2, 5, capacity=_CAPACITY, seed=41, shift_end=_SHIFT, service_time=_SERVICE
    )
    cfg = SimulatorConfig(shift_end=_SHIFT)
    return {
        "coords": data.coords, "demand": data.demand, "capacity": data.capacity,
        "time_windows": data.time_windows, "service_time": data.service_time,
        "shift_end": data.shift_end, "bank": build_edge_bank(data.coords, cfg),
        "sim_cfg": cfg,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kw: kw.update(bank=kw["bank"][..., :3, :]),                  # K mismatch
        lambda kw: kw.update(bank=kw["bank"][:, :4, :4]),                   # N mismatch
        lambda kw: kw.update(bank=kw["bank"][..., 0]),                      # not 5-D
        lambda kw: kw.update(shift_end=5.0),                                # clock desync
        lambda kw: kw.update(tardiness_weight=-1.0),
        lambda kw: kw.update(instance_idx=torch.tensor([0, 9])),            # out of range
        lambda kw: kw.update(instance_idx=torch.tensor([0.0, 1.0])),        # wrong dtype
        lambda kw: kw.update(instance_idx=torch.tensor([0, 1, 0])),         # wrong length
        lambda kw: kw.update(noise=torch.zeros(2, 6, 6, 3)),                # noise shape
    ],
)
def test_constructor_guards_raise(mutate) -> None:
    kwargs = _minimal_kwargs()
    mutate(kwargs)
    with pytest.raises(ValueError):
        CVRPTWStochEnv(**kwargs)


def test_state_tensors_live_on_the_coords_device() -> None:
    _, stoch = _make_envs(SimulatorConfig(shift_end=_SHIFT))
    stoch.reset()
    dev = stoch.coords.device
    for name in ("time", "tardiness", "late_count", "instance_idx", "_depot_idx"):
        assert getattr(stoch, name).device == dev, name