"""Tests for the stochastic travel-time simulator (Edge Bank + lognormal sampler).

Two tests are load-bearing:

1. `test_reduces_to_variant_a_bit_exactly` -- with `alpha = 0` and `beta_var = 0` the
   Edge Bank must collapse to variant A's travel-time model BIT-EXACTLY (mu == the
   Euclidean seam, sigma^2 == 0, sampled travel == mu). This is what protects the
   certified deterministic case when the stochastic machinery is switched off.
2. `test_sampler_matches_requested_moments` -- the lognormal moment matching must
   actually deliver E[travel] = mu and Var[travel] = sigma2. Tolerances are derived from
   a batch-means standard error computed from the draws themselves, not hand-tuned.
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
    apply_noise,
    assert_ripple_reachable,
    bin_of,
    build_edge_bank,
    draw_noise,
    horizon_average,
    lookup_bin,
    lookup_row,
    ripple_reachability_floor,
    sample_travel,
)


def _coords(b: int = 4, n: int = 12, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(b, n, 2, generator=gen)  # (B, N, 2)


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    """Flatten the off-diagonal entries of a (B, N, N) tensor -> (M,)."""
    n = x.size(1)
    eye = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand_as(x)
    return x[~eye]


# --------------------------------------------------------------------------- config


def test_config_validation_raises() -> None:
    for kwargs in (
        {"alpha": 1.0},          # would allow mu <= 0
        {"alpha": -0.1},
        {"beta_var": -1e-9},
        {"eta": 1.0},            # would allow r <= 0
        {"n_bins": 0},
        {"period": 0.0},
        {"field_scale": 0.0},
        {"shift_end": -1.0},
    ):
        with pytest.raises(ValueError):
            SimulatorConfig(**kwargs)


def test_bin_width() -> None:
    cfg = SimulatorConfig(n_bins=24, shift_end=4.0)
    assert cfg.bin_width == pytest.approx(4.0 / 24)


# ----------------------------------------------------------------- shapes and guards


def test_bank_shape_device_dtype() -> None:
    cfg = SimulatorConfig(n_bins=8)
    coords = _coords(3, 7).double()
    bank = build_edge_bank(coords, cfg)
    assert bank.shape == (3, 7, 7, 8, 2)
    assert bank.dtype == torch.float64          # dtype follows coords
    assert bank.device == coords.device
    assert horizon_average(bank).shape == (3, 7, 7, 2)


def test_shape_guards_raise() -> None:
    cfg = SimulatorConfig()
    with pytest.raises(ValueError):
        build_edge_bank(torch.rand(7, 2), cfg)              # missing batch dim
    with pytest.raises(ValueError):
        build_edge_bank(torch.rand(3, 7, 3), cfg)           # not 2-D coordinates
    with pytest.raises(ValueError):
        horizon_average(torch.rand(3, 7, 7, 2))             # not a bank
    bank = build_edge_bank(_coords(3, 5), cfg)
    with pytest.raises(ValueError):
        lookup_bin(bank, torch.zeros(2, dtype=torch.long))  # wrong batch size
    with pytest.raises(ValueError):
        lookup_row(bank, torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long))


# ------------------------------------------------------- the reduction to variant A


@pytest.mark.parametrize("eta", [0.0, 0.6])
def test_reduces_to_variant_a_bit_exactly(eta: float) -> None:
    """alpha = 0 and beta_var = 0 must reproduce variant A's travel-time model exactly."""
    cfg = SimulatorConfig(alpha=0.0, beta_var=0.0, eta=eta)
    data = generate_cvrptw(6, 20, seed=11)
    bank = build_edge_bank(data.coords, cfg)                       # (B, N, N, K, 2)
    d = pairwise_distances(data.coords)                            # (B, N, N)

    for k in range(cfg.n_bins):
        assert torch.equal(bank[..., k, MU], d), f"mu != distance in bin {k}"
    assert torch.equal(bank[..., SIGMA2], torch.zeros_like(bank[..., SIGMA2]))

    # The horizon average the encoder sees is the plain distance matrix. Not asserted
    # bit-exact: a K-way float mean of identical values need not round back to that value.
    # The env never uses it -- it reads per-bin slices, which ARE exact (asserted above).
    avg = horizon_average(bank)                                    # (B, N, N, 2)
    assert torch.allclose(avg[..., MU], d, rtol=1e-6, atol=0.0)
    assert torch.equal(avg[..., SIGMA2], torch.zeros_like(d))

    # Sampling is the identity when there is no variance -- bit-exact, not just close.
    gen = torch.Generator(device="cpu").manual_seed(3)
    sampled = sample_travel(bank[..., 0, MU], bank[..., 0, SIGMA2], generator=gen)
    assert torch.equal(sampled, d)


# --------------------------------------------------------------- the moment model


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.99])
def test_mu_strictly_positive_off_diagonal(alpha: float) -> None:
    cfg = SimulatorConfig(alpha=alpha)
    bank = build_edge_bank(_coords(4, 12, seed=5), cfg)
    mu = bank[..., MU]                                             # (B, N, N, K)
    diag = mu.diagonal(dim1=1, dim2=2)                             # (B, K, N)
    assert torch.equal(diag, torch.zeros_like(diag))               # self-loops stay 0
    n = mu.size(1)
    eye = torch.eye(n, dtype=torch.bool).view(1, n, n, 1)
    assert (mu.masked_fill(eye, 1.0) > 0.0).all()


def test_bank_is_symmetric() -> None:
    """Congestion is modelled symmetrically (d and the midpoint both are). Documented limit."""
    cfg = SimulatorConfig(eta=0.6)
    bank = build_edge_bank(_coords(3, 9, seed=6), cfg)
    assert torch.allclose(bank, bank.transpose(1, 2), atol=1e-6)


def test_is_mean_preserving_flag() -> None:
    assert SimulatorConfig(period=4.0, shift_end=4.0, n_bins=24).is_mean_preserving
    assert SimulatorConfig(period=2.0, shift_end=4.0, n_bins=24).is_mean_preserving   # 2 cycles
    assert not SimulatorConfig(period=8.0, shift_end=4.0, n_bins=24).is_mean_preserving
    assert not SimulatorConfig(period=4.0, shift_end=4.0, n_bins=1).is_mean_preserving


def test_ripple_is_mean_preserving() -> None:
    """With period == shift_end the K bin centres span one full cycle, so mean_k mu == d.

    This is what keeps Stage 2's feasibility-by-construction calibration valid.
    """
    coords = _coords(4, 12, seed=7)
    d = pairwise_distances(coords)
    for alpha in (0.3, 0.9):
        for period in (4.0, 2.0):                                  # 1 and 2 whole cycles
            cfg = SimulatorConfig(alpha=alpha, period=period, shift_end=4.0)
            assert cfg.is_mean_preserving
            mu_bar = horizon_average(build_edge_bank(coords, cfg))[..., MU]
            assert torch.allclose(mu_bar, d, atol=1e-6)

    # A non-integer number of cycles is NOT mean-preserving -- the flag must catch it.
    skewed = SimulatorConfig(alpha=0.9, period=8.0, shift_end=4.0)  # half a cycle
    assert not skewed.is_mean_preserving
    assert not torch.allclose(
        horizon_average(build_edge_bank(coords, skewed))[..., MU], d, atol=1e-3
    )


def test_horizon_variance_matches_closed_form_when_homogeneous() -> None:
    """eta = 0 => sigma_bar^2 / mu_bar^2 == beta_var * (1 + alpha^2 / 2) on every edge.

    mean_k sin = 0 and mean_k sin^2 = 1/2 over a full cycle. This pins the algebra that
    motivates the reliability field: without it, variance is a fixed multiple of d^2.
    """
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.1, eta=0.0)
    avg = horizon_average(build_edge_bank(_coords(4, 12, seed=8), cfg))
    ratio = _offdiag(avg[..., SIGMA2] / avg[..., MU].square())
    expected = cfg.beta_var * (1.0 + cfg.alpha**2 / 2.0)
    assert torch.allclose(ratio, torch.full_like(ratio, expected), rtol=1e-4)


def test_reliability_field_decouples_variance_from_distance() -> None:
    """eta > 0 must make sigma_bar^2 / mu_bar^2 vary across edges; eta = 0 must not.

    This is the property the variance gate needs: without it sigma^2 is a deterministic
    function of distance and the gate degenerates into a squared-distance penalty.
    """
    coords = _coords(4, 12, seed=9)

    homogeneous = horizon_average(build_edge_bank(coords, SimulatorConfig(eta=0.0)))
    r_hom = _offdiag(homogeneous[..., SIGMA2] / homogeneous[..., MU].square())
    assert (r_hom.std() / r_hom.mean()).item() < 1e-4               # constant

    heterogeneous = horizon_average(build_edge_bank(coords, SimulatorConfig(eta=0.6)))
    r_het = _offdiag(heterogeneous[..., SIGMA2] / heterogeneous[..., MU].square())
    assert (r_het.std() / r_het.mean()).item() > 0.10               # genuinely spread

    # mu is untouched by the field: the instance calibration cannot move with eta.
    assert torch.equal(homogeneous[..., MU], heterogeneous[..., MU])


def test_phase_lag_staggers_peaks_across_edges() -> None:
    """phase_k > 0 must make different edges peak in different bins."""
    coords = _coords(2, 14, seed=10)
    mu = build_edge_bank(coords, SimulatorConfig(alpha=0.3, phase_k=1.0))[..., MU]
    peak_bin = mu.argmax(dim=-1)                                   # (B, N, N)
    assert _offdiag(peak_bin).unique().numel() > 1

    flat = build_edge_bank(coords, SimulatorConfig(alpha=0.3, phase_k=0.0))[..., MU]
    assert _offdiag(flat.argmax(dim=-1)).unique().numel() == 1     # all peak together


# -------------------------------------------------------------- bins and lookups


def test_bin_of_boundaries_are_clamped() -> None:
    cfg = SimulatorConfig(n_bins=24, shift_end=4.0)
    w = cfg.bin_width
    clock = torch.tensor([0.0, 0.99 * w, w, 2.5 * w, cfg.shift_end, 10.0, -1.0])
    got = bin_of(clock, cfg)
    assert got.dtype == torch.long
    assert torch.equal(got, torch.tensor([0, 0, 1, 2, 23, 23, 0]))


def test_lookups_match_direct_indexing() -> None:
    cfg = SimulatorConfig(n_bins=6, eta=0.5)
    bank = build_edge_bank(_coords(5, 9, seed=12), cfg)
    bin_idx = torch.tensor([0, 3, 5, 2, 1])
    from_idx = torch.tensor([0, 4, 8, 1, 7])

    at_bin = lookup_bin(bank, bin_idx)                              # (B, N, N, 2)
    row = lookup_row(bank, from_idx, bin_idx)                       # (B, N, 2)
    for b in range(5):
        assert torch.equal(at_bin[b], bank[b, :, :, bin_idx[b], :])
        assert torch.equal(row[b], bank[b, from_idx[b], :, bin_idx[b], :])


# ------------------------------------------------------------------- the sampler


@pytest.mark.parametrize(
    "mu_val, cv",
    [(0.5, 0.1), (0.5, 0.32), (1.2, 0.6), (0.05, 1.0)],
)
def test_sampler_matches_requested_moments(mu_val: float, cv: float) -> None:
    """Empirical mean/variance must match (mu, sigma2) within a batch-means standard error.

    The tolerance is 5 standard errors, where the standard error is estimated from the
    spread of per-group statistics -- so it adapts to the heavy tail at high CV instead of
    being a hand-picked constant.
    """
    groups, per_group = 32, 20_000
    sigma2_val = (cv * mu_val) ** 2
    mu = torch.full((groups, per_group), mu_val, dtype=torch.float64)
    sigma2 = torch.full_like(mu, sigma2_val)

    gen = torch.Generator(device="cpu").manual_seed(2024)
    draws = sample_travel(mu, sigma2, generator=gen)                # (G, P)
    assert (draws > 0.0).all()                                      # lognormal support

    group_mean = draws.mean(dim=1)                                  # (G,)
    group_var = draws.var(dim=1, unbiased=True)                     # (G,)
    se_mean = group_mean.std(unbiased=True) / math.sqrt(groups)
    se_var = group_var.std(unbiased=True) / math.sqrt(groups)

    assert abs(group_mean.mean().item() - mu_val) <= 5.0 * se_mean.item()
    assert abs(group_var.mean().item() - sigma2_val) <= 5.0 * se_var.item()


def test_sampler_is_reproducible_and_seed_sensitive() -> None:
    mu = torch.full((256,), 0.7)
    sigma2 = torch.full((256,), 0.05)
    a = sample_travel(mu, sigma2, generator=torch.Generator().manual_seed(1))
    b = sample_travel(mu, sigma2, generator=torch.Generator().manual_seed(1))
    c = sample_travel(mu, sigma2, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_common_random_numbers_give_identical_travel() -> None:
    """One shared noise field => POMO start replicas face the same realised day."""
    cfg = SimulatorConfig(eta=0.6)
    avg = horizon_average(build_edge_bank(_coords(2, 8, seed=13), cfg))
    mu, sigma2 = avg[..., MU], avg[..., SIGMA2]

    z = draw_noise(mu.shape, device=mu.device, dtype=mu.dtype,
                   generator=torch.Generator().manual_seed(17))
    assert torch.equal(apply_noise(mu, sigma2, z), apply_noise(mu, sigma2, z))
    # ... and fresh noise does not.
    z2 = draw_noise(mu.shape, device=mu.device, dtype=mu.dtype,
                    generator=torch.Generator().manual_seed(18))
    assert not torch.equal(apply_noise(mu, sigma2, z), apply_noise(mu, sigma2, z2))


def test_apply_noise_broadcasts_and_keeps_zero_mu_at_zero() -> None:
    mu = torch.tensor([[0.0, 0.4, 1.1]])                            # (1, 3) incl. a self-loop
    sigma2 = torch.tensor([[0.0, 0.02, 0.30]])                      # (1, 3)
    z = torch.randn(64, 3)                                          # (64, 3)
    out = apply_noise(mu, sigma2, z)                                # (64, 3)
    assert out.shape == (64, 3)
    assert torch.equal(out[:, 0], torch.zeros(64))                  # mu = 0 stays exactly 0
    assert (out[:, 1:] > 0.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draw_noise_rejects_generator_device_mismatch() -> None:
    with pytest.raises(ValueError):
        draw_noise((4,), device="cuda", generator=torch.Generator(device="cpu"))


# ------------------------------------------------------------ reachability guard


def test_ripple_reachability_floor_and_guard() -> None:
    service = 0.2
    safe = SimulatorConfig(alpha=0.3, shift_end=4.0)
    assert ripple_reachability_floor(safe, service) == pytest.approx(
        1.3 * 2.0 * math.sqrt(2.0) + service
    )
    assert_ripple_reachable(safe, service)                           # must not raise

    unsafe = SimulatorConfig(alpha=0.5, shift_end=4.0)               # 4.44 > 4.0
    with pytest.raises(ValueError):
        assert_ripple_reachable(unsafe, service)