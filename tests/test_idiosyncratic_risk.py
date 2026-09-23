"""Tests for the idiosyncratic risk field (zeta): risk that edge LENGTH cannot predict.

Load-bearing tests:

  - `test_zeta_zero_is_bit_exact`: zeta = 0 must reproduce every earlier result exactly,
    or the extension invalidates the experiments already run.
  - `test_bank_stays_augmentation_invariant`: 8x evaluation is only fair if the travel-time
    model survives the dihedral maps. The integer frequency is what guarantees it, and a
    non-integer one must be refused.
  - `test_zeta_decouples_risk_from_distance`: the POINT of the field. Without it, a policy
    can look risk-aware by avoiding long edges.
"""

from __future__ import annotations

import pytest
import torch

from src.data.simulator import MU, SIGMA2, SimulatorConfig, build_edge_bank
from src.eval.evaluate_stoch import bank_is_augmentation_invariant

_BASE = {"alpha": 0.3, "beta_var": 0.3, "shift_end": 4.0}


def _coords(seed: int = 0, b: int = 4, n: int = 21) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, 2, generator=gen)


def _risk_per_length(coords: torch.Tensor, cfg: SimulatorConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean mu, sigma^2 / mu^2) over the off-diagonal edges, horizon-averaged."""
    bank = build_edge_bank(coords, cfg)
    mu = bank[..., MU].mean(dim=-1).flatten()
    s2 = bank[..., SIGMA2].mean(dim=-1).flatten()
    keep = mu > 1e-6                                  # drop the zero diagonal
    return mu[keep], s2[keep] / mu[keep] ** 2


def test_zeta_zero_is_bit_exact() -> None:
    coords = _coords()
    old = build_edge_bank(coords, SimulatorConfig(eta=0.6, **_BASE))
    new = build_edge_bank(coords, SimulatorConfig(eta=0.6, zeta=0.0, **_BASE))
    assert torch.equal(old, new)


def test_zeta_leaves_mu_untouched() -> None:
    """The instance calibration and the reachability floor depend on mu alone."""
    coords = _coords()
    a = build_edge_bank(coords, SimulatorConfig(eta=0.3, **_BASE))
    b = build_edge_bank(coords, SimulatorConfig(eta=0.3, zeta=0.6, **_BASE))
    assert torch.equal(a[..., MU], b[..., MU])
    assert not torch.equal(a[..., SIGMA2], b[..., SIGMA2])


def test_zeta_decouples_risk_from_distance() -> None:
    coords = _coords()

    def corr(cfg: SimulatorConfig) -> float:
        mu, risk = _risk_per_length(coords, cfg)
        return float(torch.corrcoef(torch.stack([mu, risk]))[0, 1].abs())

    without = corr(SimulatorConfig(eta=0.6, **_BASE))
    with_zeta = corr(SimulatorConfig(eta=0.3, zeta=0.6, **_BASE))
    assert with_zeta < without * 0.75                 # measurably less predictable


def test_bank_stays_augmentation_invariant() -> None:
    """8x evaluation requires the travel-time model to survive the dihedral maps."""
    coords = _coords()
    cfg = SimulatorConfig(eta=0.3, zeta=0.6, risk_freq=3, **_BASE)
    assert bank_is_augmentation_invariant(coords, cfg)


def test_risk_field_is_symmetric_in_the_two_endpoints() -> None:
    coords = _coords()
    bank = build_edge_bank(coords, SimulatorConfig(eta=0.3, zeta=0.6, **_BASE))
    assert torch.allclose(bank[..., SIGMA2], bank[..., SIGMA2].transpose(1, 2), atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zeta": 1.0},                       # out of range
        {"zeta": -0.1},                      # negative
        {"eta": 0.6, "zeta": 0.5},           # |eta| + zeta >= 1 would allow r <= 0
        {"zeta": 0.3, "risk_freq": 2.5},     # non-integer breaks dihedral invariance
        {"zeta": 0.3, "risk_freq": 0},
    ],
)
def test_invalid_configs_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SimulatorConfig(**{**_BASE, **kwargs})