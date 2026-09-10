"""Tests for the stochastic CVRPTW evaluator.

Load-bearing tests:
  - `test_augmentation_equals_best_of_eight_separate_runs`: proves the 8 augmentations all
    face the SAME realised day. If each drew its own noise, "best of 8" would partly be
    selecting the luckiest weather instead of the best route, and every augmented number
    in the thesis would be optimistically biased by an unknown amount.
  - `test_chunking_is_bitwise_result_preserving`: eval_batch_size exists only to fit 4 GB.
    If it also changed which day an instance saw, the chunk size would silently become an
    experimental variable.
  - `test_bank_is_invariant_under_the_dihedral_maps`: augmentation must change what the
    POLICY sees without changing the travel-time model it is judged on.
"""

from __future__ import annotations

import pytest
import torch

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.evaluate_stoch import (
    _augment,
    bank_is_augmentation_invariant,
    evaluate_stoch,
)
from src.models.factory import build_cvrptw_stoch_policy

_SHIFT, _SERVICE, _CAPACITY = 4.0, 0.2, 20


def _data(batch: int = 4, n_customers: int = 8, seed: int = 0) -> CVRPTWData:
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


def _cfg(**over) -> SimulatorConfig:
    return SimulatorConfig(**{"alpha": 0.3, "beta_var": 0.3, "eta": 0.6,
                              "shift_end": _SHIFT, **over})


def _policy(seed: int = 0):
    torch.manual_seed(seed)
    return build_cvrptw_stoch_policy().eval()   # eval mode is the caller's job, by design


# ------------------------------------------------------------- augmentation is sound


@pytest.mark.parametrize("cfg", [_cfg(), _cfg(eta=0.0), _cfg(alpha=0.0, beta_var=0.0)])
def test_bank_is_invariant_under_the_dihedral_maps(cfg: SimulatorConfig) -> None:
    """Distances, midpoints and the centroid all map consistently under an isometry."""
    assert bank_is_augmentation_invariant(_data(batch=5, n_customers=9, seed=1).coords, cfg)


def test_augment_keeps_node_indexing_and_moves_only_coordinates() -> None:
    data = _data(batch=3, n_customers=6, seed=2)
    aug = _augment(data)
    b, n = 3, 7
    assert aug.coords.shape == (8 * b, n, 2)
    # Block 0 is the identity map, so the originals must come through untouched.
    assert torch.equal(aug.coords[:b], data.coords)
    # Everything index-based is simply repeated -- that is what lets the day be shared.
    for a in range(8):
        sl = slice(a * b, (a + 1) * b)
        assert torch.equal(aug.demand[sl], data.demand)
        assert torch.equal(aug.time_windows[sl], data.time_windows)
        assert torch.equal(aug.service_time[sl], data.service_time)


def test_augmentation_equals_best_of_eight_separate_runs() -> None:
    """The 8 augmentations must face one shared day, not eight independent ones."""
    data, cfg, policy = _data(batch=3, n_customers=8, seed=3), _cfg(), _policy()
    b = data.coords.size(0)
    seeds = (7,)

    together = evaluate_stoch(
        policy, data, cfg, tardiness_weight=2.0, n_starts=3, augment=True,
        scenario_seeds=seeds,
    )["cost"]

    # draw_noise depends only on shape and seed, so each separate run below gets the very
    # same field -- which is exactly the property being checked.
    aug = _augment(data)
    separately = []
    for a in range(8):
        sl = slice(a * b, (a + 1) * b)
        variant = CVRPTWData(
            coords=aug.coords[sl], demand=aug.demand[sl], capacity=data.capacity,
            time_windows=aug.time_windows[sl], service_time=aug.service_time[sl],
            shift_end=data.shift_end,
        )
        separately.append(
            evaluate_stoch(policy, variant, cfg, tardiness_weight=2.0, n_starts=3,
                           augment=False, scenario_seeds=seeds)["cost"]
        )
    assert torch.equal(together, torch.stack(separately).min(dim=0).values)


def test_augmentation_never_hurts() -> None:
    """Block 0 is the identity map, so best-of-8 can only improve on the unaugmented run."""
    data, cfg, policy = _data(batch=4, n_customers=8, seed=4), _cfg(), _policy()
    kwargs = dict(tardiness_weight=2.0, n_starts=3, scenario_seeds=(11,))
    plain = evaluate_stoch(policy, data, cfg, augment=False, **kwargs)["cost"]
    augmented = evaluate_stoch(policy, data, cfg, augment=True, **kwargs)["cost"]
    assert bool((augmented <= plain + 1e-6).all())


# --------------------------------------------------------------------- the mechanics


def test_chunking_is_bitwise_result_preserving() -> None:
    data, cfg, policy = _data(batch=6, n_customers=8, seed=5), _cfg(), _policy()
    kwargs = dict(tardiness_weight=1.0, n_starts=3, augment=False, scenario_seeds=(0, 1))
    whole = evaluate_stoch(policy, data, cfg, **kwargs)
    chunked = evaluate_stoch(policy, data, cfg, eval_batch_size=2, **kwargs)
    for key in whole:
        assert torch.equal(whole[key], chunked[key]), key


def test_scenarios_are_averaged_and_paired_across_calls() -> None:
    data, cfg, policy = _data(batch=4, n_customers=8, seed=6), _cfg(), _policy()
    kwargs = dict(tardiness_weight=1.5, n_starts=3, augment=False)
    both = evaluate_stoch(policy, data, cfg, scenario_seeds=(0, 1), **kwargs)["cost"]
    first = evaluate_stoch(policy, data, cfg, scenario_seeds=(0,), **kwargs)["cost"]
    second = evaluate_stoch(policy, data, cfg, scenario_seeds=(1,), **kwargs)["cost"]
    assert torch.allclose(both, (first + second) / 2, atol=1e-6)
    # Same seeds => same days => two models can be compared as a paired sample.
    assert torch.equal(first, evaluate_stoch(policy, data, cfg,
                                             scenario_seeds=(0,), **kwargs)["cost"])
    assert not torch.equal(first, second)


def test_reported_components_belong_to_the_selected_route() -> None:
    """cost is the selection criterion; distance/tardiness must be that route's, not a mean."""
    data, cfg, policy = _data(batch=4, n_customers=8, seed=7), _cfg(), _policy()
    w = 2.0
    out = evaluate_stoch(policy, data, cfg, tardiness_weight=w, n_starts=3,
                         augment=True, scenario_seeds=(3,))
    assert torch.allclose(
        out["cost"], out["distance"] + w * (out["late_count"] + out["tardiness"]),
        atol=1e-4,
    )


def test_reduces_to_a_pure_distance_objective_with_the_bank_off() -> None:
    data, policy = _data(batch=4, n_customers=8, seed=8), _policy()
    out = evaluate_stoch(
        policy, data, _cfg(alpha=0.0, beta_var=0.0), tardiness_weight=5.0,
        n_starts=3, augment=True, scenario_seeds=(0,),
    )
    assert float(out["tardiness"].sum()) == 0.0
    assert float(out["late_count"].sum()) == 0.0
    assert torch.equal(out["cost"], out["distance"])


def test_ood_shift_raises_tardiness() -> None:
    data, policy = _data(batch=8, n_customers=10, seed=9), _policy()
    kwargs = dict(tardiness_weight=2.0, n_starts=4, augment=False, scenario_seeds=(0, 1))
    id_out = evaluate_stoch(policy, data, _cfg(beta_var=0.3), **kwargs)
    ood_out = evaluate_stoch(policy, data, _cfg(beta_var=0.9), **kwargs)
    assert float(ood_out["tardiness"].mean()) > float(id_out["tardiness"].mean())


# -------------------------------------------------------------------------- guards


def test_train_mode_is_refused() -> None:
    policy = _policy()
    policy.train()
    with pytest.raises(RuntimeError, match="train mode"):
        evaluate_stoch(policy, _data(batch=2, n_customers=8, seed=10), _cfg(), n_starts=2)


@pytest.mark.parametrize("bad", [{"n_starts": 0}, {"n_starts": 99}, {"scenario_seeds": ()}])
def test_argument_guards_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        evaluate_stoch(_policy(), _data(batch=2, n_customers=8, seed=11), _cfg(), **bad)