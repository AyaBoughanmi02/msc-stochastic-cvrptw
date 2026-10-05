"""Tests for a-priori (before-the-day) route selection.

Load-bearing claims:
  - the 'hindsight' rule reproduces evaluate_stoch exactly (same days, same selection);
  - an a-priori rule can never beat hindsight on the realised cost (hindsight is the
    minimum over the same candidates);
  - all rules report the components of the SAME selected trajectory (cost identity);
  - SAA selection days can never be the test days.
"""

from __future__ import annotations

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.apriori_selection import RULES, evaluate_stoch_apriori
from src.eval.evaluate_stoch import evaluate_stoch
from src.models.factory import build_cvrptw_meanonly_policy

_SHIFT, _SERVICE, _CAPACITY, _W = 4.0, 0.2, 20, 2.0


def _setup():
    torch.manual_seed(0)
    policy = build_cvrptw_meanonly_policy().eval()
    data = generate_cvrptw(5, 8, capacity=_CAPACITY, seed=0, shift_end=_SHIFT,
                           service_time=_SERVICE)
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.6, eta=0.6, omega=1.0, shift_end=_SHIFT)
    return policy, data, cfg


def test_hindsight_reproduces_evaluate_stoch() -> None:
    policy, data, cfg = _setup()
    ref = evaluate_stoch(policy, data, cfg, tardiness_weight=_W, augment=True,
                         eval_batch_size=2, scenario_seeds=(0, 1))
    out = evaluate_stoch_apriori(policy, data, cfg, tardiness_weight=_W, augment=True,
                                 eval_batch_size=2, scenario_seeds=(0, 1))
    assert set(out) == set(RULES)
    for key in ("cost", "distance", "tardiness", "late_count"):
        torch.testing.assert_close(out["hindsight"][key], ref[key], rtol=1e-5, atol=1e-6)


def test_no_apriori_rule_beats_hindsight_and_costs_add_up() -> None:
    policy, data, cfg = _setup()
    out = evaluate_stoch_apriori(policy, data, cfg, tardiness_weight=_W, augment=False,
                                 scenario_seeds=(0, 1, 2), n_selection_days=3)
    for rule in ("expected", "planned_distance", "saa", "average"):
        assert bool((out[rule]["cost"] >= out["hindsight"]["cost"] - 1e-5).all())
    for rule in RULES:
        rebuilt = out[rule]["distance"] + _W * (out[rule]["late_count"]
                                                + out[rule]["tardiness"])
        torch.testing.assert_close(out[rule]["cost"], rebuilt, rtol=1e-5, atol=1e-5)


def test_selection_days_never_overlap_test_days() -> None:
    import pytest

    from src.eval.apriori_selection import SELECTION_SEED_OFFSET

    policy, data, cfg = _setup()
    with pytest.raises(ValueError, match="overlap"):
        evaluate_stoch_apriori(policy, data, cfg, tardiness_weight=_W, augment=False,
                               scenario_seeds=(SELECTION_SEED_OFFSET,), n_selection_days=2)