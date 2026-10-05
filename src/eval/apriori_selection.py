"""Evaluation with the route chosen BEFORE the day: hindsight vs a-priori selection.

WHY
---
`evaluate_stoch` runs every candidate (POMO starts x 8 augmentations) on the realised day
and keeps the cheapest REALISED one. That selection sees the noise -- its own docstring
calls it optimistic -- and no dispatcher can do it: the plan must be fixed before the day.
It is identical for every model, so model-vs-model comparisons stay fair, but absolute
costs are flattered and a comparison with a planner (e.g. OR-Tools) would not be.

Here the candidate is COMMITTED TO in advance and then driven online by the same policy on
the realised day. Three choice rules, each a column of the result:

  hindsight          cheapest realised cost (= evaluate_stoch; the optimistic reference)
  expected           cheapest EXPECTED cost of the candidate's planned route, computed
                     with the analytic reward -- i.e. with this day's travel-time VARIANCE.
                     The uncertainty-aware decision.
  planned_distance   shortest planned route: what a mean-only planner would choose.
                     The uncertainty-blind decision.
  saa                lowest AVERAGE realised cost over K independent SELECTION days
                     (sample-average approximation), each candidate driven online by the
                     policy exactly as on the test day. Uncertainty-aware with no model
                     mismatch: unlike "expected" it scores what will actually be executed.
                     Selection days never coincide with test days.
  average            no selection: the mean realised cost over all candidates, i.e. what a
                     start picked at random would cost. The floor every rule must beat.

The planned route comes from the same greedy policy in 'analytic' mode (clock on the mean,
lateness priced in expectation), which is exactly how the policy was trained. The
committed choice is a (start, augmentation) pair; on the day the policy re-decides at
every step from the realised clock, as in evaluate_stoch.

The variance enters the "expected" rule through an EXACT formula, not through a learned
network, so nothing has to extrapolate when the variance grows.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SimulatorConfig, draw_noise
from src.eval.evaluate_stoch import _augment
from src.train.pomo import build_starts
from src.train.pomo_stoch import assert_ood_eval_mode, build_stoch_env

RULES = ("hindsight", "expected", "planned_distance", "saa", "average")
SELECTION_SEED_OFFSET = 10_000      # selection days: seeds 10000, 10001, ... (never test)
_COMPONENTS = ("distance", "tardiness", "late_count")


def _slice(data: CVRPTWData, sl: slice) -> CVRPTWData:
    return CVRPTWData(
        coords=data.coords[sl], demand=data.demand[sl], capacity=data.capacity,
        time_windows=data.time_windows[sl], service_time=data.service_time[sl],
        shift_end=data.shift_end,
    )


@torch.no_grad()
def _chunk(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float,
    n_starts: int,
    augment: bool,
    noises: Sequence[Tensor],      # one (b, N, N, K) realised day per scenario
    selection_noises: Sequence[Tensor],  # one (b, N, N, K) day per SELECTION day
) -> dict[str, dict[str, Tensor]]:
    """rule -> component -> (b,) tensor, averaged over the scenarios."""
    b, n, _ = data.coords.shape
    dev = data.coords.device
    n_aug = 8 if augment else 1
    data_a = _augment(data) if augment else data
    starts = build_starts(n_aug * b, n_starts, dev)                  # (A*b*S,)

    def fold(flat: Tensor) -> Tensor:                                # (A*b*S,) -> (b, A*S)
        return flat.reshape(n_aug, b, n_starts).permute(1, 0, 2).reshape(b, n_aug * n_starts)

    # 1. PLAN, before the day: analytic rollout -> expected cost and planned distance.
    plan_env = build_stoch_env(data_a, sim_cfg, n_starts, tardiness_weight,
                               reward_mode="analytic")
    plan_actions, _, expected = policy(plan_env, decode_type="greedy", force_first=starts)
    planned_dist = plan_env.cost_components(plan_actions)["distance"]
    choice = {
        "expected": fold(expected).argmin(dim=1, keepdim=True),          # (b, 1)
        "planned_distance": fold(planned_dist).argmin(dim=1, keepdim=True),
    }

    def run_day(noise: Tensor) -> dict[str, Tensor]:
        """Every candidate driven online on one day. key -> (b, A*S)."""
        day = noise
        if augment:
            day = noise.unsqueeze(0).expand(8, *noise.shape).reshape(8 * b, n, n, -1)
        env = build_stoch_env(data_a, sim_cfg, n_starts, tardiness_weight, noise=day)
        actions, _, cost = policy(env, decode_type="greedy", force_first=starts)
        parts = env.cost_components(actions)
        return {"cost": fold(cost), **{k: fold(parts[k]) for k in _COMPONENTS}}

    # 2. SAA: score every candidate on independent selection days, before the test day.
    if len(selection_noises) == 0:
        raise ValueError("the 'saa' rule needs at least one selection day")
    saa_score = sum(run_day(d)["cost"] for d in selection_noises) / len(selection_noises)
    choice["saa"] = saa_score.argmin(dim=1, keepdim=True)            # (b, 1)

    # 3. EXECUTE on each test day; every candidate is driven online by the policy.
    totals: dict[str, dict[str, Tensor]] = {r: {} for r in RULES}
    for noise in noises:
        realised = run_day(noise)
        picks = {"hindsight": realised["cost"].argmin(dim=1, keepdim=True), **choice}
        for key, val in realised.items():
            vals = {rule: val.gather(1, idx).squeeze(1) for rule, idx in picks.items()}
            vals["average"] = val.mean(dim=1)                        # (b,) no selection
            for rule, got in vals.items():
                totals[rule][key] = got if key not in totals[rule] else totals[rule][key] + got
    return {r: {k: v / len(noises) for k, v in d.items()} for r, d in totals.items()}


@torch.no_grad()
def evaluate_stoch_apriori(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float = 1.0,
    n_starts: int | None = None,
    augment: bool = True,
    eval_batch_size: int | None = None,
    scenario_seeds: Sequence[int] = (0,),
    n_selection_days: int = 8,
) -> dict[str, dict[str, Tensor]]:
    """rule -> {'cost', 'distance', 'tardiness', 'late_count'} -> (B,), averaged over days.

    Selection days use seeds SELECTION_SEED_OFFSET + k; the test (scenario) seeds must not
    overlap them, or the 'saa' rule would be choosing on the very days it is scored on.

    Uses EXACTLY the days of evaluate_stoch (same seeds, drawn once for the whole set and
    then sliced), so the 'hindsight' column reproduces evaluate_stoch's numbers.
    """
    if len(scenario_seeds) == 0:
        raise ValueError("scenario_seeds must be non-empty")
    assert_ood_eval_mode(policy)
    b, n, _ = data.coords.shape
    starts = (n - 1) if n_starts is None else n_starts
    if not 1 <= starts <= n - 1:
        raise ValueError(f"n_starts must be in [1, {n - 1}], got {starts}")
    chunk = b if eval_batch_size is None else min(eval_batch_size, b)
    dev = data.coords.device
    selection_seeds = [SELECTION_SEED_OFFSET + k for k in range(n_selection_days)]
    if set(selection_seeds) & set(scenario_seeds):
        raise ValueError("selection days and test days overlap")

    def draw(seed: int) -> Tensor:                                   # (B, N, N, K)
        return draw_noise((b, n, n, sim_cfg.n_bins), device=dev, dtype=data.coords.dtype,
                          generator=torch.Generator(device=dev).manual_seed(seed))

    days = [draw(seed) for seed in scenario_seeds]
    sel_days = [draw(seed) for seed in selection_seeds]
    pieces = []
    for i in range(0, b, chunk):
        sl = slice(i, i + chunk)
        pieces.append(_chunk(policy, _slice(data, sl), sim_cfg, tardiness_weight, starts,
                             augment, [d[sl] for d in days], [d[sl] for d in sel_days]))
    return {r: {k: torch.cat([p[r][k] for p in pieces], dim=0) for k in pieces[0][r]}
            for r in RULES}