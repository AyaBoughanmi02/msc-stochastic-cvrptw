"""Per-candidate costs for studying a-priori selection rules (CPAIOR paper, tasks A2/A3).

WHY A SECOND MODULE
-------------------
`apriori_selection.evaluate_stoch_apriori` applies each choice rule inside the rollout loop
and returns only the chosen candidate's cost. To vary the number of selection days K and
the number of candidates N without re-running the policy, this module runs every
candidate (POMO start x augmentation) ONCE per day and KEEPS the per-candidate costs.
Every rule is then a tensor operation (`apply_rules`), and the raw tensors can be saved
for the order-statistics analysis (A3).

The rollouts are the same as in `apriori_selection` (same plan rollout, same online
execution, same day seeds: test days 0..D-1, selection days SELECTION_SEED_OFFSET + k).
`scripts.selection_study` checks this numerically against `evaluate_stoch_apriori` before
every study (`--self-check`).

RULES (each scored on the D test days; "before" = chosen before the test day)
  hindsight           cheapest REALISED candidate on each test day (after; optimistic)
  single              candidate 0 = augmentation 0 (identity, checked) and the first forced
                      start: one plain greedy route, no selection (before)
  average             mean over all candidates = expected cost of a randomly picked single
                      candidate (before)
  planned_distance    shortest planned route (analytic plan rollout) (before)
  expected            cheapest analytic expected cost of the planned route (before)
  saa_out_K{k}        cheapest mean realised cost on the first k SELECTION days, scored on
                      the test days (before; out-of-sample)
  saa_in_K{k}         the SAME choice, but reported with its own selection score, i.e. as a
                      paper would report it if it selected and evaluated on the same k days.
                      In-sample optimism(k) = saa_in_K{k} - saa_out_K{k}.
  N{m}_<rule>         <rule> restricted to a random subset of m candidates per instance,
                      averaged over R subsets (rule in hindsight / planned_distance /
                      saa_out_K{k_sub} / average)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SimulatorConfig, draw_noise
from src.eval.apriori_selection import SELECTION_SEED_OFFSET, _slice
from src.eval.evaluate_stoch import _augment
from src.train.pomo import build_starts
from src.train.pomo_stoch import assert_ood_eval_mode, build_stoch_env

COMPONENTS = ("cost", "distance", "tardiness", "late_count")
_PARTS = ("distance", "tardiness", "late_count")


@dataclass
class CandidateCosts:
    """Per-candidate costs for one (policy, beta). C = n_aug * n_starts candidates.

    Candidate index c = aug * n_starts + start (the layout of `apriori_selection`'s fold).
    All tensors are float32 on the CPU.
    """

    expected: Tensor                 # (B, C) analytic expected cost of the planned route
    planned_distance: Tensor         # (B, C) planned route length
    selection: Tensor                # (K, B, C) realised cost on selection days
    test: dict[str, Tensor]          # component -> (D, B, C) on test days
    n_aug: int
    n_starts: int

    @property
    def n_candidates(self) -> int:
        return self.n_aug * self.n_starts

    def to_dict(self) -> dict:
        return {"expected": self.expected, "planned_distance": self.planned_distance,
                "selection": self.selection, "test": self.test,
                "n_aug": self.n_aug, "n_starts": self.n_starts}


@torch.no_grad()
def _collect_chunk(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float,
    n_starts: int,
    augment: bool,
    test_days: Sequence[Tensor],     # each (b, N, N, n_bins), on the data's device
    sel_days: Sequence[Tensor],      # each (b, N, N, n_bins), on the data's device
) -> dict[str, Tensor]:
    b, n, _ = data.coords.shape
    dev = data.coords.device
    n_aug = 8 if augment else 1
    data_a = _augment(data) if augment else data
    if augment and not torch.equal(data_a.coords[:b], data.coords):
        raise RuntimeError("augmentation 0 is not the identity transform: the 'single' rule "
                           "would not be a plain greedy route on the original instance")
    starts = build_starts(n_aug * b, n_starts, dev)                  # (A*b*S,)

    def fold(flat: Tensor) -> Tensor:                                # (A*b*S,) -> (b, A*S)
        return flat.reshape(n_aug, b, n_starts).permute(1, 0, 2).reshape(b, n_aug * n_starts)

    plan_env = build_stoch_env(data_a, sim_cfg, n_starts, tardiness_weight,
                               reward_mode="analytic")
    plan_actions, _, expected = policy(plan_env, decode_type="greedy", force_first=starts)
    planned = plan_env.cost_components(plan_actions)["distance"]

    def run_day(noise: Tensor) -> dict[str, Tensor]:                 # key -> (b, A*S)
        day = noise
        if augment:
            day = noise.unsqueeze(0).expand(8, *noise.shape).reshape(8 * b, n, n, -1)
        env = build_stoch_env(data_a, sim_cfg, n_starts, tardiness_weight, noise=day)
        actions, _, cost = policy(env, decode_type="greedy", force_first=starts)
        parts = env.cost_components(actions)
        return {"cost": fold(cost), **{k: fold(parts[k]) for k in _PARTS}}

    sel = torch.stack([run_day(d)["cost"].float().cpu() for d in sel_days])   # (K, b, C)
    per_day = [run_day(d) for d in test_days]
    test = {k: torch.stack([r[k].float().cpu() for r in per_day]) for k in COMPONENTS}
    return {"expected": fold(expected).float().cpu(),
            "planned_distance": fold(planned).float().cpu(),
            "selection": sel, **{f"test_{k}": v for k, v in test.items()}}


@torch.no_grad()
def collect_candidate_costs(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float = 1.0,
    n_starts: int | None = None,
    augment: bool = True,
    eval_batch_size: int | None = None,
    scenario_seeds: Sequence[int] = (0,),
    n_selection_days: int = 64,
) -> CandidateCosts:
    """Run every candidate on every selection day and test day; keep all costs.

    Days are drawn ONCE for the whole set on the data's device and then sliced, exactly as
    in `evaluate_stoch_apriori`, so the first 8 selection days and the test days are the
    same random numbers as there. Drawn days are parked on the CPU to bound GPU memory.
    """
    if len(scenario_seeds) == 0:
        raise ValueError("scenario_seeds must be non-empty")
    if n_selection_days < 1:
        raise ValueError("n_selection_days must be >= 1")
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

    def draw(seed: int) -> Tensor:                                   # (B, N, N, n_bins) CPU
        return draw_noise((b, n, n, sim_cfg.n_bins), device=dev, dtype=data.coords.dtype,
                          generator=torch.Generator(device=dev).manual_seed(seed)).cpu()

    days = [draw(s) for s in scenario_seeds]
    sel_days = [draw(s) for s in selection_seeds]
    pieces = []
    for i in range(0, b, chunk):
        sl = slice(i, i + chunk)
        pieces.append(_collect_chunk(
            policy, _slice(data, sl), sim_cfg, tardiness_weight, starts, augment,
            [d[sl].to(dev) for d in days], [d[sl].to(dev) for d in sel_days]))

    def cat(key: str, dim: int) -> Tensor:
        return torch.cat([p[key] for p in pieces], dim=dim)

    return CandidateCosts(
        expected=cat("expected", 0), planned_distance=cat("planned_distance", 0),
        selection=cat("selection", 1),
        test={k: cat(f"test_{k}", 1) for k in COMPONENTS},
        n_aug=8 if augment else 1, n_starts=starts,
    )


# ----------------------------------------------------------------------------- rules
def _score_choice(test: dict[str, Tensor], idx: Tensor) -> dict[str, Tensor]:
    """idx (B,) candidate per instance -> component -> (B,) mean over test days.

    Days are accumulated in order and divided at the end, as in `apriori_selection`.
    """
    d, b, _ = test["cost"].shape
    out: dict[str, Tensor] = {}
    for k, v in test.items():                                        # v: (D, B, C)
        got = v.gather(2, idx.view(1, b, 1).expand(d, b, 1)).squeeze(2)   # (D, B)
        acc = got[0].clone()
        for j in range(1, d):
            acc = acc + got[j]
        out[k] = acc / d
    return out


def _saa_score(selection: Tensor, k: int) -> Tensor:
    """(K, B, C) -> (B, C) mean of the first k days, summed in day order (bit-compatible
    with `apriori_selection`, so near-ties are broken the same way)."""
    if not 1 <= k <= selection.shape[0]:
        raise ValueError(f"K={k} outside [1, {selection.shape[0]}] collected selection days")
    acc = selection[0].clone()
    for j in range(1, k):
        acc = acc + selection[j]
    return acc / k


def _hindsight(test: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per test day, the cheapest realised candidate; mean over days."""
    idx = test["cost"].argmin(dim=2, keepdim=True)                   # (D, B, 1)
    out = {}
    for k, v in test.items():
        got = v.gather(2, idx).squeeze(2)                            # (D, B)
        acc = got[0].clone()
        for j in range(1, got.shape[0]):
            acc = acc + got[j]
        out[k] = acc / got.shape[0]
    return out


def _average(test: dict[str, Tensor]) -> dict[str, Tensor]:
    out = {}
    for k, v in test.items():
        m = v.mean(dim=2)                                            # (D, B)
        acc = m[0].clone()
        for j in range(1, m.shape[0]):
            acc = acc + m[j]
        out[k] = acc / m.shape[0]
    return out


def _restrict(cc_tensor: Tensor, subset: Tensor) -> Tensor:
    """(..., B, C) gathered on subset (B, m) -> (..., B, m)."""
    lead = cc_tensor.shape[:-2]
    b, m = subset.shape
    idx = subset.view(*([1] * len(lead)), b, m).expand(*lead, b, m)
    return cc_tensor.gather(-1, idx)


def apply_rules(
    cc: CandidateCosts,
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    ns: Sequence[int] = (1, 8, 20, 40, 160),
    n_subsets: int = 32,
    subset_seed: int = 0,
    k_subset: int = 8,
) -> dict[str, dict[str, Tensor]]:
    """rule -> component -> (B,) float64. saa_in_K* has only the 'cost' component."""
    res: dict[str, dict[str, Tensor]] = {
        "hindsight": _hindsight(cc.test),
        "average": _average(cc.test),
    }
    b = cc.expected.shape[0]
    res["single"] = _score_choice(cc.test, torch.zeros(b, dtype=torch.long))
    res["planned_distance"] = _score_choice(cc.test, cc.planned_distance.argmin(dim=1))
    res["expected"] = _score_choice(cc.test, cc.expected.argmin(dim=1))
    for k in ks:
        score = _saa_score(cc.selection, k)                          # (B, C)
        idx = score.argmin(dim=1)
        res[f"saa_out_K{k}"] = _score_choice(cc.test, idx)
        res[f"saa_in_K{k}"] = {"cost": score.gather(1, idx.view(b, 1)).squeeze(1)}

    c = cc.n_candidates
    if len(ns) > 0 and n_subsets < 1:
        raise ValueError("n_subsets must be >= 1 when subset sizes are requested")
    gen = torch.Generator().manual_seed(subset_seed)
    for m in ns:
        if not 1 <= m <= c:
            raise ValueError(f"subset size {m} outside [1, {c}]")
        acc: dict[str, dict[str, Tensor]] = {}
        for _ in range(n_subsets):
            subset = torch.rand(b, c, generator=gen).argsort(dim=1)[:, :m]     # (B, m)
            test_s = {k: _restrict(v, subset) for k, v in cc.test.items()}     # (D, B, m)
            rules = {
                "hindsight": _hindsight(test_s),
                "average": _average(test_s),
                "planned_distance": _score_choice(
                    test_s, _restrict(cc.planned_distance, subset).argmin(dim=1)),
                f"saa_out_K{k_subset}": _score_choice(
                    test_s, _saa_score(_restrict(cc.selection, subset), k_subset).argmin(dim=1)),
            }
            for r, comp in rules.items():
                tgt = acc.setdefault(r, {})
                for k, v in comp.items():
                    tgt[k] = v.double() if k not in tgt else tgt[k] + v.double()
        for r, comp in acc.items():
            res[f"N{m}_{r}"] = {k: v / n_subsets for k, v in comp.items()}

    return {r: {k: v.double() for k, v in comp.items()} for r, comp in res.items()}