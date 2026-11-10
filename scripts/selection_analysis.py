"""Offline analysis of `selection_study` raw tensors (CPAIOR paper): R5 with CIs + D=4 check.
Loads <PREFIX>_raw_s{1..S}_b{beta}.pt (written by scripts.selection_study) and prints:

  1. R5  multi-start gain (its N1/N160 costs must equal section 1 of the study logs), per instance, seed-averaged, paired 95% CI:
         G_hind   = N1_average - N160_hindsight          (what choosing after the day shows)
         G_honest = N1_average - N160_planned_distance   (what a dispatcher actually gets)
         ratio G_hind / G_honest (description only).
     READING (fixed before running, cpaior_log session 6): claim "multi-start still pays
     under honest selection" only if the G_honest CI excludes 0 at every beta.
  2. D=4 CHECK  rules recomputed on the first 4 test days only and K=8, i.e. the thesis
     protocol (appendix B, runs\\apriori_15k_pd.json), to locate any difference between the
     thesis numbers and this study (more test days vs different checkpoints).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from scripts.compare_variants import paired_difference
from src.eval.selection_protocols import CandidateCosts, apply_rules

_N_SUBSETS = 32          # as in selection_study defaults (log header: "x 32 subsets (seed 0)")
_SUBSET_SEED = 0
_K_SUBSET = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline analysis of selection_study raw tensors")
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("BETA", "PREFIX"),
                   help="a variance level and the --out prefix used by selection_study")
    p.add_argument("--seeds", type=int, required=True, help="number of seeds (raw files s1..sS)")
    return p.parse_args()


def _load(prefix: str, seed_idx: int, beta: float) -> CandidateCosts:
    path = Path(f"{prefix}_raw_s{seed_idx}_b{beta:g}.pt")
    if not path.is_file():
        raise FileNotFoundError(f"missing raw file {path}")
    blob = torch.load(path, map_location="cpu")
    return CandidateCosts(**blob)


def _mean(rule_per_seed: list[dict[str, dict[str, Tensor]]], rule: str) -> Tensor:
    """Seed-averaged per-instance cost (B,)."""
    return torch.stack([r[rule]["cost"] for r in rule_per_seed]).mean(dim=0)


def main() -> None:
    args = parse_args()
    runs = [(float(b), prefix) for b, prefix in args.run]
    print(f"runs: {runs} | seeds {args.seeds} | subsets {_N_SUBSETS} (seed {_SUBSET_SEED})")

    print("\n1. R5  MULTI-START GAIN, N=1 -> N=160 (seed-averaged, paired 95% CI over instances)")
    full: dict[float, list] = {}
    for beta, prefix in runs:
        per_seed = []
        for s in range(1, args.seeds + 1):
            cc = _load(prefix, s, beta)
            per_seed.append(apply_rules(cc, ks=(_K_SUBSET,), ns=(1, 160), n_subsets=_N_SUBSETS,
                                        subset_seed=_SUBSET_SEED, k_subset=_K_SUBSET))
        full[beta] = per_seed
        n1 = _mean(per_seed, "N1_average")
        hind = _mean(per_seed, "N160_hindsight")
        plan = _mean(per_seed, "N160_planned_distance")
        print(f"\n  beta {beta:g}: N1_average {float(n1.mean()):.4f} | N160_hindsight "
              f"{float(hind.mean()):.4f} | N160_planned_distance {float(plan.mean()):.4f}")
        g_h = paired_difference(n1, hind)
        g_o = paired_difference(n1, plan)
        print(f"    G_hind   {g_h['mean']:+.4f} [{g_h['lo']:+.4f}, {g_h['hi']:+.4f}]")
        print(f"    G_honest {g_o['mean']:+.4f} [{g_o['lo']:+.4f}, {g_o['hi']:+.4f}]"
              f"   CI excludes 0: {'yes' if g_o['lo'] > 0 or g_o['hi'] < 0 else 'NO'}")
        print(f"    ratio G_hind / G_honest = {g_h['mean'] / g_o['mean']:.2f}")
        per = [(float(r['N1_average']['cost'].mean()) - float(r['N160_planned_distance']['cost'].mean()))
               for r in per_seed]
        print("    G_honest per seed: " + " ".join(f"{x:+.4f}" for x in per))

    print("\n2. D=4 CHECK: first 4 test days, K=8 (thesis protocol), seed-averaged")
    print(f"  {'beta':>5} {'hindsight':>10} {'expected':>10} {'plan_dist':>10} {'saa_K8':>10} {'average':>10}")
    for beta, prefix in runs:
        vals = []
        for s in range(1, args.seeds + 1):
            cc = _load(prefix, s, beta)
            cc4 = CandidateCosts(expected=cc.expected, planned_distance=cc.planned_distance,
                                 selection=cc.selection, test={k: v[:4] for k, v in cc.test.items()},
                                 n_aug=cc.n_aug, n_starts=cc.n_starts)
            vals.append(apply_rules(cc4, ks=(8,), ns=(), n_subsets=0))
        row = {r: float(_mean(vals, r).mean()) for r in
               ("hindsight", "expected", "planned_distance", "saa_out_K8", "average")}
        print(f"  {beta:>5g} {row['hindsight']:>10.4f} {row['expected']:>10.4f} "
              f"{row['planned_distance']:>10.4f} {row['saa_out_K8']:>10.4f} {row['average']:>10.4f}")


if __name__ == "__main__":
    main()