"""Aggregate several training seeds per variant into one defensible comparison.

Example:
    python -m scripts.seed_study \
        --baseline checkpoints/B_rand_s1234.pt checkpoints/B_rand_s7.pt checkpoints/B_rand_s2024.pt \
        --candidate checkpoints/D_rand_s1234.pt checkpoints/D_rand_s7.pt checkpoints/D_rand_s2024.pt

WHY THIS EXISTS. The paired comparison reports a confidence interval across INSTANCES: it
answers "given these two trained networks, is one reliably cheaper?". It says nothing
about whether a different random initialisation would have produced the same two networks.
Training-run variability is a separate source of uncertainty, and a single run per variant
cannot measure it. That is the standard objection to a result like this, and the only
answer is to train several seeds.

THREE VIEWS, because they answer different questions:

  1. PER-SEED MEANS -- the raw material. Does every seed point the same way, or is the
     headline carried by one lucky run? Reported before any aggregation so nothing hides.

  2. SEED-AVERAGED PAIRED -- each variant's per-instance cost is averaged over its seeds
     first, then the two are compared instance by instance. Averaging damps the seed noise
     and the pairing damps the instance noise, so this is the tightest honest estimator.
     It is the headline number.

  3. ACROSS-SEED SPREAD -- the difference of means computed separately per seed, then its
     mean and standard deviation. With only a handful of seeds a formal interval is very
     wide (t has few degrees of freedom), so consistency of SIGN and magnitude is the
     informative part. Reported as such rather than dressed up as significance.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import replace

import torch
from torch import Tensor

from src.data.cvrptw_generator import generate_cvrptw
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import evaluate_stoch

# Two-sided 95% t critical values by degrees of freedom, for small seed counts.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228}
_Z95 = 1.959964


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed comparison of two variants")
    p.add_argument("--baseline", type=str, nargs="+", required=True,
                   help="one checkpoint per training seed")
    p.add_argument("--candidate", type=str, nargs="+", required=True)
    p.add_argument("--shift-param", type=str, default="beta_var",
                   choices=["beta_var", "eta", "field_scale", "alpha"],
                   help="which simulator parameter to shift. 'beta_var' is a MAGNITUDE "
                        "shift (everything gets uniformly riskier, the risk RANKING is "
                        "unchanged). 'eta' is a STRUCTURAL shift: flipping its sign "
                        "inverts WHICH corridors are unreliable while keeping the amount "
                        "of heterogeneity identical")
    p.add_argument("--shift-values", type=float, nargs="+", default=None,
                   help="values of --shift-param to evaluate; the FIRST must be the "
                        "trained value, since it is treated as in-distribution")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9],
                   help="shorthand for --shift-param beta_var --shift-values ...")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--n-starts", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _interval(values: list[float]) -> tuple[float, float, float]:
    """Mean, half-width of a 95% t interval, and sd. Half-width is inf below 2 samples."""
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("inf"), float("inf")
    sd = statistics.stdev(values)
    t = _T95.get(n - 1, _Z95)
    return mean, t * sd / math.sqrt(n), sd


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    augment = not args.no_augment
    scenario_seeds = tuple(range(args.scenarios))

    param = args.shift_param
    values = args.shift_values if args.shift_values is not None else args.beta_var

    baselines = [load_stoch_policy(p, device) for p in args.baseline]
    candidates = [load_stoch_policy(p, device) for p in args.candidate]
    ref = baselines[0]
    weight = float(ref.blob["tardiness_weight"])
    for loaded in baselines + candidates:
        if float(loaded.blob["tardiness_weight"]) != weight:
            raise ValueError(
                "checkpoints disagree on tardiness_weight; the reward defines the "
                "problem, so the comparison would be uninterpretable"
            )
        for key in ("n_customers", "capacity", "shift_end", "tw_half_width",
                    "service_time"):
            if loaded.blob[key] != ref.blob[key]:
                raise ValueError(f"checkpoints disagree on '{key}' -- not one problem")

    # The first value is reported as in-distribution, so it must be what was trained on;
    # otherwise every "degradation" below would be measured from the wrong baseline.
    trained = getattr(ref.sim_cfg, param)
    if abs(values[0] - trained) > 1e-9:
        raise ValueError(
            f"the first --shift-values entry ({values[0]}) is not the trained "
            f"{param}={trained}. The first value is treated as in-distribution, so "
            f"degradation would be measured from the wrong reference"
        )

    data = generate_cvrptw(
        args.test_size, ref.blob["n_customers"], ref.blob["capacity"],
        service_time=ref.blob["service_time"], shift_end=ref.blob["shift_end"],
        tw_half_width=ref.blob["tw_half_width"], seed=args.test_seed, device=device,
    )

    print(f"baseline  ({baselines[0].variant}): {len(baselines)} seeds")
    print(f"candidate ({candidates[0].variant}): {len(candidates)} seeds")
    kind = "structural" if param in ("eta", "field_scale") else "magnitude"
    print(f"  shifting {param} over {values}  ({kind} shift; trained at {trained})")
    print(f"  tardiness_weight {weight} FIXED | {args.test_size} instances "
          f"(seed {args.test_seed}) | {args.scenarios} days | augment={augment}\n")

    # costs[group][value][seed] -> (n_instances,)
    # Every table below needs the full grid, so nothing can be printed until the last
    # evaluation finishes. Progress is reported as it goes -- a long silent run is
    # indistinguishable from a hung one, and this grid can be tens of evaluations.
    total = len(values) * (len(baselines) + len(candidates))
    done = 0
    t0 = time.time()
    print(f"evaluating {total} checkpoint x {param} combinations...", flush=True)

    costs: dict[str, dict[float, list[Tensor]]] = {"base": {}, "cand": {}}
    for group, loaded_list in (("base", baselines), ("cand", candidates)):
        label = loaded_list[0].variant
        for value in values:
            per_seed = []
            for seed_idx, loaded in enumerate(loaded_list):
                cfg = replace(loaded.sim_cfg, **{param: value})
                per_seed.append(evaluate_stoch(
                    loaded.policy, data, cfg, tardiness_weight=weight,
                    n_starts=args.n_starts, augment=augment,
                    eval_batch_size=args.eval_batch_size, scenario_seeds=scenario_seeds,
                )["cost"])
                done += 1
                elapsed = time.time() - t0
                eta_s = elapsed / done * (total - done)
                print(f"  [{done:>3}/{total}] {label} seed {seed_idx + 1} "
                      f"{param}={value:<7g} cost {float(per_seed[-1].mean()):.4f} "
                      f"| {elapsed / 60:.1f} min elapsed, ~{eta_s / 60:.1f} min left",
                      flush=True)
            costs[group][value] = per_seed
    print()

    # ------------------------------------------------------------- 1. per-seed means
    print("PER-SEED MEAN COST  (raw material -- does every seed point the same way?)")
    width = 10 * max(len(baselines), len(candidates))
    print(f"{param:>9} {'variant':>9} {'per seed':>{width}} {'mean':>9} {'sd':>8}")
    print("-" * (9 + 10 + width + 19))
    for beta_var in values:
        for group, label in (("base", baselines[0].variant),
                             ("cand", candidates[0].variant)):
            means = [float(c.mean()) for c in costs[group][beta_var]]
            mean, _, sd = _interval(means)
            cells = "".join(f"{m:>10.4f}" for m in means)
            print(f"{beta_var:>9.3f} {label:>9} {cells:>{width}} {mean:>9.4f} {sd:>8.4f}")

    # ------------------------------------------------- 2. seed-averaged paired (headline)
    print("\nSEED-AVERAGED PAIRED COST  (each variant averaged over its seeds, then")
    print("compared instance by instance; negative = candidate cheaper)")
    header = f"{param:>9} {'baseline':>10} {'candidate':>10} {'diff':>9} {'95% CI':>21}"
    print(header)
    print("-" * (len(header) + 20))
    avg: dict[str, dict[float, Tensor]] = {"base": {}, "cand": {}}
    for beta_var in values:
        for group in ("base", "cand"):
            avg[group][beta_var] = torch.stack(costs[group][beta_var]).mean(dim=0)
        diff = (avg["cand"][beta_var] - avg["base"][beta_var]).double()
        n = diff.numel()
        mean = float(diff.mean())
        se = float(diff.std(unbiased=True)) / math.sqrt(n)
        lo, hi = mean - _Z95 * se, mean + _Z95 * se
        flag = "  significant" if (lo > 0 or hi < 0) else "  (not significant)"
        print(f"{beta_var:>9.3f} {float(avg['base'][beta_var].mean()):>10.4f} "
              f"{float(avg['cand'][beta_var].mean()):>10.4f} {mean:>+9.4f} "
              f"[{lo:>+8.4f}, {hi:>+8.4f}]{flag}")

    id_beta = values[0]
    if len(values) > 1:
        print(f"\nSEED-AVERAGED PAIRED DEGRADATION from {param}={id_beta} "
              f"(THE THESIS CLAIM)")
        print(header)
        print("-" * (len(header) + 20))
        for beta_var in values[1:]:
            b_deg = avg["base"][beta_var] - avg["base"][id_beta]
            c_deg = avg["cand"][beta_var] - avg["cand"][id_beta]
            diff = (c_deg - b_deg).double()
            n = diff.numel()
            mean = float(diff.mean())
            se = float(diff.std(unbiased=True)) / math.sqrt(n)
            lo, hi = mean - _Z95 * se, mean + _Z95 * se
            flag = "  significant" if (lo > 0 or hi < 0) else "  (not significant)"
            print(f"{beta_var:>9.3f} {float(b_deg.mean()):>10.4f} "
                  f"{float(c_deg.mean()):>10.4f} {mean:>+9.4f} "
                  f"[{lo:>+8.4f}, {hi:>+8.4f}]{flag}")

    # ------------------------------------------------------- 3. across-seed variability
    n_pairs = min(len(baselines), len(candidates))
    print(f"\nACROSS-SEED SPREAD  (difference of means computed separately per seed pair;")
    print(f"with {n_pairs} seeds the interval is wide by construction -- read the SIGN)")
    print(f"{param:>9} {'per-seed diff':>{width}} {'mean':>9} {'sd':>8} "
          f"{'95% CI (t)':>22} {'all agree':>10}")
    print("-" * (9 + width + 52))
    for beta_var in values:
        diffs = [float(costs["cand"][beta_var][i].mean()
                       - costs["base"][beta_var][i].mean()) for i in range(n_pairs)]
        mean, half, sd = _interval(diffs)
        cells = "".join(f"{d:>+10.4f}" for d in diffs)
        agree = "yes" if all(d < 0 for d in diffs) or all(d > 0 for d in diffs) else "NO"
        ci = (f"[{mean - half:>+8.4f}, {mean + half:>+8.4f}]"
              if math.isfinite(half) else "n/a (1 seed)")
        print(f"{beta_var:>9.3f} {cells:>{width}} {mean:>+9.4f} {sd:>8.4f} "
              f"{ci:>22} {agree:>10}")

    print("\n  'all agree' = every seed pair shows the same sign. With a handful of seeds")
    print("  that consistency, plus the seed-averaged interval above, is the honest claim;")
    print("  a t-interval on 3 numbers is not a strong test on its own.")


if __name__ == "__main__":
    main()
