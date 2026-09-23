"""Tail-risk comparison of two trained variants: does the gate reshape the UPPER TAIL?

WHY THIS EXISTS. The variance gate is a risk mechanism: penalising high-sigma^2 edges is
meant to avoid BAD REALISATIONS, not to lower the average. Every comparison so far
averaged the cost over simulated days and then over instances, which is precisely how a
tail effect is erased. Herrera et al. (risk-aware CVRPTW under persistent disruptions)
report the same asymmetry: temporal persistence raises upper-tail tardiness while mean
impact is controlled, and a CVaR-aligned objective is what moves it.

So this script keeps the PER-DAY distribution instead of collapsing it, and reports, per
instance, the mean, the q-quantile, the CVaR (mean of the worst (1-q) fraction of days)
and the worst day, for cost and for tardiness.

PREREGISTERED HYPOTHESIS (H1): at equal mean cost, the gated variant has a lighter upper
tail than the baseline, and the gap grows with beta_var. A negative, significant paired
difference on CVaR(cost) supports it; a flat or positive one refutes it.

WHAT IS PAIRED WITH WHAT. Both variants see the SAME instances and the SAME realised days
(scenario seeds are fixed), so differences are taken instance by instance and the
instance difficulty cancels -- the same argument as scripts/compare_variants.py. With
several checkpoints per variant, each variant's per-instance metric is averaged over its
seeds first (the seed-averaged estimator of scripts/seed_study.py).

RESOLUTION. CVaR at q needs enough days to be meaningful: with 4 days, CVaR(0.9) IS the
worst day. Use --scenarios 32 or more; the run is linear in the number of days.

Example:
    python -m scripts.tail_risk --baseline checkpoints/B_fix15k_s2024.pt \
        --candidate checkpoints/D_fix15k_s2024.pt --beta-var 0.3 0.6 0.9 \
        --scenarios 32 --no-augment --eval-batch-size 16

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig, draw_noise
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import _one_scenario
from scripts.compare_variants import paired_difference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tail-risk comparison of two variants")
    p.add_argument("--baseline", type=str, nargs="+", required=True,
                   help="one checkpoint per training seed, e.g. variant B")
    p.add_argument("--candidate", type=str, nargs="+", required=True)
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=32,
                   help="independent days per instance; CVaR needs >= ~20 to mean anything")
    p.add_argument("--quantile", type=float, default=0.9,
                   help="tail level q: CVaR(q) averages the worst (1-q) fraction of days")
    p.add_argument("--n-starts", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def per_day_costs(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float,
    n_starts: int | None,
    augment: bool,
    eval_batch_size: int,
    n_scenarios: int,
) -> dict[str, Tensor]:
    """Evaluate WITHOUT averaging over days. Returns (S, B) tensors per key.

    Mirrors evaluate_stoch exactly -- same selection rule, same shared noise field per
    day, same chunking -- but keeps the day axis instead of collapsing it. Scenario seeds
    are 0..S-1, so two variants evaluated here face identical days.
    """
    b, n, _ = data.coords.shape
    starts = (n - 1) if n_starts is None else n_starts
    chunk = min(eval_batch_size, b)
    dev = data.coords.device
    out: dict[str, list[Tensor]] = {}

    for seed in range(n_scenarios):
        noise = draw_noise(
            (b, n, n, sim_cfg.n_bins), device=dev, dtype=data.coords.dtype,
            generator=torch.Generator(device=dev).manual_seed(seed),
        )                                                            # (B, N, N, K)
        pieces: list[dict[str, Tensor]] = []
        for i in range(0, b, chunk):
            sl = slice(i, i + chunk)
            piece = CVRPTWData(
                coords=data.coords[sl], demand=data.demand[sl], capacity=data.capacity,
                time_windows=data.time_windows[sl], service_time=data.service_time[sl],
                shift_end=data.shift_end,
            )
            pieces.append(
                _one_scenario(policy, piece, sim_cfg, tardiness_weight, starts,
                              augment, noise[sl])
            )
        for key in pieces[0]:
            out.setdefault(key, []).append(torch.cat([p[key] for p in pieces], dim=0))
    return {k: torch.stack(v, dim=0).cpu() for k, v in out.items()}   # (S, B)


def tail_metrics(per_day: Tensor, q: float) -> dict[str, Tensor]:
    """Per-instance summaries of the day distribution. per_day: (S, B) -> (B,) each.

    cvar = mean of the worst ceil((1-q)*S) days, i.e. the average cost of a bad day
    rather than the single worst one (which is noisier and depends on S).
    """
    s = per_day.size(0)
    k = max(1, math.ceil((1.0 - q) * s))
    worst_k = per_day.topk(k, dim=0).values                          # (k, B)
    return {
        "mean": per_day.mean(dim=0),                                 # (B,)
        "quantile": per_day.quantile(q, dim=0),                      # (B,)
        "cvar": worst_k.mean(dim=0),                                 # (B,)
        "worst": per_day.max(dim=0).values,                          # (B,)
    }


def _mark(p: dict) -> str:
    return "  significant" if p["significant"] else "  (not significant)"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    augment = not args.no_augment
    baselines = [load_stoch_policy(p, device) for p in args.baseline]
    candidates = [load_stoch_policy(p, device) for p in args.candidate]
    ref = baselines[0]

    weight = float(ref.blob["tardiness_weight"])
    for loaded in baselines + candidates:
        if float(loaded.blob["tardiness_weight"]) != weight:
            raise ValueError("checkpoints disagree on tardiness_weight -- not comparable")

    data = generate_cvrptw(
        args.test_size, ref.blob["n_customers"], ref.blob["capacity"],
        service_time=ref.blob["service_time"], shift_end=ref.blob["shift_end"],
        tw_half_width=ref.blob["tw_half_width"], seed=args.test_seed, device=device,
    )

    q = args.quantile
    tail_k = max(1, math.ceil((1.0 - q) * args.scenarios))
    print(f"baseline  ({baselines[0].variant}): {len(baselines)} seeds")
    print(f"candidate ({candidates[0].variant}): {len(candidates)} seeds")
    print(f"  {args.test_size} instances (seed {args.test_seed}) | {args.scenarios} days "
          f"| augment={augment} | tardiness_weight {weight} FIXED")
    print(f"  CVaR({q:g}) = mean of each instance's {tail_k} worst days out of "
          f"{args.scenarios}\n")

    total = (len(baselines) + len(candidates)) * len(args.beta_var)
    done = 0
    results: dict[tuple[str, float], dict[str, dict[str, Tensor]]] = {}
    for group, loaded_list in (("base", baselines), ("cand", candidates)):
        for beta_var in args.beta_var:
            per_seed_cost: list[dict[str, Tensor]] = []
            per_seed_tard: list[dict[str, Tensor]] = []
            late_rate: list[Tensor] = []
            for loaded in loaded_list:
                cfg = replace(loaded.sim_cfg, beta_var=beta_var)
                days = per_day_costs(
                    loaded.policy, data, cfg, weight, args.n_starts, augment,
                    args.eval_batch_size, args.scenarios,
                )                                                    # (S, B) each
                per_seed_cost.append(tail_metrics(days["cost"], q))
                per_seed_tard.append(tail_metrics(days["tardiness"], q))
                late_rate.append((days["late_count"] > 0).to(torch.float32).mean(dim=0))
                done += 1
                print(f"\r  evaluated {done}/{total} (checkpoint x beta_var)",
                      end="", flush=True)
            # Seed-averaged per-instance metrics (the seed_study estimator).
            results[(group, beta_var)] = {
                "cost": {k: torch.stack([m[k] for m in per_seed_cost]).mean(0)
                         for k in per_seed_cost[0]},
                "tard": {k: torch.stack([m[k] for m in per_seed_tard]).mean(0)
                         for k in per_seed_tard[0]},
                "late_rate": torch.stack(late_rate).mean(0),
            }
    print("\r" + " " * 60 + "\r", end="")

    b_label, c_label = baselines[0].variant, candidates[0].variant
    for field, title in (("cost", "PENALISED COST"), ("tard", "TARDINESS")):
        print(f"{title} -- day distribution per instance, averaged over instances")
        head = (f"{'beta_var':>9} {'variant':>8} {'mean':>10} {f'q{q:g}':>10} "
                f"{'CVaR':>10} {'worst':>10}")
        print(head)
        print("-" * len(head))
        for beta_var in args.beta_var:
            for group, label in (("base", b_label), ("cand", c_label)):
                m = results[(group, beta_var)][field]
                print(f"{beta_var:>9.3f} {label:>8} {float(m['mean'].mean()):>10.4f} "
                      f"{float(m['quantile'].mean()):>10.4f} "
                      f"{float(m['cvar'].mean()):>10.4f} "
                      f"{float(m['worst'].mean()):>10.4f}")
        print()

    print("PAIRED TAIL DIFFERENCE  (candidate - baseline, per instance; negative = "
          "candidate has the lighter tail)")
    print("  H1: the gate buys TAIL robustness at equal MEAN cost.")
    head = (f"{'beta_var':>9} {'metric':>12} {'baseline':>10} {'candidate':>10} "
            f"{'diff':>9} {'95% CI':>20}")
    print(head)
    print("-" * (len(head) + 4))
    for beta_var in args.beta_var:
        base = results[("base", beta_var)]
        cand = results[("cand", beta_var)]
        rows = (
            ("cost mean", base["cost"]["mean"], cand["cost"]["mean"]),
            (f"cost CVaR", base["cost"]["cvar"], cand["cost"]["cvar"]),
            ("cost worst", base["cost"]["worst"], cand["cost"]["worst"]),
            (f"tard CVaR", base["tard"]["cvar"], cand["tard"]["cvar"]),
            ("late rate", base["late_rate"], cand["late_rate"]),
        )
        for name, b_vals, c_vals in rows:
            p = paired_difference(c_vals, b_vals)
            print(f"{beta_var:>9.3f} {name:>12} {float(b_vals.mean()):>10.4f} "
                  f"{float(c_vals.mean()):>10.4f} {p['mean']:>+9.4f} "
                  f"[{p['lo']:>+8.4f}, {p['hi']:>+8.4f}]{_mark(p)}")
        print()

    print("  Reading: a negative CVaR difference with a mean difference near zero is the")
    print("  H1 signature -- same average, fewer bad days. A negative mean AND CVaR is a")
    print("  plain win; a positive CVaR refutes H1 at that variance level.")


if __name__ == "__main__":
    main()