"""Calibrate --tardiness-weight before training a stochastic CVRPTW variant.

Example:
    python scripts/calibrate_tardiness.py --n-customers 20 --beta-var 0.3

`tardiness_weight` prices a time-window violation against tour distance in the reward.
Too small and lateness is invisible to the gradient (the problem collapses back to the
deterministic variant A); too large and distance stops mattering (the policy learns timid,
short routes). The readable dial is the PENALTY SHARE -- the fraction of total cost that
lateness accounts for. This script routes the instances with a greedy-feasible proxy,
measures the lateness that produces, then prices it across a grid of candidate weights.
One rollout plus arithmetic: no training, seconds to run.

Read the output as follows:
  * time_only vs capacity_only  -- confirms the instances are still TIME-binding under the
    ripple (Stage 2 measured ~0.275 vs ~0.024). If capacity dominates, the whole premise
    is wrong and the dials need revisiting before anything is trained.
  * recommended_weight          -- the swept weight landing closest to --target-share.
  * grid saturated              -- if True the recommendation is pinned to an END of the
    grid, which means the target lies OUTSIDE the range swept: widen --weights and re-run.
    A boundary value is not a recommendation.

The proxy makes no attempt to avoid lateness, so it OVERSTATES what a trained policy will
incur; the weight it suggests is a lower bound. After training, re-check with
policy_tardiness_stats on the checkpoint.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig, assert_ripple_reachable
from src.envs.cvrptw_stoch_diagnostics import DEFAULT_TARGET_SHARE, stoch_calibration_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate the CVRPTW tardiness weight")
    # problem (the locked, calibrated dials)
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--capacity", type=int, default=20)
    p.add_argument("--shift-end", type=float, default=4.0)
    p.add_argument("--tw-half-width", type=float, default=0.5)
    p.add_argument("--service-time", type=float, default=0.2)
    # simulator
    p.add_argument("--n-bins", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3],
                   help="one or more variance levels; sweep several to pick the ID regime")
    p.add_argument("--eta", type=float, default=0.6)
    p.add_argument("--field-scale", type=float, default=0.3)
    p.add_argument("--phase-k", type=float, default=1.0)
    # sweep
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    p.add_argument("--target-share", type=float, default=DEFAULT_TARGET_SHARE,
                   help="fraction of total cost the penalty should account for")
    # run
    p.add_argument("--batch-size", type=int, default=128, help="instances to measure over")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data = generate_cvrptw(
        args.batch_size, args.n_customers, args.capacity,
        service_time=args.service_time, shift_end=args.shift_end,
        tw_half_width=args.tw_half_width, seed=args.seed, device=args.device,
    )
    print(f"{args.batch_size} instances | n={args.n_customers} cap={args.capacity} "
          f"shift={args.shift_end} tw_half={args.tw_half_width} "
          f"service={args.service_time} | alpha={args.alpha} eta={args.eta} "
          f"K={args.n_bins} | target share {args.target_share:.0%}\n")

    header = (f"{'beta_var':>9} {'time_only':>10} {'cap_only':>9} {'distance':>9} "
              f"{'tard':>7} {'n_late':>7} {'late_rate':>10} {'w*':>7} {'saturated':>10}")
    print(header)
    print("-" * len(header))

    results = []
    for beta_var in args.beta_var:
        sim_cfg = SimulatorConfig(
            n_bins=args.n_bins, period=args.shift_end, alpha=args.alpha,
            beta_var=beta_var, phase_k=args.phase_k, shift_end=args.shift_end,
            eta=args.eta, field_scale=args.field_scale,
        )
        assert_ripple_reachable(sim_cfg, args.service_time)
        stats = stoch_calibration_stats(
            data, sim_cfg, weights=args.weights,
            target_share=args.target_share, seed=args.seed,
        )
        results.append((beta_var, stats))
        print(f"{beta_var:>9.3f} {stats['time_only_rate']:>10.3f} "
              f"{stats['capacity_only_rate']:>9.3f} {stats['mean_distance']:>9.3f} "
              f"{stats['mean_tardiness']:>7.3f} {stats['mean_late_count']:>7.2f} "
              f"{stats['late_rate']:>10.2f} {stats['recommended_weight']:>7} "
              f"{str(stats['weight_grid_saturated']):>10}")

    for beta_var, stats in results:
        print(f"\npenalty share by weight, beta_var={beta_var}:")
        print("   " + "  ".join(
            f"w={p.weight:g}: {p.penalty_share:.1%}" for p in stats["profiles"]
        ))

    print("\nHow to read this:")
    print("  * time_only must stay well above cap_only (Stage 2: ~0.275 vs ~0.024), or the")
    print("    instances are capacity problems and the variance gate has nothing to manage.")
    print("  * pick the beta_var whose recommended weight is a moderate number and whose")
    print("    tardiness is clearly non-zero: that is the in-distribution training regime.")
    print("  * if 'saturated' is True the target share lies outside the swept grid --")
    print("    widen --weights and re-run; a boundary value is not a recommendation.")
    print("  * the greedy proxy never tries to avoid lateness, so these weights are a")
    print("    LOWER bound; re-check on the trained checkpoint afterwards.")
    for beta_var, stats in results:
        if not stats["weight_grid_saturated"]:
            print(f"\n  suggested: --beta-var {beta_var} "
                  f"--tardiness-weight {stats['recommended_weight']:g}")


if __name__ == "__main__":
    main()