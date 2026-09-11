"""Check a trained stochastic CVRPTW checkpoint: gate chain, OOD shift, calibration, saturation.

Example:
    python -m scripts.check_trained --checkpoint checkpoints/cvrptwB.pt --ood-beta-var 0.6 0.9

Runs the four things that have to be true before a checkpoint is used as evidence, and
prints them as one table you can read into the thesis:

  1. GATE CHAIN -- every term stacked on top of variant A is still inert when switched off
     (bank off => certified env; zeroed w_edge => variant-A pointer; empty gate slot =>
     variant B). Training cannot have drifted the reduction property.
  2. ID / OOD -- the same checkpoint evaluated under its training variance and under
     shifted ones, with FIXED scenario seeds so every row and every variant faces the same
     days. This is the load-bearing measurement; variant D's whole claim is that its
     degradation from ID to OOD is smaller than variant B's.
  3. CALIBRATION (level 2) -- the penalty share under the TRAINED policy's own routing.
     The greedy proxy used before training overstates lateness, so the share here is
     lower. If it has collapsed to near zero, the weight was effectively invisible during
     training and the run is not a clean test of the reward it claims to optimise.
  4. SATURATION -- how much of the pointer's signal sits in tanh's flat region, overall and
     next to tight windows. This is the evidence for or against `--edge-bias-mode
     post_tanh`; a negligible number means the pre-tanh placement costs nothing.

Everything is measured on a FRESH test set by default (--test-seed), not on the validation
set that selected the checkpoint -- reporting a headline number on the set used for model
selection is optimistic.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.envs.cvrptw_stoch_diagnostics import (
    measure_pointer_saturation,
    policy_tardiness_stats,
)
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import evaluate_stoch
from src.models.gate_checks import assert_gate_chain_inert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a trained stochastic CVRPTW checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ood-beta-var", type=float, nargs="*", default=[0.6, 0.9],
                   help="shifted variance levels to evaluate; empty = ID only")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242,
                   help="fresh instances, NOT the validation seed used for model selection")
    p.add_argument("--scenarios", type=int, default=4,
                   help="independent days averaged per row (fixed seeds => paired rows)")
    p.add_argument("--n-starts", type=int, default=None, help="None = all customers")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true", help="skip 8x augmentation (faster)")
    p.add_argument("--skip-saturation", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    policy, sim_cfg, blob, variant = load_stoch_policy(args.checkpoint, device)
    weight = float(blob["tardiness_weight"])

    data = generate_cvrptw(
        args.test_size, blob["n_customers"], blob["capacity"],
        service_time=blob["service_time"], shift_end=blob["shift_end"],
        tw_half_width=blob["tw_half_width"], seed=args.test_seed, device=device,
    )
    scenario_seeds = tuple(range(args.scenarios))
    augment = not args.no_augment

    print(f"checkpoint {args.checkpoint}  (variant {variant})")
    print(f"  trained at beta_var={sim_cfg.beta_var} alpha={sim_cfg.alpha} "
          f"eta={sim_cfg.eta} K={sim_cfg.n_bins} | tardiness_weight={weight}")
    print(f"  stored val_cost {blob.get('val_cost', float('nan')):.4f}")
    print(f"  test: {args.test_size} fresh instances (seed {args.test_seed}), "
          f"{args.scenarios} days, augment={augment}\n")

    # --- 1. gate chain -------------------------------------------------------------
    assert_gate_chain_inert(policy, data, sim_cfg, seed=0)
    print("gate chain: PASS (bank off => certified env; zeroed edge term => variant-A "
          "pointer; empty gate slot => variant B)\n")

    # --- 2. ID / OOD ---------------------------------------------------------------
    def row(cfg: SimulatorConfig) -> dict[str, float]:
        out = evaluate_stoch(
            policy, data, cfg, tardiness_weight=weight, n_starts=args.n_starts,
            augment=augment, eval_batch_size=args.eval_batch_size,
            scenario_seeds=scenario_seeds,
        )
        return {k: float(v.mean()) for k, v in out.items()}

    header = (f"{'beta_var':>9} {'cost':>9} {'distance':>9} {'tard':>8} {'n_late':>8} "
              f"{'d_cost':>9} {'d_cost_%':>9}")
    print("ID / OOD  (same checkpoint, same days, tardiness_weight held FIXED)")
    print(header)
    print("-" * len(header))
    id_stats = row(sim_cfg)
    print(f"{sim_cfg.beta_var:>9.3f} {id_stats['cost']:>9.4f} {id_stats['distance']:>9.4f} "
          f"{id_stats['tardiness']:>8.4f} {id_stats['late_count']:>8.3f} "
          f"{'--':>9} {'(ID)':>9}")
    for beta_var in args.ood_beta_var:
        ood = row(replace(sim_cfg, beta_var=beta_var))
        delta = ood["cost"] - id_stats["cost"]
        print(f"{beta_var:>9.3f} {ood['cost']:>9.4f} {ood['distance']:>9.4f} "
              f"{ood['tardiness']:>8.4f} {ood['late_count']:>8.3f} "
              f"{delta:>+9.4f} {100.0 * delta / id_stats['cost']:>+8.2f}%")
    print("\n  d_cost is the degradation carried from the training variance to the shifted")
    print("  one. It is the quantity the variance gate is supposed to shrink: compare it")
    print("  between variants B and D, not against any absolute scale.\n")

    # --- 3. calibration under the trained policy -----------------------------------
    level2 = policy_tardiness_stats(
        policy, data, sim_cfg, n_starts=args.n_starts, seed=0,
    )
    share = next(
        (p.penalty_share for p in level2["profiles"] if abs(p.weight - weight) < 1e-9),
        None,
    )
    print("calibration under the TRAINED policy (level 2)")
    print(f"  mean distance {level2['mean_distance']:.4f} "
          f"({level2['mean_distance_per_customer']:.4f}/customer) | "
          f"tardiness {level2['mean_tardiness']:.4f} | "
          f"violations {level2['mean_late_count']:.3f} | "
          f"late_rate {level2['late_rate']:.2f}")
    if share is None:
        print(f"  trained weight {weight:g} was not on the swept grid; "
              f"level-2 recommendation {level2['recommended_weight']:g}")
    else:
        print(f"  penalty share at the trained weight {weight:g}: {share:.1%} "
              f"| level-2 recommendation {level2['recommended_weight']:g}")
        if share < 0.05:
            print("  WARNING: the penalty is under 5% of cost under this policy -- the "
                  "reward it optimised was effectively distance-only. Treat the run as a "
                  "weak test of the tardiness objective and consider a larger weight.")
        elif share > 0.5:
            print("  WARNING: the penalty is over 50% of cost -- distance has stopped "
                  "driving the route. Consider a smaller weight.")
    print()

    # --- 4. pointer saturation ------------------------------------------------------
    if not args.skip_saturation:
        sat = measure_pointer_saturation(
            policy, data, sim_cfg, n_starts=args.n_starts, seed=0,
        )
        print("pointer saturation (|u| > 8, feasible candidates only)")
        print(f"  overall {sat['saturated_frac']:.4f} | "
              f"near a tight window {sat['saturated_frac_near_tight_window']:.4f} | "
              f"mean |u| {sat['mean_abs_u']:.3f} | "
              f"share of candidates near a tight window {sat['frac_near_tight']:.3f}")
        if sat["saturated_frac_near_tight_window"] < 0.01:
            print("  -> pre-tanh placement costs essentially nothing here; "
                  "--edge-bias-mode post_tanh is not needed.")
        else:
            print("  -> a non-trivial share of tight-window candidates sits in tanh's flat "
                  "region; --edge-bias-mode post_tanh is worth testing for variant D.")

    # Policy-owned scalars: variant D reports softplus(w) here, variant B nothing.
    metrics = getattr(policy, "policy_metrics", None)
    if callable(metrics) and metrics():
        print("\npolicy metrics: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics().items()))


if __name__ == "__main__":
    main()