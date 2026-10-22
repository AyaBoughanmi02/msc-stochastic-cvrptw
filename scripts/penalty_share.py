"""How much of the cost is lateness? Re-measures the kappa calibration on the FINAL models.

Why: kappa = 2.0 was calibrated on 10 Sept 2026 (commit 7c0b8c4), before the generator fix of
21 Sept, so the "13.7-14.8 % under a trained policy" figure came from a policy trained on a
fixed set of 32 instances. This script measures the same quantity on the current checkpoints,
under the protocol used for every reported result (route chosen by planned distance, then
driven online on the 4 test days).

    share = kappa * (late arrivals + total lateness) / total cost

Inference only. Example (PowerShell, repo root, .venv active):
    python -u -m scripts.penalty_share --ckpt checkpoints\\M0_w1_15k_s2024_last.pt `
        checkpoints\\M0_w1_15k_s7_last.pt checkpoints\\M0_w1_15k_s1234_last.pt `
        2>&1 | Tee-Object -FilePath runs\\penalty_share.log
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.eval.apriori_selection import evaluate_stoch_apriori
from src.eval.checkpoints import load_stoch_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Penalty share of the cost on trained models")
    p.add_argument("--ckpt", type=str, nargs="+", required=True, help="one per seed")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3])
    p.add_argument("--rule", type=str, default="planned_distance")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaded = [load_stoch_policy(path, device) for path in args.ckpt]
    ref = loaded[0].blob
    weight = float(ref["tardiness_weight"])
    for l in loaded:
        if float(l.blob["tardiness_weight"]) != weight:
            raise ValueError("checkpoints disagree on tardiness_weight")
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    print(f"kappa {weight} | rule {args.rule} | {args.test_size} instances "
          f"(seed {args.test_seed}) | {args.scenarios} test days\n")
    print(f"{'beta':>5} {'seed':>5} {'cost':>8} {'distance':>9} {'late':>7} "
          f"{'lateness':>9} {'share':>7}")
    for beta in args.beta_var:
        shares = []
        for s, l in enumerate(loaded):
            if hasattr(l.policy, "gate_w") or hasattr(l.policy, "risk_w"):
                raise ValueError("use mean-only (M0) checkpoints: no gate to switch off here")
            l.policy.eval()
            out = evaluate_stoch_apriori(
                l.policy, data, replace(l.sim_cfg, beta_var=beta),
                tardiness_weight=weight, augment=True,
                eval_batch_size=args.eval_batch_size,
                scenario_seeds=tuple(range(args.scenarios)),
                n_selection_days=1,          # the saa rule is not used here
            )[args.rule]
            cost = float(out["cost"].mean())
            dist = float(out["distance"].mean())
            late = float(out["late_count"].mean())
            tard = float(out["tardiness"].mean())
            share = weight * (late + tard) / cost
            shares.append(share)
            print(f"{beta:>5g} {s + 1:>5} {cost:>8.4f} {dist:>9.4f} {late:>7.4f} "
                  f"{tard:>9.4f} {share:>7.1%}")
        print(f"{beta:>5g}  mean share over seeds: {sum(shares) / len(shares):.1%}\n")


if __name__ == "__main__":
    main()