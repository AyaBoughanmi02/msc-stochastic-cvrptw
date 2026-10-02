"""WHY does the gate hurt at decision time? Split its effect into distance and lateness.

Example (PowerShell, one line):
    python -u -m scripts.gate_components --ckpt checkpoints\\Dp_w1_15k_s2024_last.pt
        checkpoints\\Dp_w1_15k_s7_last.pt checkpoints\\Dp_w1_15k_s1234_last.pt
        --out runs\\gate_components_w1

Inference only. For each D-prime checkpoint, the gate strength lambda is FORCED to each
value on a grid (0 and the strength each seed learned are always included) and the
selected route's cost is split into its parts, paired against lambda = 0 on the same
instances and the same simulated days.

THE HYPOTHESIS UNDER TEST ("double counting"). The policy was trained on a reward that
already prices expected lateness, so its routes already carry the caution the reward
asked for. Adding lambda * sigma^2 at decision time prices the same risk a SECOND time.
Predicted signature, fixed before running:
  * distance RISES with lambda (detours around risky roads);
  * the lateness penalty FALLS, but by LESS than distance rises;
  -> net cost rises, i.e. the decision-time gate buys safety at more than it is worth.
If instead lateness does NOT fall, the gate is not even buying safety: it is mis-aimed.

Outputs: <out>.log (console copy), <out>.json (all numbers, per seed).
Run from the repo root so `src` and `scripts` are importable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from scripts.bn_mechanism import _tee_console
from scripts.compare_variants import forced_gate, paired_difference
from src.data.cvrptw_generator import generate_cvrptw
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import evaluate_stoch

_KEYS = ("cost", "distance", "tardiness", "late_count")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decision-time gate: distance vs lateness")
    p.add_argument("--ckpt", type=str, nargs="+", required=True,
                   help="D-prime checkpoints (_last), one per seed")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.9])
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", type=str, default="runs/gate_components")
    add_budget_argument(p)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)

    loaded = [load_stoch_policy(p, device) for p in args.ckpt]
    for path, l in zip(args.ckpt, loaded):
        if not hasattr(l.policy, "gate_w"):
            raise ValueError(f"{path} (variant {l.variant}) has no gate to force")
    assert_matched_budget([(p, l.blob) for p, l in zip(args.ckpt, loaded)],
                          allow=args.allow_budget_mismatch)
    ref = loaded[0].blob
    weight = float(ref["tardiness_weight"])
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    learned = [float(l.policy.gate_strength().detach()) for l in loaded]
    print(f"learned lambda per seed: {', '.join(f'{x:.4f}' for x in learned)}")

    # results[seed][beta][lambda] -> {key: (n_inst,) tensor}
    results: list[dict[float, dict[float, dict[str, torch.Tensor]]]] = []
    t0 = time.time()
    total = len(loaded) * len(args.beta_var) * (len(set(args.lambdas)) + 1)
    done = 0
    for seed_idx, l in enumerate(loaded):
        grid = sorted(set(args.lambdas) | {0.0, round(learned[seed_idx], 4)})
        per_beta: dict[float, dict[float, dict[str, torch.Tensor]]] = {}
        for beta in args.beta_var:
            per_beta[beta] = {}
            for lam in grid:
                with forced_gate(l.policy, lam):
                    out = evaluate_stoch(
                        l.policy, data, replace(l.sim_cfg, beta_var=beta),
                        tardiness_weight=weight, augment=not args.no_augment,
                        eval_batch_size=args.eval_batch_size,
                        scenario_seeds=tuple(range(args.scenarios)),
                    )
                per_beta[beta][lam] = {k: out[k].double().cpu() for k in _KEYS}
                done += 1
                print(f"  [{done:>3}/{total}] seed {seed_idx + 1} beta {beta:g} "
                      f"lambda {lam:<7.4g} cost {float(out['cost'].mean()):.4f} "
                      f"| {(time.time() - t0) / 60:.1f} min", flush=True)
        results.append(per_beta)

    print("\nDECISION-TIME GATE, paired against lambda = 0 (seed-averaged; per-seed in json)")
    print("  penalty = tardiness_weight * (late_count + tardiness)")
    head = (f"{'beta':>5} {'lambda':>7} {'cost':>8} {'d cost':>8} {'d dist':>8} "
            f"{'d penalty':>10} {'d late':>8}   paired d cost 95% CI")
    print(head)
    print("-" * (len(head) + 8))
    summary: dict = {"learned_lambda": learned, "rows": []}
    for beta in args.beta_var:
        common = sorted(set.intersection(*[set(r[beta]) for r in results]))
        for lam in common:
            def avg(key: str, lam_: float) -> torch.Tensor:
                return torch.stack([r[beta][lam_][key] for r in results]).mean(dim=0)
            d = {k: avg(k, lam) - avg(k, 0.0) for k in _KEYS}
            d_pen = weight * (d["late_count"] + d["tardiness"])
            p = paired_difference(avg("cost", lam), avg("cost", 0.0))
            per_seed = [float((r[beta][lam]["cost"] - r[beta][0.0]["cost"]).mean())
                        for r in results]
            row = {"beta": beta, "lambda": lam, "cost": float(avg("cost", lam).mean()),
                   "d_cost": float(d["cost"].mean()), "d_distance": float(d["distance"].mean()),
                   "d_penalty": float(d_pen.mean()), "d_late": float(d["late_count"].mean()),
                   "ci": [p["lo"], p["hi"]], "per_seed_d_cost": per_seed}
            summary["rows"].append(row)
            print(f"{beta:>5g} {lam:>7.3g} {row['cost']:>8.4f} {row['d_cost']:>+8.4f} "
                  f"{row['d_distance']:>+8.4f} {row['d_penalty']:>+10.4f} "
                  f"{row['d_late']:>+8.4f}   [{p['lo']:+.4f}, {p['hi']:+.4f}]")
    # The learned values differ per seed, so they are averaged over seeds separately.
    for beta in args.beta_var:
        own = [float((r[beta][round(learned[i], 4)]["cost"] - r[beta][0.0]["cost"]).mean())
               for i, r in enumerate(results)]
        print(f"  beta {beta:g}: cost change at each seed's OWN learned lambda "
              f"(the training condition): mean {statistics.fmean(own):+.4f}, per seed "
              + " ".join(f"{x:+.4f}" for x in own))
        summary.setdefault("own_learned", {})[str(beta)] = own

    print("\n  Reading (fixed in advance): double counting <=> d dist > 0 and "
          "-d penalty < d dist.")
    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda x: None if isinstance(x, float)
                  and not math.isfinite(x) else x)
    print(f"  wrote {args.out}.json")


if __name__ == "__main__":
    main()