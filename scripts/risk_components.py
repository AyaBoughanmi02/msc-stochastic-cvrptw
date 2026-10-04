"""WHY does VGT-R degrade under a variance shift? Sweep its risk-bias strength alpha.

Example (PowerShell, one line):
    python -u -m scripts.risk_components --ckpt checkpoints\\VGTR_w1_15k_s2024_last.pt
        checkpoints\\VGTR_w1_15k_s7_last.pt checkpoints\\VGTR_w1_15k_s1234_last.pt
        --out runs\\risk_components_w1

Inference only, nothing is trained or written back. For each VGT-R checkpoint the
training-time sigma^2 gate is forced to 0 (exactly as in every evaluation of M and VGT-R),
and the decoder's risk-bias strength alpha is FORCED to each value on a grid (0 and the
strength each seed learned are always included). The selected route's cost is split into
distance and lateness penalty and compared, paired on the same instances and days,
against the LEARNED alpha -- the condition VGT-R is actually deployed in.

HYPOTHESIS UNDER TEST ("over-reaction"), fixed before running:
  alpha was learned under the analytic reward, where the clock runs on MEAN travel times
  and is never actually late, and for beta_var in [0.15, 0.6]. At evaluation the clock is
  the REALISED time, so on risky days the vehicle is often already behind schedule and
  P(late) is high for many candidates at once; a large alpha then pushes hard in states
  never seen in training.
Predicted signature:
  * at beta 0.3 the learned alpha is (near-)best;
  * at beta 0.9 a SMALLER alpha is cheaper than the learned one, and the gap grows with beta.
If instead the learned alpha is best at every beta, the bias itself is not the problem and
the degradation must come from what the network learned around it.
alpha = 0 is NOT M0 or M: the network was trained WITH the bias, so alpha = 0 removes an
input it relies on. It is included as the "bias switched off" end of the dose-response.

Outputs: <out>.log (console copy), <out>.json (all numbers, per seed).
Run from the repo root so `src` and `scripts` are importable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import torch

from scripts.bn_mechanism import _tee_console
from scripts.compare_variants import forced_gate, paired_difference
from src.data.cvrptw_generator import generate_cvrptw
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import evaluate_stoch

_KEYS = ("cost", "distance", "tardiness", "late_count")


@contextmanager
def forced_risk(policy, strength: float) -> Iterator[None]:
    """Temporarily override alpha = softplus(risk_w) at inference. Restores on exit.

    Patches the method, not the parameter (same design as compare_variants.forced_gate),
    so alpha = 0 exactly is reachable and the checkpoint is never modified.
    """
    original = policy.risk_strength
    value = torch.tensor(float(strength), device=policy.risk_w.device,
                         dtype=policy.risk_w.dtype)
    policy.risk_strength = lambda: value
    try:
        yield
    finally:
        policy.risk_strength = original


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGT-R: risk-bias dose-response under shift")
    p.add_argument("--ckpt", type=str, nargs="+", required=True,
                   help="VGT-R checkpoints (_last), one per seed")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0])
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", type=str, default="runs/risk_components")
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
        if not hasattr(l.policy, "risk_w"):
            raise ValueError(f"{path} (variant {l.variant}) has no risk bias to force")
    assert_matched_budget([(p, l.blob) for p, l in zip(args.ckpt, loaded)],
                          allow=args.allow_budget_mismatch)
    ref = loaded[0].blob
    weight = float(ref["tardiness_weight"])
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    learned = [round(float(l.policy.risk_strength().detach()), 4) for l in loaded]
    print(f"learned alpha per seed: {', '.join(f'{x:.4f}' for x in learned)}")
    print("training-time sigma^2 gate FORCED to 0 throughout (as in every evaluation)\n")

    # results[seed][beta][alpha] -> {key: (n_inst,) tensor}
    results: list[dict[float, dict[float, dict[str, torch.Tensor]]]] = []
    grids = [sorted(set(args.alphas) | {0.0, learned[i]}) for i in range(len(loaded))]
    total = sum(len(g) for g in grids) * len(args.beta_var)
    done, t0 = 0, time.time()
    for seed_idx, l in enumerate(loaded):
        per_beta: dict[float, dict[float, dict[str, torch.Tensor]]] = {}
        for beta in args.beta_var:
            per_beta[beta] = {}
            for alpha in grids[seed_idx]:
                with forced_gate(l.policy, 0.0), forced_risk(l.policy, alpha):
                    out = evaluate_stoch(
                        l.policy, data, replace(l.sim_cfg, beta_var=beta),
                        tardiness_weight=weight, augment=not args.no_augment,
                        eval_batch_size=args.eval_batch_size,
                        scenario_seeds=tuple(range(args.scenarios)),
                    )
                per_beta[beta][alpha] = {k: out[k].double().cpu() for k in _KEYS}
                done += 1
                print(f"  [{done:>3}/{total}] seed {seed_idx + 1} beta {beta:g} "
                      f"alpha {alpha:<7.4g} cost {float(out['cost'].mean()):.4f} "
                      f"| {(time.time() - t0) / 60:.1f} min", flush=True)
        results.append(per_beta)

    print("\nRISK-BIAS DOSE-RESPONSE, paired against each seed's LEARNED alpha "
          "(the deployed condition)")
    print("  rows with a fixed alpha are averaged over seeds; negative d cost = cheaper "
          "than deployed VGT-R")
    print("  penalty = tardiness_weight * (late_count + tardiness)")
    head = (f"{'beta':>5} {'alpha':>7} {'cost':>8} {'d cost':>8} {'d dist':>8} "
            f"{'d penalty':>10} {'d late':>8}   paired d cost 95% CI     per-seed d cost")
    print(head)
    print("-" * (len(head) + 4))
    summary: dict = {"learned_alpha": learned, "rows": []}
    common = sorted(set.intersection(*[set(g) for g in grids]))       # fixed grid values
    for beta in args.beta_var:
        def deployed(key: str) -> torch.Tensor:                        # (n_inst,)
            return torch.stack([r[beta][learned[i]][key]
                                for i, r in enumerate(results)]).mean(dim=0)

        def fixed(key: str, a: float) -> torch.Tensor:                 # (n_inst,)
            return torch.stack([r[beta][a][key] for r in results]).mean(dim=0)

        rows = [(f"{a:g}", {k: fixed(k, a) for k in _KEYS}) for a in common]
        rows.append(("learned", {k: deployed(k) for k in _KEYS}))
        for label, vals in rows:
            d = {k: vals[k] - deployed(k) for k in _KEYS}
            d_pen = weight * (d["late_count"] + d["tardiness"])
            p = paired_difference(vals["cost"], deployed("cost"))
            if label == "learned":
                per_seed = [0.0] * len(results)
            else:
                a = float(label)
                per_seed = [float((r[beta][a]["cost"] - r[beta][learned[i]]["cost"]).mean())
                            for i, r in enumerate(results)]
            row = {"beta": beta, "alpha": label, "cost": float(vals["cost"].mean()),
                   "d_cost": float(d["cost"].mean()),
                   "d_distance": float(d["distance"].mean()),
                   "d_penalty": float(d_pen.mean()), "d_late": float(d["late_count"].mean()),
                   "ci": [p["lo"], p["hi"]], "per_seed_d_cost": per_seed}
            summary["rows"].append(row)
            print(f"{beta:>5g} {label:>7} {row['cost']:>8.4f} {row['d_cost']:>+8.4f} "
                  f"{row['d_distance']:>+8.4f} {row['d_penalty']:>+10.4f} "
                  f"{row['d_late']:>+8.4f}   [{p['lo']:+.4f}, {p['hi']:+.4f}]   "
                  + " ".join(f"{x:+.4f}" for x in per_seed))
        print()

    # Which alpha would each beta prefer? (descriptive; never used to tune a reported number)
    print("CHEAPEST alpha per beta (seed-averaged, fixed grid + learned):")
    for beta in args.beta_var:
        cand = [r for r in summary["rows"] if r["beta"] == beta]
        best = min(cand, key=lambda r: r["cost"])
        print(f"  beta {beta:g}: alpha {best['alpha']} (cost {best['cost']:.4f}, "
              f"{best['d_cost']:+.4f} vs deployed)")
    print("\n  Reading (fixed in advance): over-reaction <=> at beta 0.9 some alpha below the "
          "learned one is cheaper than the learned one, by more than at beta 0.3.")
    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda x: None if isinstance(x, float)
                  and not math.isfinite(x) else x)
    print(f"  wrote {args.out}.json")


if __name__ == "__main__":
    main()