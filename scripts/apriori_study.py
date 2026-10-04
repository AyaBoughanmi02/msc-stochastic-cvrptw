"""Choose the route BEFORE the day: uncertainty-aware vs uncertainty-blind selection.

Example (PowerShell, one line per --arm; seeds in the SAME order in every arm):
    python -u -m scripts.apriori_study
        --arm M0 checkpoints\\M0_w1_15k_s2024_last.pt checkpoints\\M0_w1_15k_s7_last.pt
                 checkpoints\\M0_w1_15k_s1234_last.pt
        --arm M  checkpoints\\Dp_w1_15k_s2024_last.pt ...
        --out runs\\apriori_w1

Inference only. For every arm, seed and variance level it runs src/eval/apriori_selection
and reports three choice rules (see that module):

  hindsight          best realised route (what every earlier table reported; optimistic)
  expected           best EXPECTED cost, computed with this day's variance (uncertainty-aware)
  planned_distance   shortest planned route (uncertainty-blind)

Gates: every arm with a training-time gate is evaluated with lambda forced to --gate
(default 0, as in every evaluation so far). VGT-R arms use --risk-alpha if given (0 =
bias switched off at test), otherwise their learned alpha.

READING (fixed before running):
  * "expected" beats "planned_distance" on every seed, with the gap growing with beta
    => using the variance at test, OUTSIDE the network, helps -- the system is
    uncertainty-aware where the network is not.
  * hindsight - expected = how optimistic the earlier tables were (always <= 0).
  * the model ranking under "expected" is the deployable ranking.

Outputs: <out>.log (console copy), <out>.json (per seed, per rule, per beta).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import torch

from scripts.bn_mechanism import _tee_console
from scripts.compare_variants import forced_gate, paired_difference
from scripts.risk_components import forced_risk
from src.data.cvrptw_generator import generate_cvrptw
from src.eval.apriori_selection import RULES, evaluate_stoch_apriori
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A-priori route selection study")
    p.add_argument("--arm", nargs="+", action="append", required=True,
                   metavar=("NAME", "CKPT"),
                   help="an arm: a name then one checkpoint per seed (same seed order "
                        "in every arm). The FIRST arm is the reference for comparisons")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--gate", type=float, default=0.0,
                   help="lambda forced on every arm that has a training-time gate")
    p.add_argument("--risk-alpha", type=float, default=None,
                   help="alpha forced on VGT-R arms (None = learned)")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", type=str, default="runs/apriori")
    add_budget_argument(p)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _evaluate(loaded, data, beta, args, weight) -> dict[str, dict[str, torch.Tensor]]:
    policy = loaded.policy
    with ExitStack() as stack:
        if hasattr(policy, "gate_w"):
            stack.enter_context(forced_gate(policy, args.gate))
        if hasattr(policy, "risk_w") and args.risk_alpha is not None:
            stack.enter_context(forced_risk(policy, args.risk_alpha))
        out = evaluate_stoch_apriori(
            policy, data, replace(loaded.sim_cfg, beta_var=beta),
            tardiness_weight=weight, augment=not args.no_augment,
            eval_batch_size=args.eval_batch_size,
            scenario_seeds=tuple(range(args.scenarios)),
        )
    return {r: {k: v.double().cpu() for k, v in d.items()} for r, d in out.items()}


def main() -> None:
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)

    arms: list[tuple[str, list]] = []
    for spec in args.arm:
        if len(spec) < 2:
            raise ValueError(f"--arm needs a name and at least one checkpoint, got {spec}")
        arms.append((spec[0], [load_stoch_policy(p, device) for p in spec[1:]]))
    n_seeds = {len(ls) for _, ls in arms}
    if len(n_seeds) != 1:
        raise ValueError("every arm must have the same number of seeds (paired by position)")
    all_ckpts = [(p, l.blob) for spec, (_, ls) in zip(args.arm, arms)
                 for p, l in zip(spec[1:], ls)]
    assert_matched_budget(all_ckpts, allow=args.allow_budget_mismatch)

    ref = arms[0][1][0].blob
    weight = float(ref["tardiness_weight"])
    for _, ls in arms:
        for l in ls:
            if float(l.blob["tardiness_weight"]) != weight:
                raise ValueError("arms disagree on tardiness_weight -- not one problem")
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    print(f"arms: {', '.join(f'{name} ({ls[0].variant}, {len(ls)} seeds)' for name, ls in arms)}")
    print(f"gate lambda forced to {args.gate:g} where present | VGT-R alpha "
          f"{'learned' if args.risk_alpha is None else f'forced to {args.risk_alpha:g}'}")
    print(f"{args.test_size} instances (seed {args.test_seed}) | {args.scenarios} days | "
          f"augment={not args.no_augment} | tardiness_weight {weight}\n")

    # res[arm][beta][seed][rule][key] -> (n_inst,)
    res: dict[str, dict[float, list]] = {}
    total = sum(len(ls) for _, ls in arms) * len(args.beta_var)
    done, t0 = 0, time.time()
    for name, ls in arms:
        res[name] = {}
        for beta in args.beta_var:
            res[name][beta] = []
            for s, l in enumerate(ls):
                res[name][beta].append(_evaluate(l, data, beta, args, weight))
                done += 1
                e = res[name][beta][-1]
                print(f"  [{done:>3}/{total}] {name} seed {s + 1} beta {beta:g} | "
                      + " | ".join(f"{r} {float(e[r]['cost'].mean()):.4f}" for r in RULES)
                      + f" | {(time.time() - t0) / 60:.1f} min", flush=True)

    def avg(name: str, beta: float, rule: str, key: str = "cost") -> torch.Tensor:
        return torch.stack([e[rule][key] for e in res[name][beta]]).mean(dim=0)

    def per_seed(name: str, beta: float, rule: str) -> list[float]:
        return [float(e[rule]["cost"].mean()) for e in res[name][beta]]

    summary: dict = {"arms": [n for n, _ in arms], "rules": list(RULES), "rows": []}

    print("\n1. COST PER CHOICE RULE (seed-averaged; lower is better)")
    head = f"{'arm':>10} {'beta':>5} " + " ".join(f"{r:>17}" for r in RULES)
    print(head)
    print("-" * len(head))
    for name, _ in arms:
        for beta in args.beta_var:
            vals = {r: float(avg(name, beta, r).mean()) for r in RULES}
            print(f"{name:>10} {beta:>5g} " + " ".join(f"{vals[r]:>17.4f}" for r in RULES))
            summary["rows"].append({"arm": name, "beta": beta, "cost": vals,
                                    "per_seed": {r: per_seed(name, beta, r) for r in RULES}})

    print("\n2. DOES THE VARIANCE HELP THE CHOICE?  expected - planned_distance "
          "(negative = the uncertainty-aware choice is cheaper)")
    head = (f"{'arm':>10} {'beta':>5} {'diff':>9} {'95% CI':>21}   "
            f"{'d dist':>8} {'d late':>8}   per-seed diff            all agree")
    print(head)
    print("-" * len(head))
    for name, _ in arms:
        for beta in args.beta_var:
            p = paired_difference(avg(name, beta, "expected"),
                                  avg(name, beta, "planned_distance"))
            dd = float((avg(name, beta, "expected", "distance")
                        - avg(name, beta, "planned_distance", "distance")).mean())
            dl = float((avg(name, beta, "expected", "late_count")
                        - avg(name, beta, "planned_distance", "late_count")).mean())
            ps = [a - b for a, b in zip(per_seed(name, beta, "expected"),
                                        per_seed(name, beta, "planned_distance"))]
            agree = all(x < 0 for x in ps) or all(x > 0 for x in ps)
            print(f"{name:>10} {beta:>5g} {p['mean']:>+9.4f} "
                  f"[{p['lo']:+8.4f}, {p['hi']:+8.4f}]   {dd:>+8.4f} {dl:>+8.4f}   "
                  + " ".join(f"{x:+.4f}" for x in ps) + f"   {'yes' if agree else 'NO'}")

    print("\n3. HOW OPTIMISTIC WERE THE EARLIER TABLES?  hindsight - expected")
    for name, _ in arms:
        gaps = [float((avg(name, b, "hindsight") - avg(name, b, "expected")).mean())
                for b in args.beta_var]
        print(f"{name:>10}: " + "  ".join(f"beta {b:g}: {g:+.4f}"
                                          for b, g in zip(args.beta_var, gaps)))

    ref_name = arms[0][0]
    if len(arms) > 1:
        lo_b, hi_b = args.beta_var[0], args.beta_var[-1]
        print(f"\n4. DEPLOYABLE MODEL RANKING (rule 'expected'), each arm - {ref_name}, "
              f"paired; degradation = cost(beta {hi_b:g}) - cost(beta {lo_b:g})")
        head = (f"{'arm':>10} {'beta':>6} {'arm':>9} {ref_name:>9} {'diff':>9} "
                f"{'95% CI':>21}   per-seed diff            all agree")
        print(head)
        print("-" * len(head))
        for name, _ in arms[1:]:
            for beta in args.beta_var:
                p = paired_difference(avg(name, beta, "expected"),
                                      avg(ref_name, beta, "expected"))
                ps = [a - b for a, b in zip(per_seed(name, beta, "expected"),
                                            per_seed(ref_name, beta, "expected"))]
                agree = all(x < 0 for x in ps) or all(x > 0 for x in ps)
                print(f"{name:>10} {beta:>6g} {float(avg(name, beta, 'expected').mean()):>9.4f} "
                      f"{float(avg(ref_name, beta, 'expected').mean()):>9.4f} {p['mean']:>+9.4f} "
                      f"[{p['lo']:+8.4f}, {p['hi']:+8.4f}]   "
                      + " ".join(f"{x:+.4f}" for x in ps) + f"   {'yes' if agree else 'NO'}")
            deg = lambda n: avg(n, hi_b, "expected") - avg(n, lo_b, "expected")  # noqa: E731
            p = paired_difference(deg(name), deg(ref_name))
            print(f"{name:>10} {'degr.':>6} {float(deg(name).mean()):>9.4f} "
                  f"{float(deg(ref_name).mean()):>9.4f} {p['mean']:>+9.4f} "
                  f"[{p['lo']:+8.4f}, {p['hi']:+8.4f}]")

    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda x: None if isinstance(x, float)
                  and not math.isfinite(x) else x)
    print(f"\n  wrote {args.out}.json")


if __name__ == "__main__":
    main()