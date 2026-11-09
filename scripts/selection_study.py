"""A2/A3 for the CPAIOR paper: how should one candidate be chosen BEFORE the day?

Example (PowerShell, one line):
    python -u -m scripts.selection_study --ckpt checkpoints\\M0_w1_15k_s2024_last.pt
        checkpoints\\M0_w1_15k_s7_last.pt checkpoints\\M0_w1_15k_s1234_last.pt
        --out runs\\selection_M0 2>&1 | Tee-Object -FilePath runs\\selection_M0.log

Inference only. One model family (one checkpoint per seed). For every seed and variance
level it runs every candidate once per day (src/eval/selection_protocols), then applies
all rules on the stored tensors. Before the study it re-runs `evaluate_stoch_apriori` on
the first --self-check instances and stops if the two pipelines disagree.

READING (decision rules fixed before running):
  R1  planned_distance - single < 0, 95% CI excluding 0, all seeds agree, at every beta
      => "an honest choice made before the day beats one greedy route".
  R2  saa_out_K{Kmax} - planned_distance at beta 0.6 and 0.9:
        CI below 0      => "with enough selection days SAA is the best honest rule"
                           (report the smallest K whose CI is below 0)
        CI above 0      => "the planned-distance proxy beats SAA up to K={Kmax}"
        CI contains 0   => "no difference detected between SAA (K={Kmax}) and the proxy"
  R3  in-sample optimism(K) = saa_in_K - saa_out_K. Claim "selecting and reporting on the
      same K days is optimistic, and less so as K grows" only if optimism(K) < 0 for every
      K and |optimism| decreases with K at every beta. The log-log slope over K >= 2 is
      reported as a description, not tested (1/sqrt(K) is only expected when many
      candidates are near-tied, so it is not a fair pass/fail line).
  R4  (A3, offline from the saved tensors) model vs measured hindsight optimism per N.

Outputs: <out>.log (console), <out>.json (per seed, per rule, per beta),
<out>_raw_s{seed}_b{beta}.pt (per-candidate tensors; large, do not commit).
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
from torch import Tensor

from scripts.bn_mechanism import _tee_console
from scripts.compare_variants import forced_gate, paired_difference
from scripts.risk_components import forced_risk
from src.data.cvrptw_generator import generate_cvrptw
from src.eval.apriori_selection import _slice, evaluate_stoch_apriori
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy
from src.eval.selection_protocols import CandidateCosts, apply_rules, collect_candidate_costs

_SELF_CHECK_RULES = ("hindsight", "expected", "planned_distance", "average")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A-priori selection protocols (CPAIOR A2/A3)")
    p.add_argument("--ckpt", nargs="+", required=True, help="one checkpoint per seed")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--gate", type=float, default=0.0,
                   help="lambda forced on a model with a training-time gate")
    p.add_argument("--risk-alpha", type=float, default=None)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4, help="test days (seeds 0..D-1)")
    p.add_argument("--select-days", type=int, default=64,
                   help="selection days collected (seeds 10000+k); K values must be <= this")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--ns", type=int, nargs="+", default=[1, 8, 20, 40, 160])
    p.add_argument("--subsets", type=int, default=32, help="random subsets per N")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--k-subset", type=int, default=8, help="K used by the SAA rule inside subsets")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--self-check", type=int, default=16,
                   help="instances re-run through evaluate_stoch_apriori (0 = skip)")
    p.add_argument("--no-save-raw", action="store_true")
    p.add_argument("--out", type=str, default="runs/selection")
    add_budget_argument(p)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _ctx(policy, args: argparse.Namespace) -> ExitStack:
    stack = ExitStack()
    if hasattr(policy, "gate_w"):
        stack.enter_context(forced_gate(policy, args.gate))
    if hasattr(policy, "risk_w") and args.risk_alpha is not None:
        stack.enter_context(forced_risk(policy, args.risk_alpha))
    return stack


def _self_check(loaded, data, beta: float, args, weight: float) -> None:
    """Both pipelines on the same instances and days: rules must agree to float precision."""
    n = min(args.self_check, data.coords.shape[0])
    sub = _slice(data, slice(0, n))
    cfg = replace(loaded.sim_cfg, beta_var=beta)
    kw = dict(tardiness_weight=weight, augment=not args.no_augment,
              eval_batch_size=args.eval_batch_size, scenario_seeds=tuple(range(args.scenarios)))
    with _ctx(loaded.policy, args):
        ref = evaluate_stoch_apriori(loaded.policy, sub, cfg, n_selection_days=8, **kw)
        cc = collect_candidate_costs(loaded.policy, sub, cfg, n_selection_days=8, **kw)
    new = apply_rules(cc, ks=(8,), ns=(), n_subsets=0)
    pairs = [(r, r) for r in _SELF_CHECK_RULES] + [("saa", "saa_out_K8")]
    worst = 0.0
    for old_r, new_r in pairs:
        for k in ("cost", "distance", "tardiness", "late_count"):
            a, b = ref[old_r][k].double().cpu(), new[new_r][k]
            worst = max(worst, float((a - b).abs().max()))
    print(f"self-check on {n} instances, beta {beta:g}: max |difference| = {worst:.2e}")
    if worst > 1e-5:
        raise RuntimeError("selection_protocols disagrees with evaluate_stoch_apriori; "
                           "do not trust the study until this is explained")


def main() -> None:
    args = parse_args()
    if max(args.ks) > args.select_days or args.k_subset > args.select_days:
        raise ValueError("every K must be <= --select-days")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)

    models = [load_stoch_policy(p, device) for p in args.ckpt]
    assert_matched_budget([(p, m.blob) for p, m in zip(args.ckpt, models)],
                          allow=args.allow_budget_mismatch)
    ref = models[0].blob
    weight = float(ref["tardiness_weight"])
    if any(float(m.blob["tardiness_weight"]) != weight for m in models):
        raise ValueError("checkpoints disagree on tardiness_weight -- not one problem")
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    print(f"{len(models)} seeds of {models[0].variant} | {args.test_size} instances "
          f"(seed {args.test_seed}) | {args.scenarios} test days | {args.select_days} "
          f"selection days | K {args.ks} | N {args.ns} x {args.subsets} subsets "
          f"(seed {args.subset_seed}) | augment={not args.no_augment} | tardiness_weight {weight}")

    if args.self_check > 0:
        _self_check(models[0], data, args.beta_var[0], args, weight)

    # res[beta][seed] -> rule -> component -> (B,)
    res: dict[float, list[dict[str, dict[str, Tensor]]]] = {}
    t0 = time.time()
    for beta in args.beta_var:
        res[beta] = []
        for s, m in enumerate(models):
            cfg = replace(m.sim_cfg, beta_var=beta)
            with _ctx(m.policy, args):
                cc: CandidateCosts = collect_candidate_costs(
                    m.policy, data, cfg, tardiness_weight=weight,
                    augment=not args.no_augment, eval_batch_size=args.eval_batch_size,
                    scenario_seeds=tuple(range(args.scenarios)),
                    n_selection_days=args.select_days)
            if not args.no_save_raw:
                torch.save(cc.to_dict(), f"{args.out}_raw_s{s + 1}_b{beta:g}.pt")
            res[beta].append(apply_rules(cc, ks=args.ks, ns=args.ns, n_subsets=args.subsets,
                                         subset_seed=args.subset_seed, k_subset=args.k_subset))
            e = res[beta][-1]
            print(f"  beta {beta:g} seed {s + 1} | single {float(e['single']['cost'].mean()):.4f}"
                  f" | planned_distance {float(e['planned_distance']['cost'].mean()):.4f}"
                  f" | hindsight {float(e['hindsight']['cost'].mean()):.4f}"
                  f" | {(time.time() - t0) / 60:.1f} min", flush=True)

    rules = list(res[args.beta_var[0]][0].keys())

    def avg(beta: float, rule: str) -> Tensor:                       # (B,) seed-averaged
        return torch.stack([e[rule]["cost"] for e in res[beta]]).mean(dim=0)

    def per_seed(beta: float, rule: str) -> list[float]:
        return [float(e[rule]["cost"].mean()) for e in res[beta]]

    summary: dict = {"rules": rules, "ks": args.ks, "ns": args.ns, "rows": []}
    print("\n1. COST PER RULE (seed-averaged; lower is better)")
    for beta in args.beta_var:
        print(f"\n  beta {beta:g}")
        for r in rules:
            print(f"    {r:>28} {float(avg(beta, r).mean()):>9.4f}   per seed "
                  + " ".join(f"{x:.4f}" for x in per_seed(beta, r)))
            summary["rows"].append({"beta": beta, "rule": r, "cost": float(avg(beta, r).mean()),
                                    "per_seed": per_seed(beta, r)})

    def diff_line(beta: float, a: str, b: str) -> dict:
        p = paired_difference(avg(beta, a), avg(beta, b))
        ps = [x - y for x, y in zip(per_seed(beta, a), per_seed(beta, b))]
        agree = all(x < 0 for x in ps) or all(x > 0 for x in ps)
        print(f"    {a:>18} - {b:<18} {p['mean']:>+9.4f} [{p['lo']:+8.4f}, {p['hi']:+8.4f}]   "
              + " ".join(f"{x:+.4f}" for x in ps) + f"   {'yes' if agree else 'NO'}")
        return {"beta": beta, "a": a, "b": b, "mean": p["mean"], "lo": p["lo"], "hi": p["hi"],
                "per_seed": ps, "all_agree": agree}

    kmax = max(args.ks)
    summary["diffs"] = []
    print("\n2. R1  planned_distance - single   |   R2  saa_out_K - planned_distance "
          "(diff, 95% CI, per-seed diff, all agree)")
    for beta in args.beta_var:
        print(f"\n  beta {beta:g}")
        summary["diffs"].append(diff_line(beta, "planned_distance", "single"))
        for k in args.ks:
            summary["diffs"].append(diff_line(beta, f"saa_out_K{k}", "planned_distance"))

    print("\n3. R3  IN-SAMPLE OPTIMISM  saa_in_K - saa_out_K  (seed-averaged)")
    summary["in_sample"] = []
    for beta in args.beta_var:
        opt = [float((avg(beta, f"saa_in_K{k}") - avg(beta, f"saa_out_K{k}")).mean())
               for k in args.ks]
        pts = [(math.log(k), math.log(-o)) for k, o in zip(args.ks, opt) if k >= 2 and o < 0]
        slope = float("nan")
        if len(pts) >= 2:
            mx = sum(x for x, _ in pts) / len(pts)
            my = sum(y for _, y in pts) / len(pts)
            slope = (sum((x - mx) * (y - my) for x, y in pts)
                     / sum((x - mx) ** 2 for x, _ in pts))
        print(f"  beta {beta:g}: " + "  ".join(f"K{k} {o:+.4f}" for k, o in zip(args.ks, opt))
              + f"   | log-log slope (K>=2) {slope:+.3f}")
        if any(o >= 0 for k, o in zip(args.ks, opt) if k >= 2):
            print("    note: some K >= 2 have optimism >= 0; they are left out of the slope")
        summary["in_sample"].append({"beta": beta, "optimism": opt, "slope": slope})

    print("\n4. HINDSIGHT OPTIMISM AND HONEST GAIN vs NUMBER OF CANDIDATES N "
          f"(random subsets, SAA K={args.k_subset})")
    summary["by_n"] = []
    for beta in args.beta_var:
        print(f"\n  beta {beta:g}  {'N':>5} {'hindsight':>10} {'plan_dist':>10} "
              f"{'saa':>10} {'average':>10} {'hind-plan':>10}")
        for m in args.ns:
            v = {r: float(avg(beta, f"N{m}_{r}").mean()) for r in
                 ("hindsight", "planned_distance", f"saa_out_K{args.k_subset}", "average")}
            print(f"  {'':>9} {m:>5} {v['hindsight']:>10.4f} {v['planned_distance']:>10.4f} "
                  f"{v[f'saa_out_K{args.k_subset}']:>10.4f} {v['average']:>10.4f} "
                  f"{v['hindsight'] - v['planned_distance']:>+10.4f}")
            summary["by_n"].append({"beta": beta, "n": m, **v})

    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda x: None if isinstance(x, float)
                  and not math.isfinite(x) else x)
    print(f"\n  wrote {args.out}.json")


if __name__ == "__main__":
    main()