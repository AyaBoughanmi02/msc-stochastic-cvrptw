"""X0 / X0b: PyVRP classical reference against the neural policies, on the SAME instances
and the SAME realised days as scripts/apriori_study.py.

Example (PowerShell; arms exactly as in apriori_study, the FIRST arm supplies the
instance parameters):
    python -u -m scripts.pyvrp_reference `
      --arm M  checkpoints\\Dp_w1_15k_s2024_last.pt checkpoints\\Dp_w1_15k_s7_last.pt checkpoints\\Dp_w1_15k_s1234_last.pt `
      --arm M0 checkpoints\\M0_w1_15k_s2024_last.pt checkpoints\\M0_w1_15k_s7_last.pt checkpoints\\M0_w1_15k_s1234_last.pt `
      --workers 4 --out runs\\pyvrp_ref_w1 2>&1 | Tee-Object -FilePath runs\\pyvrp_ref_w1.console.log

What it reports (see src/eval/pyvrp_reference.py for what PyVRP is and is NOT):

  (a) DETERMINISTIC (alpha = 0, beta_var = 0): PyVRP plans at each --time-limits value vs
      each arm's best of 20 starts (no augmentation) and best of 160 (8 augmentations).
      Best-of-N is honest here: with no noise, a route's cost is known before driving.
      -> gap to a near-optimal solver, and inference time per instance.
  (b) STOCHASTIC (the checkpoint's simulator, beta_var in --beta-var): the PyVRP
      MEAN-VALUE plan (solved at --plan-limit, fixed in advance) executed on the test days
      vs each arm under the planned_distance rule (honest) and hindsight (for reference).
  (c) X0b, if --buffer-k is given: PyVRP plans on d + k * sqrt(sigma_bar^2), executed on
      the same days. k is CHOSEN on a held-out run (--test-seed 777 --test-size 512
      --beta-var 0.15 0.375 0.6 --skip-neural), never on the test set.

READING (fixed before running; also in experiment_plan_final.md, X0 / X0b):
  * deterministic gap of the best arm: < 5% close to near-optimal; 5-15% typical small-
    budget neural solver; > 15% a limitation to state with every negative result.
  * stochastic: PyVRP mean-value plan below every arm at every beta -> "a near-optimal
    mean-only plan is strongest here"; an arm below PyVRP -> "online re-decision is worth
    the difference".
  * X0b: k > 0 selected on held-out AND buffered < unbuffered at test -> sigma^2 is useful
    information that a simple classical rule exploits. k = 0 selected -> even a classical
    planner gains nothing from sigma^2 in this simulator.

BUILT-IN CHECK: each arm's planned_distance and hindsight means in (b) must reproduce
runs/apriori_saa_w1 for the same arm, PROVIDED this runs on the same device type (a CUDA
and a CPU generator give different days for the same seed).

Outputs: <out>.log (console copy), <out>.json (summary), <out>_per_instance.pt
(per-instance tensors for paired figures).
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from scripts.bn_mechanism import _tee_console
from scripts.compare_variants import forced_gate, paired_difference
from scripts.risk_components import forced_risk
from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.eval.apriori_selection import _slice, evaluate_stoch_apriori
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import _augment
from src.eval.pyvrp_reference import (
    PlanResult,
    buffered_durations,
    draw_test_days,
    execute_actions,
    fallback_to_feasible,
    pack_instances,
    routes_to_actions,
    solve_batch,
    summarise_plans,
    wall_time,
)
from src.train.pomo import build_starts
from src.train.pomo_stoch import assert_ood_eval_mode, build_stoch_env

_COMPONENTS = ("distance", "late_count", "tardiness", "total")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyVRP classical reference (X0 / X0b)")
    p.add_argument("--arm", nargs="+", action="append", required=True,
                   metavar=("NAME", "CKPT"),
                   help="a name then one checkpoint per seed (same seed order in every "
                        "arm). The FIRST arm's checkpoint supplies the instance parameters")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--gate", type=float, default=0.0,
                   help="lambda forced on every arm that has a training-time gate")
    p.add_argument("--risk-alpha", type=float, default=0.0,
                   help="alpha forced on VGT-R arms (default 0 = VGT-R0, as selected)")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--time-limits", type=float, nargs="+", default=[0.1, 1.0, 5.0],
                   help="PyVRP wall-clock seconds per instance, deterministic table")
    p.add_argument("--plan-limit", type=float, default=5.0,
                   help="the time limit whose plan is executed in (b); must be in "
                        "--time-limits")
    p.add_argument("--buffer-k", type=float, nargs="*", default=[],
                   help="X0b buffers, e.g. 0 0.25 0.5 0.75 (0 is added if missing)")
    p.add_argument("--buffer-limit", type=float, default=1.0,
                   help="PyVRP seconds per instance for every X0b plan (k = 0 included)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel PyVRP processes; at most the number of PHYSICAL cores")
    p.add_argument("--pyvrp-seed", type=int, default=0)
    p.add_argument("--skip-neural", action="store_true",
                   help="PyVRP only (e.g. the held-out X0b selection run)")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--out", type=str, default="runs/pyvrp_ref")
    add_budget_argument(p)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- neural side


@torch.no_grad()
def candidate_costs(
    policy, data: CVRPTWData, sim_cfg, weight: float, augment: bool,
    eval_batch_size: int, device: torch.device,
) -> tuple[Tensor, float]:
    """Cost of every candidate on ONE deterministic day -> ((B, A*S), seconds).

    Mirrors the rollout inside apriori_selection._chunk (same env builder, same forced
    starts, same augmentation fold). With sigma^2 = 0 the sampled env is deterministic.
    """
    assert_ood_eval_mode(policy)
    b, n, _ = data.coords.shape
    s = n - 1
    n_aug = 8 if augment else 1

    def run() -> Tensor:
        outs = []
        for i in range(0, b, eval_batch_size):
            part = _slice(data, slice(i, i + eval_batch_size))
            bb = part.coords.size(0)
            part_a = _augment(part) if augment else part
            env = build_stoch_env(part_a, sim_cfg, s, weight)
            starts = build_starts(n_aug * bb, s, device)                 # (A*bb*S,)
            _, _, cost = policy(env, decode_type="greedy", force_first=starts)
            outs.append(cost.reshape(n_aug, bb, s).permute(1, 0, 2).reshape(bb, n_aug * s))
        return torch.cat(outs, dim=0)                                    # (B, A*S)

    return wall_time(run, device)


def forced(policy, args) -> ExitStack:
    """Context that forces lambda (gate arms) and alpha (VGT-R arms) as in apriori_study."""
    stack = ExitStack()
    if hasattr(policy, "gate_w"):
        stack.enter_context(forced_gate(policy, args.gate))
    if hasattr(policy, "risk_w") and args.risk_alpha is not None:
        stack.enter_context(forced_risk(policy, args.risk_alpha))
    return stack


def paired_line(name: str, a_seeds: list[Tensor], ref: Tensor) -> str:
    """'arm - ref' paired over instances (seed-averaged arm), with per-seed signs."""
    arm = torch.stack(a_seeds).mean(dim=0)
    p = paired_difference(arm, ref)
    ps = [float((x - ref).mean()) for x in a_seeds]
    agree = all(v < 0 for v in ps) or all(v > 0 for v in ps)
    return (f"{name:>8} {float(arm.mean()):>9.4f} {float(ref.mean()):>9.4f} "
            f"{p['mean']:>+9.4f} [{p['lo']:+8.4f}, {p['hi']:+8.4f}]   "
            + " ".join(f"{v:+.4f}" for v in ps) + f"   {'yes' if agree else 'NO'}")


# --------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)
    if args.plan_limit not in args.time_limits:
        raise ValueError("--plan-limit must be one of --time-limits")

    arms: list[tuple[str, list]] = []
    for spec in args.arm:
        if len(spec) < 2:
            raise ValueError(f"--arm needs a name and at least one checkpoint, got {spec}")
        arms.append((spec[0], [load_stoch_policy(p, device) for p in spec[1:]]))
    if len({len(ls) for _, ls in arms}) != 1:
        raise ValueError("every arm must have the same number of seeds (paired by position)")
    all_ckpts = [(p, l.blob) for spec, (_, ls) in zip(args.arm, arms)
                 for p, l in zip(spec[1:], ls)]
    assert_matched_budget(all_ckpts, allow=args.allow_budget_mismatch)

    ref_loaded = arms[0][1][0]
    ref = ref_loaded.blob
    weight = float(ref["tardiness_weight"])
    for _, ls in arms:
        for l in ls:
            if float(l.blob["tardiness_weight"]) != weight:
                raise ValueError("arms disagree on tardiness_weight -- not one problem")
    sim_cfg = ref_loaded.sim_cfg
    capacity_units = int(ref["capacity"])
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    b, n, _ = data.coords.shape
    dtype = data.coords.dtype
    print(f"{b} instances (seed {args.test_seed}), n = {n - 1} customers, capacity "
          f"{capacity_units}, tardiness_weight {weight}, device {device.type}")
    print(f"simulator from checkpoint: alpha {sim_cfg.alpha}, omega {sim_cfg.omega}, "
          f"eta {sim_cfg.eta}, mean-preserving ripple: {sim_cfg.is_mean_preserving}")
    if not sim_cfg.is_mean_preserving:
        raise RuntimeError("the ripple is not mean-preserving, so the distance plan is NOT "
                           "the day-average (mean-value) plan; the comparison would be wrong")
    summary: dict = {"args": vars(args), "n_instances": b, "n_customers": n - 1}
    per_inst: dict[str, Tensor] = {}

    # ---------------------------------------------------------------- PyVRP plans on d
    packed = pack_instances(data, capacity_units)
    plans: dict[float, list[PlanResult]] = {}
    print("\n0. PYVRP PLANS (deterministic, travel time = distance)")
    for limit in args.time_limits:
        plans[limit] = solve_batch(packed, limit, args.pyvrp_seed, args.workers)
        s = summarise_plans(plans[limit])
        print(f"  limit {limit:>5g} s: feasible {int(s['n_feasible'])}/{b}, complete "
              f"{int(s['n_complete'])}/{b}, mean runtime {s['mean_runtime_s']:.3f} s")
        summary.setdefault("plans", {})[str(limit)] = s
    actions = {lim: routes_to_actions(plans[lim], n, device) for lim in args.time_limits}

    # ---------------------------------------------------------------- (a) deterministic
    sim_det = replace(sim_cfg, alpha=0.0, beta_var=0.0)
    det_day = draw_test_days(b, n, sim_det, [0], device, dtype)        # degenerate: sigma2 = 0
    print("\n1a. DETERMINISTIC (alpha 0, beta 0): PyVRP")
    det_ref: Tensor | None = None
    for limit in args.time_limits:
        parts = execute_actions(data, actions[limit], sim_det, weight, det_day)
        tot = parts["total"].double().cpu()
        bad = int((parts["penalty"] > 1e-5).sum())
        print(f"  limit {limit:>5g} s: cost {float(tot.mean()):.4f}"
              + (f"   WARNING {bad} plans incur lateness (PyVRP infeasible)" if bad else ""))
        per_inst[f"det/pyvrp/{limit:g}"] = tot
        if limit == args.plan_limit:
            det_ref = tot
    summary["deterministic"] = {"pyvrp": {str(k): float(per_inst[f"det/pyvrp/{k:g}"].mean())
                                          for k in args.time_limits}}

    if not args.skip_neural:
        print(f"\n1b. DETERMINISTIC: neural arms vs PyVRP @ {args.plan_limit:g} s "
              "(gap = (arm - pyvrp) / pyvrp, per instance, %)")
        head = (f"{'arm':>8} {'seed':>4} {'best20':>9} {'gap20%':>7} {'ms/inst':>8} "
                f"{'best160':>9} {'gap160%':>8} {'ms/inst':>8}")
        print(head)
        print("-" * len(head))
        # warm-up: CUDA context and kernels, so the first timed call is not penalised
        with forced(arms[0][1][0].policy, args):
            candidate_costs(arms[0][1][0].policy, _slice(data, slice(0, 2)), sim_det,
                            weight, False, 2, device)
        assert det_ref is not None
        for name, ls in arms:
            b20s, b160s = [], []
            for si, l in enumerate(ls):
                with forced(l.policy, args):
                    c20, t20 = candidate_costs(l.policy, data, sim_det, weight, False,
                                               args.eval_batch_size, device)
                    c160, t160 = candidate_costs(l.policy, data, sim_det, weight, True,
                                                 args.eval_batch_size, device)
                best20 = c20.min(dim=1).values.double().cpu()
                best160 = c160.min(dim=1).values.double().cpu()
                g20 = float((100 * (best20 - det_ref) / det_ref).mean())
                g160 = float((100 * (best160 - det_ref) / det_ref).mean())
                print(f"{name:>8} {si + 1:>4} {float(best20.mean()):>9.4f} {g20:>7.2f} "
                      f"{1000 * t20 / b:>8.2f} {float(best160.mean()):>9.4f} {g160:>8.2f} "
                      f"{1000 * t160 / b:>8.2f}")
                b20s.append(best20)
                b160s.append(best160)
                per_inst[f"det/{name}/s{si}/best20"] = best20
                per_inst[f"det/{name}/s{si}/best160"] = best160
                summary.setdefault("deterministic", {}).setdefault(name, []).append(
                    {"best20": float(best20.mean()), "gap20_pct": g20, "ms20": 1000 * t20 / b,
                     "best160": float(best160.mean()), "gap160_pct": g160,
                     "ms160": 1000 * t160 / b})
            print(f"  {name} seed-averaged, paired vs PyVRP:")
            print("    best20 " + paired_line(name, b20s, det_ref))
            print("    best160" + paired_line(name, b160s, det_ref))
        print(f"  PyVRP time per instance: the limit itself ({args.plan_limit:g} s, one CPU "
              "core); neural ms/inst includes env building, on this device")

    # ---------------------------------------------------------------- (b) stochastic
    print(f"\n2. STOCHASTIC: PyVRP mean-value plan (@ {args.plan_limit:g} s) executed on "
          f"the test days, vs arms (rule planned_distance; hindsight for reference)")
    plan_actions = actions[args.plan_limit]
    days_by_beta: dict[float, list[Tensor]] = {}
    pv_total: dict[float, Tensor] = {}
    for beta in args.beta_var:
        sim_b = replace(sim_cfg, beta_var=beta)
        days = draw_test_days(b, n, sim_b, range(args.scenarios), device, dtype)
        days_by_beta[beta] = days
        parts = execute_actions(data, plan_actions, sim_b, weight, days)
        pv_total[beta] = parts["total"].double().cpu()
        per_inst[f"stoch/pyvrp/{beta:g}/total"] = pv_total[beta]
        comp = {k: float(parts[k].mean()) for k in _COMPONENTS}
        summary.setdefault("stochastic", {}).setdefault("pyvrp", {})[str(beta)] = comp
        print(f"  PyVRP beta {beta:g}: " + " | ".join(f"{k} {v:.4f}" for k, v in comp.items()))

    if not args.skip_neural:
        head = (f"{'arm':>8} {'arm(pd)':>9} {'pyvrp':>9} {'diff':>9} {'95% CI':>21}   "
                f"per-seed diff            all agree")
        for beta in args.beta_var:
            sim_b = replace(sim_cfg, beta_var=beta)
            print(f"\n  beta {beta:g}: arm (planned_distance) - PyVRP mean-value plan")
            print("  " + head)
            for name, ls in arms:
                pd_seeds, hs_seeds = [], []
                for si, l in enumerate(ls):
                    with forced(l.policy, args):
                        out = evaluate_stoch_apriori(
                            l.policy, data, sim_b, tardiness_weight=weight, augment=True,
                            eval_batch_size=args.eval_batch_size,
                            scenario_seeds=tuple(range(args.scenarios)),
                            n_selection_days=1,          # 'saa' unused here
                        )
                    pd = out["planned_distance"]
                    pd_seeds.append(pd["cost"].double().cpu())
                    hs_seeds.append(out["hindsight"]["cost"].double().cpu())
                    per_inst[f"stoch/{name}/s{si}/{beta:g}/planned_distance"] = pd_seeds[-1]
                    per_inst[f"stoch/{name}/s{si}/{beta:g}/hindsight"] = hs_seeds[-1]
                    summary["stochastic"].setdefault(name, {}).setdefault(str(beta), []).append(
                        {"planned_distance": {k: float(pd[k].mean()) for k in
                                              ("cost", "distance", "late_count", "tardiness")},
                         "hindsight": float(out["hindsight"]["cost"].mean())})
                print("  " + paired_line(name, pd_seeds, pv_total[beta]))
                print(f"  {'':>8} (hindsight mean {float(torch.stack(hs_seeds).mean()):.4f}; "
                      "check against apriori_saa_w1)")

    lo_b, hi_b = args.beta_var[0], args.beta_var[-1]
    deg = float((pv_total[hi_b] - pv_total[lo_b]).mean())
    print(f"\n  PyVRP degradation (beta {hi_b:g} - beta {lo_b:g}): {deg:+.4f}")
    summary["stochastic"]["pyvrp_degradation"] = deg

    # ---------------------------------------------------------------- (c) X0b buffers
    if args.buffer_k:
        ks = sorted(set([0.0] + [float(k) for k in args.buffer_k]))
        print(f"\n3. X0b: sigma-buffered PyVRP plans @ {args.buffer_limit:g} s, "
              f"k in {ks} (infeasible buffered plans fall back to the next smaller k)")
        base_plans = (plans[args.buffer_limit] if args.buffer_limit in plans
                      else solve_batch(packed, args.buffer_limit, args.pyvrp_seed, args.workers))
        table: dict[float, dict[float, float]] = {}
        for beta in args.beta_var:
            sim_b = replace(sim_cfg, beta_var=beta)
            by_k = [base_plans]
            for k in ks[1:]:
                pk = pack_instances(data, capacity_units, buffered_durations(data, sim_b, k))
                by_k.append(solve_batch(pk, args.buffer_limit, args.pyvrp_seed, args.workers))
            chosen, fallbacks = fallback_to_feasible(by_k)
            table[beta] = {}
            for k, plans_k, fb in zip(ks, chosen, fallbacks):
                parts = execute_actions(data, routes_to_actions(plans_k, n, device), sim_b,
                                        weight, days_by_beta[beta])
                tot = parts["total"].double().cpu()
                table[beta][k] = float(tot.mean())
                per_inst[f"buffer/{beta:g}/{k:g}/total"] = tot
                print(f"  beta {beta:g} k {k:<5g} cost {float(tot.mean()):.4f} "
                      f"(distance {float(parts['distance'].mean()):.4f}, penalty "
                      f"{float(parts['penalty'].mean()):.4f}) fallbacks {fb}/{b}")
                summary.setdefault("buffer", {}).setdefault(str(beta), {})[str(k)] = {
                    "cost": float(tot.mean()), "fallbacks": fb,
                    "distance": float(parts["distance"].mean()),
                    "penalty": float(parts["penalty"].mean())}
        mean_over_beta = {k: sum(table[bt][k] for bt in args.beta_var) / len(args.beta_var)
                          for k in ks}
        best_k = min(mean_over_beta, key=mean_over_beta.get)
        print("  mean over beta: " + " | ".join(f"k {k:g}: {v:.4f}"
                                                for k, v in mean_over_beta.items()))
        print(f"  argmin k on THIS instance set: {best_k:g}  (select k only from the held-out "
              "run; on the test set, report the held-out k against k = 0)")
        summary["buffer_mean_over_beta"] = {str(k): v for k, v in mean_over_beta.items()}
        summary["buffer_argmin_k"] = best_k

    with open(f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(per_inst, f"{args.out}_per_instance.pt")
    print(f"\nwrote {args.out}.json and {args.out}_per_instance.pt")


if __name__ == "__main__":     # required: PyVRP workers are spawned processes on Windows
    main()