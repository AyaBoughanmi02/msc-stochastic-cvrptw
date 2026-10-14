"""X3: PyVRP ROLLING HORIZON (online re-planning) vs the neural policies, on the SAME
instances and the SAME realised days as scripts/apriori_study.py and pyvrp_reference.py.

Why: X0 compared the neural policies (which re-decide at every stop with the real clock)
with a FROZEN PyVRP plan, so part of the gap was "online vs frozen". Here PyVRP also
re-plans at every stop from the real position, clock and remaining capacity, and only the
next move is executed (src/eval/pyvrp_rolling.py). Online vs online.

Example (PowerShell; the FIRST arm supplies the instance parameters):
    python -u -m scripts.pyvrp_rolling `
      --arm M0_ftH checkpoints\\M0_ftH_s2024_last.pt checkpoints\\M0_ftH_s7_last.pt checkpoints\\M0_ftH_s1234_last.pt `
      --workers 4 --out runs\\pyvrp_rolling 2>&1 | Tee-Object -FilePath runs\\pyvrp_rolling.console.log

What it reports, per beta:
  * RH       : rolling-horizon PyVRP on mean (day-average) travel times;
  * RH k=... : optional, rolling horizon on the X0b buffer d + k*sqrt(sigma_bar^2)
               (--buffer-k; k was selected for the FROZEN plan, so for RH it is a
               sensitivity check, not a tuned arm);
  * FROZEN   : the X0 mean-value plan (one PyVRP plan per instance, fixed), as a check
               that this run reproduces runs/pyvrp_ref_w1 section 2 on the same device;
  * each neural arm under the honest planned_distance rule, paired against RH.
Plus wall-clock time per instance-day for RH vs one neural forward pass.

READING (fixed before running, 14 Oct 2026):
  * RH below the best neural arm at every beta (CI < 0) -> "with re-planning, a
    near-optimal classical solver is better than our neural policy; the neural policy's
    advantage over PyVRP in X0 was online vs frozen". Report with the time per decision.
  * CI includes 0 at some beta -> "the neural policy matches online re-planning there,
    at a fraction of the computation".
  * Neural below RH (CI > 0 for RH - neural) -> "the learned policy beats myopic
    mean-value re-planning under uncertainty".
  No outcome is preferred; all are reported.

BUILT-IN CHECKS: FROZEN must reproduce pyvrp_ref_w1 (14.33 / 17.27 / 18.94, 5 s plans;
small differences come from the plan time limit) and each arm's planned_distance mean must
reproduce its apriori_study value, PROVIDED the same device type (CUDA vs CPU draw
different days for the same seed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from scripts.bn_mechanism import _tee_console
from scripts.compare_variants import paired_difference
from scripts.pyvrp_reference import forced, paired_line
from src.data.cvrptw_generator import generate_cvrptw
from src.eval.apriori_selection import evaluate_stoch_apriori
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy
from src.eval.pyvrp_reference import (
    buffered_durations,
    draw_test_days,
    execute_actions,
    pack_instances,
    routes_to_actions,
    solve_batch,
    summarise_plans,
)
from src.eval.pyvrp_rolling import pack_static, rolling_horizon

_COMPONENTS = ("distance", "late_count", "tardiness", "total")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyVRP rolling horizon vs neural (X3)")
    p.add_argument("--arm", nargs="+", action="append", required=True,
                   metavar=("NAME", "CKPT"),
                   help="a name then one checkpoint per seed. The FIRST arm's checkpoint "
                        "supplies the instance parameters")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--gate", type=float, default=0.0)
    p.add_argument("--risk-alpha", type=float, default=None,
                   help="alpha forced on VGT-R arms; default None = learned (as apriori_study)")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--iterations", type=int, default=500,
                   help="PyVRP iterations per re-plan (load-independent stopping)")
    p.add_argument("--buffer-k", type=float, nargs="*", default=[],
                   help="extra RH arms on the X0b buffer (sensitivity), e.g. 0.75")
    p.add_argument("--frozen-limit", type=float, default=1.0,
                   help="PyVRP seconds for the FROZEN mean-value plan (check only)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel PyVRP processes; at most physical cores minus one if "
                        "a GPU training run shares the machine")
    p.add_argument("--pyvrp-seed", type=int, default=0)
    p.add_argument("--skip-neural", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", type=str, default="runs/pyvrp_rolling")
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

    arms: list[tuple[str, list]] = []
    for spec in args.arm:
        if len(spec) < 2:
            raise ValueError(f"--arm needs a name and at least one checkpoint, got {spec}")
        arms.append((spec[0], [load_stoch_policy(p, device) for p in spec[1:]]))
    if len({len(ls) for _, ls in arms}) != 1:
        raise ValueError("every arm must have the same number of seeds")
    assert_matched_budget([(p, l.blob) for spec, (_, ls) in zip(args.arm, arms)
                           for p, l in zip(spec[1:], ls)], allow=args.allow_budget_mismatch)

    ref_loaded = arms[0][1][0]
    ref = ref_loaded.blob
    weight = float(ref["tardiness_weight"])
    sim_cfg = ref_loaded.sim_cfg
    if not sim_cfg.is_mean_preserving:
        raise RuntimeError("ripple not mean-preserving: the distance is not the day-average")
    capacity_units = int(ref["capacity"])
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    b, n, _ = data.coords.shape
    dtype = data.coords.dtype
    print(f"{b} instances (seed {args.test_seed}), n = {n - 1}, capacity {capacity_units}, "
          f"tardiness_weight {weight}, device {device.type}, {args.scenarios} days, "
          f"PyVRP {args.iterations} iterations per re-plan, {args.workers} workers")
    summary: dict = {"args": vars(args), "n_instances": b}
    per_inst: dict[str, Tensor] = {}

    frozen_plans = solve_batch(pack_instances(data, capacity_units), args.frozen_limit,
                               args.pyvrp_seed, args.workers)
    s = summarise_plans(frozen_plans)
    print(f"frozen mean-value plans @ {args.frozen_limit:g} s: feasible "
          f"{int(s['n_feasible'])}/{b}, complete {int(s['n_complete'])}/{b}")
    frozen_actions = routes_to_actions(frozen_plans, n, device)

    rh_arms: list[tuple[str, float]] = [("RH", 0.0)] + [(f"RH k={k:g}", float(k))
                                                        for k in args.buffer_k if k > 0]
    rh_total: dict[float, Tensor] = {}
    for beta in args.beta_var:
        sim_b = replace(sim_cfg, beta_var=beta)
        days = draw_test_days(b, n, sim_b, range(args.scenarios), device, dtype)
        print(f"\n=== beta {beta:g} ===")
        fz = execute_actions(data, frozen_actions, sim_b, weight, days)
        fz_tot = fz["total"].double().cpu()
        per_inst[f"{beta:g}/FROZEN/total"] = fz_tot
        comp = {k: float(fz[k].mean()) for k in _COMPONENTS}
        summary.setdefault(str(beta), {})["FROZEN"] = comp
        print("  FROZEN   " + " | ".join(f"{k} {v:.4f}" for k, v in comp.items()))

        for name, k in rh_arms:
            dur = None if k == 0.0 else buffered_durations(data, sim_b, k)
            static = pack_static(data, capacity_units, dur)
            parts, stats = rolling_horizon(
                data, static, sim_b, weight, days, max_iterations=args.iterations,
                workers=args.workers, seed=args.pyvrp_seed, log_every=args.log_every,
            )
            tot = parts["total"]
            per_inst[f"{beta:g}/{name}/total"] = tot
            for key in _COMPONENTS:
                per_inst[f"{beta:g}/{name}/{key}"] = parts[key]
            comp = {key: float(parts[key].mean()) for key in _COMPONENTS}
            summary[str(beta)][name] = {**comp, **stats}
            print(f"  {name:<8} " + " | ".join(f"{kk} {v:.4f}" for kk, v in comp.items())
                  + f" | {stats['seconds_per_instance_day']:.3f} s per instance-day, "
                  f"infeasible re-plans {int(stats['infeasible_solves'])}/"
                  f"{int(stats['solves'])}")
            if k == 0.0:
                rh_total[beta] = tot
                p = paired_difference(tot, fz_tot)
                print(f"  RH - FROZEN: {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]")
                summary[str(beta)]["RH_minus_FROZEN"] = dict(p)

        if not args.skip_neural:
            print("\n  arm (planned_distance) - RH, paired over instances:")
            print(f"  {'arm':>8} {'arm':>9} {'RH':>9} {'diff':>9} {'95% CI':>21}   "
                  f"per-seed diff   all agree")
            for name, ls in arms:
                seeds_cost: list[Tensor] = []
                t_net = 0.0
                for si, l in enumerate(ls):
                    t0 = time.perf_counter()
                    with forced(l.policy, args):
                        out = evaluate_stoch_apriori(
                            l.policy, data, sim_b, tardiness_weight=weight, augment=True,
                            eval_batch_size=args.eval_batch_size,
                            scenario_seeds=tuple(range(args.scenarios)),
                            n_selection_days=1,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_net += time.perf_counter() - t0
                    pd = out["planned_distance"]["cost"].double().cpu()
                    seeds_cost.append(pd)
                    per_inst[f"{beta:g}/{name}/s{si}/planned_distance"] = pd
                print("  " + paired_line(name, seeds_cost, rh_total[beta]))
                summary[str(beta)].setdefault("neural", {})[name] = {
                    "planned_distance_per_seed": [float(x.mean()) for x in seeds_cost],
                    "seconds_per_instance_day_all_seeds": t_net / (b * args.scenarios),
                }

    lo, hi = args.beta_var[0], args.beta_var[-1]
    if lo in rh_total and hi in rh_total:
        deg = float((rh_total[hi] - rh_total[lo]).mean())
        print(f"\nRH degradation (beta {hi:g} - beta {lo:g}): {deg:+.4f}")
        summary["RH_degradation"] = deg

    with open(f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(per_inst, f"{args.out}_per_instance.pt")
    print(f"\nwrote {args.out}.json and {args.out}_per_instance.pt")


if __name__ == "__main__":     # required: PyVRP workers are spawned processes on Windows
    main()