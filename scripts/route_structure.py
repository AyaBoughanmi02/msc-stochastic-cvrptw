"""Explain WHY one variant is more robust: route structure and planning slack.

Example:
    python -m scripts.route_structure \
        --baseline checkpoints/B_rand_s1234.pt checkpoints/B_rand_s7.pt \
        --candidate checkpoints/D_rand_s1234.pt checkpoints/D_rand_s7.pt

A cost difference says WHETHER a variant is better; it does not say why. This measures the
structural quantities that determine how a plan survives contact with sampled travel
times, so a robustness result can be given a mechanism instead of being left as a number.

WHAT IS MEASURED, per customer arrival, on the PLANNED (mu) schedule:

  * slack = latest - planned_arrival. How much room the plan leaves before the window
    closes. A plan with little slack is feasible on paper and fragile in execution: any
    travel time above the mean turns it into a violation. The 10th percentile matters more
    than the mean -- violations come from the tight arrivals, not the typical ones.

  * arrival sd = sqrt(accumulated arrival-time variance). How much uncertainty has piled
    up by the time the vehicle gets there. Resets at the depot with the clock.

  * slack / arrival sd -- the SAFETY MARGIN in standard deviations, i.e. roughly how many
    sigma of bad luck the plan can absorb. This is the quantity the variance gate should
    move if it works the way it is claimed to: not by leaving more slack everywhere, but
    by leaving more slack precisely where the accumulated uncertainty is large.

  * routes per instance and customers per route. More, shorter routes mean the clock
    resets more often, so less uncertainty accumulates per route -- robustness bought with
    distance. Fewer, longer routes are cheaper and more fragile.

No network forward pass is needed after the rollout: the action sequence is replayed
through the analytic env, which is deterministic, so these numbers carry no sampling noise.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import statistics

import torch
from torch import Tensor

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import MU, SIGMA2, gather_edge
from src.eval.checkpoints import load_stoch_policy
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route structure and slack of two variants")
    p.add_argument("--baseline", type=str, nargs="+", required=True)
    p.add_argument("--candidate", type=str, nargs="+", required=True)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--n-starts", type=int, default=None)
    p.add_argument("--tight-slack", type=float, default=0.25,
                   help="an arrival with less slack than this counts as 'tight'")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def route_structure(loaded, data, n_starts: int | None, tight: float) -> dict[str, float]:
    """Greedy rollout, then replay it to measure slack and route structure."""
    policy, sim_cfg = loaded.policy, loaded.sim_cfg
    b = data.coords.size(0)
    starts = (data.coords.size(1) - 1) if n_starts is None else n_starts

    env = build_stoch_env(data, sim_cfg, starts, float(loaded.blob["tardiness_weight"]),
                          reward_mode="analytic")
    actions, _, _ = policy(
        env, "greedy", build_starts(b, starts, data.coords.device)
    )                                                              # (B*S, T)

    # Replay through a fresh env, recording per-arrival quantities.
    env.reset()
    row = torch.arange(env.batch_size, device=data.coords.device)
    slacks: list[Tensor] = []
    sds: list[Tensor] = []
    margins: list[Tensor] = []
    route_starts = torch.zeros(env.batch_size, device=data.coords.device)
    served = torch.zeros(env.batch_size, device=data.coords.device)
    at_depot = torch.ones(env.batch_size, dtype=torch.bool, device=data.coords.device)

    for t in range(actions.size(1)):
        action = actions[:, t]
        moments = gather_edge(env.bank, env.current_node, action,
                              env.current_bin(), env.instance_idx)     # (B*S, 2)
        arrival = env.time + moments[:, MU]
        sd = (env.time_var + moments[:, SIGMA2]).clamp_min(1e-12).sqrt()
        latest = env.time_windows[row, action, 1]
        slack = latest - arrival
        is_customer = action != 0

        if bool(is_customer.any()):
            slacks.append(slack[is_customer])
            sds.append(sd[is_customer])
            margins.append((slack / sd)[is_customer])
            served = served + is_customer.to(served.dtype)
            # A route begins whenever the vehicle leaves the depot for a customer.
            route_starts = route_starts + (is_customer & at_depot).to(route_starts.dtype)
        at_depot = action == 0
        env.step(action)

    slack_all = torch.cat(slacks)
    sd_all = torch.cat(sds)
    margin_all = torch.cat(margins)
    routes = route_starts.mean()
    return {
        "routes_per_instance": float(routes),
        "customers_per_route": float(served.mean() / routes.clamp_min(1e-9)),
        "slack_mean": float(slack_all.mean()),
        "slack_p10": float(slack_all.quantile(0.10)),
        "arrival_sd_mean": float(sd_all.mean()),
        "margin_mean": float(margin_all.mean()),
        "margin_p10": float(margin_all.quantile(0.10)),
        "tight_arrival_rate": float((slack_all < tight).to(torch.float32).mean()),
    }


_ROWS = [
    ("routes_per_instance", "routes / instance", "more routes = more clock resets"),
    ("customers_per_route", "customers / route", "longer routes accumulate more risk"),
    ("slack_mean", "mean slack", "room before the window closes"),
    ("slack_p10", "slack p10", "the tight arrivals are where violations come from"),
    ("arrival_sd_mean", "mean arrival sd", "uncertainty accumulated on arrival"),
    ("margin_mean", "mean margin (sd)", "slack measured in standard deviations"),
    ("margin_p10", "margin p10 (sd)", "worst-case safety margin"),
    ("tight_arrival_rate", "tight arrival rate", "share of arrivals with little slack"),
]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    baselines = [load_stoch_policy(p, device) for p in args.baseline]
    candidates = [load_stoch_policy(p, device) for p in args.candidate]
    ref = baselines[0]

    data = generate_cvrptw(
        args.test_size, ref.blob["n_customers"], ref.blob["capacity"],
        service_time=ref.blob["service_time"], shift_end=ref.blob["shift_end"],
        tw_half_width=ref.blob["tw_half_width"], seed=args.test_seed, device=device,
    )

    stats = {}
    for group, loaded_list in (("base", baselines), ("cand", candidates)):
        per_seed = [route_structure(l, data, args.n_starts, args.tight_slack)
                    for l in loaded_list]
        stats[group] = {k: statistics.fmean(s[k] for s in per_seed)
                        for k in per_seed[0]}

    b_label, c_label = baselines[0].variant, candidates[0].variant
    print(f"ROUTE STRUCTURE on the PLANNED schedule  |  {args.test_size} instances "
          f"(seed {args.test_seed})")
    print(f"  baseline {b_label}: {len(baselines)} seeds | "
          f"candidate {c_label}: {len(candidates)} seeds | averaged over seeds\n")
    head = f"{'quantity':>20} {b_label:>10} {c_label:>10} {'diff':>9}   what it means"
    print(head)
    print("-" * (len(head) + 14))
    for key, label, meaning in _ROWS:
        bv, cv = stats["base"][key], stats["cand"][key]
        print(f"{label:>20} {bv:>10.4f} {cv:>10.4f} {cv - bv:>+9.4f}   {meaning}")

    print("\n  The mechanism to look for: if the candidate is more robust, it should show a")
    print("  LARGER margin (slack per unit of accumulated uncertainty), especially at p10.")
    print("  If instead it just runs more routes, the robustness was bought with distance")
    print("  and is not attributable to risk-aware routing.")


if __name__ == "__main__":
    main()