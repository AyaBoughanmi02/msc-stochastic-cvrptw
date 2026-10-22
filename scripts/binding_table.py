"""Table 3.2 (Chapter 3): is time or capacity the binding constraint? SEEDED re-run.

Inference only, CPU, a few seconds. For each vehicle capacity, generates the SAME seeded
instances (only the capacity changes) and runs `calibration_stats`: a greedy policy that
always drives to the nearest feasible node, recording, for every not-yet-served customer at
every step, whether it is blocked by time only, by capacity only, or by both.

Why re-run: the Stage-2 numbers (0.275 / 0.024 at capacity 20) cannot be quoted until we know
the generator was called with an explicit seed (the unseeded-generator bug fixed on
21 Sept 2026). This script passes the seed explicitly and prints it, so the table can say so.

Example (PowerShell, repo root, .venv active):
    python -u -m scripts.binding_table 2>&1 | Tee-Object -FilePath runs\\binding_table.log
"""

from __future__ import annotations

import argparse

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.envs.cvrptw_diagnostics import calibration_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seeded time-vs-capacity binding table")
    p.add_argument("--capacities", type=int, nargs="+", default=[30, 20, 15, 12])
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--shift-end", type=float, default=4.0)
    p.add_argument("--tw-half-width", type=float, default=0.5)
    p.add_argument("--service-time", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)          # belt and braces: nothing below may be unseeded
    print(f"{args.test_size} instances, seed {args.seed}, n = {args.n_customers}, "
          f"T = {args.shift_end}, half-width {args.tw_half_width}, service {args.service_time}"
          " | greedy nearest-feasible policy, deterministic travel times\n")
    rows: list[tuple[int, dict[str, float]]] = []
    for cap in args.capacities:
        data = generate_cvrptw(
            args.test_size, args.n_customers, cap, service_time=args.service_time,
            shift_end=args.shift_end, tw_half_width=args.tw_half_width,
            seed=args.seed, device="cpu",
        )
        rows.append((cap, calibration_stats(data)))

    print(f"{'capacity':>8} {'time only':>10} {'cap only':>10} {'both':>8} {'cust/route':>11}")
    for cap, s in rows:
        print(f"{cap:>8} {s['time_only_rate']:>10.3f} {s['capacity_only_rate']:>10.3f} "
              f"{s['both_rate']:>8.3f} {s['avg_customers_per_route']:>11.2f}")
    print("\nLaTeX rows for Table 3.2:")
    for cap, s in rows:
        print(f"\t\t{cap} & {s['time_only_rate']:.3f} & {s['capacity_only_rate']:.3f} & "
              f"{s['both_rate']:.3f} \\\\")


if __name__ == "__main__":
    main()