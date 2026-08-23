"""Generate an OR-Tools reference cost set for CVRP evaluation.

Solves the SAME instances scripts/eval.py scores (same generator, same seed,
same size) with OR-Tools guided local search, and saves a (test_size,) tensor of
near-optimal tour lengths that `scripts/eval.py --ref` consumes, plus a JSON
sidecar recording provenance (solver config, versions, alignment params).

The reference is near-optimal, NOT proven-optimal: negative gaps (POMO + 8x
augmentation beating it) are expected. Treat the saved .pt as the canonical
artifact (GLS under a wall-clock limit is not bitwise reproducible).

Example:
    python scripts/gen_reference.py --n-customers 20 --time-limit 2

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import ortools
import torch

from src.data.generator import generate_cvrp
from src.eval.reference import DEFAULT_SCALE, solve_reference_set

# Mirror the LOCKED search config in src/eval/reference.py (for the sidecar).
_FIRST_SOLUTION_NAME = "PATH_CHEAPEST_ARC"
_METAHEURISTIC_NAME = "GUIDED_LOCAL_SEARCH"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OR-Tools CVRP reference costs")
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--capacity", type=int, default=None, help="demand units; None=lookup")
    # Defaults MATCH scripts/eval.py so instances align by construction.
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--test-seed", type=int, default=2024)
    p.add_argument(
        "--time-limit", type=int, default=2,
        help="seconds per instance; use ~5 for n=50",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="output .pt path; default references/cvrp{n}_ref.pt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = (
        Path(args.out) if args.out
        else Path(f"references/cvrp{args.n_customers}_ref.pt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_path.with_suffix(".json")

    # Generate on CPU: generate_cvrp seeds a CPU generator then moves to device,
    # so CPU-generated instances are value-identical to eval's on any device.
    coords, _demand, _cap, dem_raw, D = generate_cvrp(
        args.test_size,
        args.n_customers,
        args.capacity,
        seed=args.test_seed,
        device="cpu",
        return_raw=True,
    )  # coords (B, N, 2); dem_raw (B, N) int64; D int

    est_min = args.test_size * args.time_limit / 60.0
    print("Generating OR-Tools reference. Run eval.py with MATCHING flags:")
    print(f"  --test-size {args.test_size}  --test-seed {args.test_seed}  "
          f"(CVRP{args.n_customers}, capacity D={D})")
    print(f"  time limit  : {args.time_limit}s/instance  (~{est_min:.0f} min total, serial)")
    print()

    costs, statuses = solve_reference_set(
        coords, dem_raw, D, scale=DEFAULT_SCALE, time_limit_s=args.time_limit
    )  # costs (B,) float32 in [0, 1] units

    torch.save(costs, out_path)

    status_counts: dict[int, int] = {}
    for s in statuses:
        status_counts[str(s)] = status_counts.get(str(s), 0) + 1

    sidecar = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "ortools": ortools.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "solver": {
            "first_solution_strategy": _FIRST_SOLUTION_NAME,
            "local_search_metaheuristic": _METAHEURISTIC_NAME,
            "time_limit_s": args.time_limit,
            "distance_scale": DEFAULT_SCALE,
        },
        # Alignment params: eval.py MUST use the same (n, seed, size) or the gap
        # is computed on different instances.
        "instances": {
            "test_size": args.test_size,
            "n_customers": args.n_customers,
            "capacity": D,
            "test_seed": args.test_seed,
        },
        "reference_pt": out_path.name,
        "status_counts": status_counts,          # routing.status() histogram; 1 == success
        "mean_reference_cost": costs.mean().item(),
        "note": (
            "gap to OR-Tools GLS reference: near-optimal, NOT proven-optimal; "
            "negative gaps from POMO+8x augmentation are expected, not bugs."
        ),
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"mean reference cost : {costs.mean().item():.4f}")
    print(f"std                 : {costs.std().item():.4f}")
    print(f"saved costs         : {out_path}  (shape {tuple(costs.shape)}, {costs.dtype})")
    print(f"saved sidecar       : {sidecar_path}")


if __name__ == "__main__":
    main()