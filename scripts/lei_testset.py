"""Convert Lei et al.'s released CVRP test set (CSV) into this repo's tensor format.

Lei's repo ships 10,000 fixed test instances per size in VRP/test_data/:
    vrp{n}_test_data.csv   (10000 * (n+1), 2)  coords, node 0 = depot
    vrp{n}_demand.csv      (10000, n+1)        demand / 10, depot column = 0
    vrp{n}_capcity.csv     (10000,)            capacity / 10 (3.0 for n=20)

Lei's convention (capacity 3, demands k/10) is the same distribution as ours
(capacity 30, integer demands 1..9): normalised demand = (k/10) / 3 = k / 30.

Example:
    python scripts/lei_testset.py --lei-dir external/lei/VRP/test_data --n-customers 20
    -> data/lei_cvrp20_test.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Lei (and Kool) capacities by customer count, in Lei's /10 units.
_LEI_CAPACITY: dict[int, float] = {20: 3.0, 50: 4.0, 100: 5.0}


def load_lei_testset(lei_dir: str | Path, n_customers: int) -> dict[str, torch.Tensor | int]:
    """Read Lei's CSVs and return tensors in this repo's convention.

    returns dict with
        coords:       (B, N, 2) float32, N = n_customers + 1, node 0 = depot
        demand:       (B, N)    float32 normalised by capacity, demand[:, 0] = 0
        demand_raw:   (B, N)    int64 integer demands 1..9, depot 0 (for OR-Tools)
        capacity_int: int       integer capacity (30 for n=20)
    """
    lei_dir = Path(lei_dir)
    n = n_customers + 1
    coords = np.loadtxt(lei_dir / f"vrp{n_customers}_test_data.csv", delimiter=",")
    demand = np.loadtxt(lei_dir / f"vrp{n_customers}_demand.csv", delimiter=",")
    cap = np.loadtxt(lei_dir / f"vrp{n_customers}_capcity.csv", delimiter=",")

    coords = coords.reshape(-1, n, 2)                      # (B, N, 2)
    b = coords.shape[0]
    demand = demand.reshape(b, n)                           # (B, N)
    cap = cap.reshape(b)                                    # (B,)

    # Crash-not-silent checks: fail loudly if the files are not what we assume.
    if not np.allclose(cap, _LEI_CAPACITY[n_customers]):
        raise ValueError(f"unexpected capacities {np.unique(cap)}")
    if not np.all(demand[:, 0] == 0):
        raise ValueError("depot demand is not zero")
    dem_raw = np.rint(demand * 10).astype(np.int64)         # (B, N) integer 0..9
    if not np.allclose(dem_raw / 10, demand):
        raise ValueError("demands are not multiples of 0.1")
    if dem_raw[:, 1:].min() < 1 or dem_raw[:, 1:].max() > 9:
        raise ValueError("customer demands outside 1..9")

    capacity_int = int(round(_LEI_CAPACITY[n_customers] * 10))  # 30 for n=20
    return {
        "coords": torch.from_numpy(coords).float(),
        "demand": torch.from_numpy(dem_raw).float() / capacity_int,
        "demand_raw": torch.from_numpy(dem_raw),
        "capacity_int": capacity_int,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Lei's CVRP test CSVs to .pt")
    p.add_argument("--lei-dir", type=str, required=True)
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    data = load_lei_testset(args.lei_dir, args.n_customers)
    out = Path(args.out or f"data/lei_cvrp{args.n_customers}_test.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    c = data["coords"]
    print(f"saved {out}: {c.shape[0]} instances, N={c.shape[1]}, "
          f"capacity={data['capacity_int']}, depot mean={c[:, 0].mean(0).tolist()}")


if __name__ == "__main__":
    main()