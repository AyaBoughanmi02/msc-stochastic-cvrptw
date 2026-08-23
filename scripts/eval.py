"""Evaluate a trained CVRP checkpoint.

Example:
    python scripts/eval.py --ckpt checkpoints/cvrp_best.pt --test-size 1000 --augment

Reports mean best-of-N greedy tour length. If --ref points to a .pt file holding
a (test_size,) tensor of reference costs (e.g. OR-Tools / LKH on the SAME
instances, same seed), also reports the mean optimality gap.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.generator import generate_cvrp
from src.eval.evaluate import evaluate, optimality_gap
from src.eval.reference import assert_reference_alignment, load_reference
from src.models.factory import build_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CVRP checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-customers", type=int, default=None, help="default: from ckpt")
    p.add_argument("--capacity", type=int, default=None)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--test-seed", type=int, default=2024)
    p.add_argument("--n-starts", type=int, default=None, help="None = all customers")
    p.add_argument("--augment", action="store_true", help="8x dihedral augmentation")
    p.add_argument("--ref", type=str, default=None, help=".pt with reference costs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy = build_policy(ckpt["model_config"]).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    n_customers = args.n_customers or ckpt.get("n_customers")
    if n_customers is None:
        raise ValueError("n_customers not in checkpoint; pass --n-customers")
    capacity = args.capacity if args.capacity is not None else ckpt.get("capacity")

    coords, demand, cap = generate_cvrp(
        args.test_size, n_customers, capacity, seed=args.test_seed, device=device
    )
    cost = evaluate(policy, coords, demand, cap, args.n_starts, augment=args.augment)

    mode = "greedy best-of-N" + (" + 8x aug" if args.augment else "")
    print(f"instances       : {args.test_size} (CVRP{n_customers})")
    print(f"decode          : {mode}")
    print(f"mean cost       : {cost.mean().item():.4f}")
    print(f"std cost        : {cost.std().item():.4f}")

    if args.ref is not None:
        ref, sidecar = load_reference(args.ref)
        ref = ref.to(cost.device)
        if sidecar is not None:
            # Fail loudly if the reference was built on different instances.
            assert_reference_alignment(
                sidecar, n_customers, args.test_seed, args.test_size
            )
        else:
            print("warning: reference has no sidecar; cannot verify instance "
                  "alignment (n/seed/size). Trust the gap only if the reference "
                  "was built on identical instances.")
        if ref.shape != cost.shape:
            raise ValueError(f"reference shape {ref.shape} != cost {cost.shape}")
        gap = optimality_gap(cost, ref)
        print(f"reference cost  : {ref.mean().item():.4f}")
        print(f"optimality gap  : {gap.mean().item():.2f}%")
        print("  (gap vs OR-Tools GLS reference: near-optimal, not proven-optimal;"
              " negative = model beat the reference)")


if __name__ == "__main__":
    main()