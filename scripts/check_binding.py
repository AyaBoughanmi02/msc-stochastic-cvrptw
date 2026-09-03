"""Check that CVRPTW instances (and a trained policy's routing) are TIME-binding.

Reports two views of what constrains routing, so you can confirm the temporal
constraints actually bite before building the stochastic variants:
  - instance-level (greedy-nearest proxy): a property of the instances;
  - policy-level (the trained checkpoint's own greedy routing).
A meaningful time_only_rate (vs a near-zero capacity_only_rate) means time drives
routing -- the precondition for the variance gate to matter.

Example:
    python scripts/check_binding.py --ckpt checkpoints/cvrptwA.pt --test-size 512

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.envs.cvrptw_diagnostics import calibration_stats, policy_binding_stats
from src.models.factory import build_cvrptw_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVRPTW time-vs-capacity binding check")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--test-seed", type=int, default=2024)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    policy = build_cvrptw_policy(ckpt["model_config"]).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    # Rebuild the instances the checkpoint was trained on (same dials).
    data = generate_cvrptw(
        args.test_size, ckpt["n_customers"], ckpt["capacity"],
        service_time=ckpt["service_time"], shift_end=ckpt["shift_end"],
        tw_half_width=ckpt["tw_half_width"], seed=args.test_seed, device=device,
    )

    inst = calibration_stats(data)             # greedy-nearest proxy (instance-level)
    pol = policy_binding_stats(policy, data)    # trained policy (policy-level)

    print(f"CVRPTW{ckpt['n_customers']} | cap {ckpt['capacity']} shift {ckpt['shift_end']} "
          f"tw_half {ckpt['tw_half_width']} | {args.test_size} instances")
    print("                     time_only  capacity_only  both")
    print(f"instance-level    :   {inst['time_only_rate']:.3f}       "
          f"{inst['capacity_only_rate']:.3f}     {inst['both_rate']:.3f}")
    print(f"trained policy    :   {pol['time_only_rate']:.3f}       "
          f"{pol['capacity_only_rate']:.3f}     {pol['both_rate']:.3f}")
    verdict = "TIME-binding" if pol["time_only_rate"] > pol["capacity_only_rate"] else "capacity-binding"
    print(f"verdict           :   {verdict} (time_only vs capacity_only)")


if __name__ == "__main__":
    main()