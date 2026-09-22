"""Task 1 / E1: score ONE checkpoint under every decoding protocol on ONE test set.

Answers "how does my CVRP20 number compare with Lei's Table 5.5?" like for like.
Lei's 'Ours (Greedy)' 6.26 is ONE greedy trajectory per instance, no multi-start,
no augmentation, on his 10,000 released test instances, with the gap computed against
Kool's published Gurobi mean 6.10 (ratio of means). This script reports every protocol
with that exact formula, plus per-instance gaps when a same-instance reference exists.

Protocols (all greedy unless named):
    single        1 trajectory, the POLICY picks the first customer   (Lei/Kool greedy)
    start_mean    mean over the N forced-start greedy rollouts        (diagnostic)
    best_of_n     best of N forced-start greedy rollouts              (POMO, no aug)
    best_of_n_x8  best of N x 8 dihedral augmentations                (POMO, certified)
    sample_mean   mean of N forced-start SAMPLED rollouts, eval mode  (diagnostic: what
                  train_cost measures, but on test instances)

Examples:
    # Lei's 10k set (convert first with scripts/lei_testset.py)
    python scripts/eval_protocols.py --ckpt checkpoints/cvrp20_cos3e4.pt \
        --testset data/lei_cvrp20_test.pt
    # our seed-2024 set, with the OR-Tools reference for per-instance gaps
    python scripts/eval_protocols.py --ckpt checkpoints/cvrp20_cos3e4.pt \
        --test-seed 2024 --test-size 1000 --ref references/cvrp20_ref.pt

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

from src.data.generator import generate_cvrp
from src.envs.cvrp_env import CVRPEnv
from src.eval.evaluate import augment_coords_8
from src.eval.reference import load_reference
from src.models.factory import build_policy
from src.models.policy import CVRPPolicy
from src.train.pomo import build_starts

# Kool et al. 2019 Table 1, Gurobi, CVRP20 -- the reference ALL rows of Lei's Table 5.5
# are measured against (Lei copies it; see notes). Ratio-of-means convention.
PUBLISHED_OPT: dict[int, float] = {20: 6.10, 50: 10.38, 100: 15.65}  # 50/100: LKH3

ALL_MODES = ("single", "start_mean", "best_of_n", "best_of_n_x8", "sample_mean")


@torch.no_grad()
def _rollout_starts(
    policy: CVRPPolicy, coords: Tensor, demand: Tensor, n_starts: int, decode: str
) -> Tensor:
    """N forced-start rollouts per instance. coords (B,N,2), demand (B,N) -> (B, S)."""
    b = coords.shape[0]
    env = CVRPEnv(
        coords.repeat_interleave(n_starts, dim=0),          # (B*S, N, 2)
        demand.repeat_interleave(n_starts, dim=0),          # (B*S, N)
        1.0,
    )
    starts = build_starts(b, n_starts, coords.device)        # (B*S,)
    _, _, cost = policy(env, decode_type=decode, force_first=starts)  # (B*S,)
    return cost.view(b, n_starts)                            # (B, S)


@torch.no_grad()
def score_chunk(policy: CVRPPolicy, coords: Tensor, demand: Tensor, mode: str) -> Tensor:
    """Per-instance cost under `mode`. coords (B,N,2), demand (B,N) -> (B,)."""
    b, n, _ = coords.shape
    s = n - 1                                                 # all customers as starts
    if mode == "single":
        # ASSUMPTION (check in CVRPEnv.reset): force_first=None leaves the first action
        # to the policy, i.e. a plain Kool/Lei greedy rollout from the depot.
        env = CVRPEnv(coords, demand, 1.0)
        _, _, cost = policy(env, decode_type="greedy", force_first=None)
        return cost                                           # (B,)
    if mode == "start_mean":
        return _rollout_starts(policy, coords, demand, s, "greedy").mean(dim=1)
    if mode == "best_of_n":
        return _rollout_starts(policy, coords, demand, s, "greedy").min(dim=1).values
    if mode == "best_of_n_x8":
        aug = augment_coords_8(coords).reshape(8 * b, n, 2)   # (8B, N, 2)
        dem = demand.unsqueeze(0).expand(8, -1, -1).reshape(8 * b, n)
        best = _rollout_starts(policy, aug, dem, s, "greedy").min(dim=1).values  # (8B,)
        return best.view(8, b).min(dim=0).values              # (B,)
    if mode == "sample_mean":
        return _rollout_starts(policy, coords, demand, s, "sampling").mean(dim=1)
    raise ValueError(f"unknown mode {mode!r}")


def score(
    policy: CVRPPolicy, coords: Tensor, demand: Tensor, mode: str, chunk: int
) -> Tensor:
    """Chunk over instances (result-preserving except sample_mean's RNG). -> (B,)."""
    out: list[Tensor] = []
    b = coords.shape[0]
    t0 = time.time()
    for k, i in enumerate(range(0, b, chunk)):
        out.append(score_chunk(policy, coords[i:i + chunk], demand[i:i + chunk], mode).cpu())
        done = min(i + chunk, b)
        if k % 20 == 0 or done == b:                           # progress, same line
            rate = done / max(time.time() - t0, 1e-9)
            eta = (b - done) / max(rate, 1e-9)
            print(f"\r  {mode}: {done}/{b}  ({rate:.0f} inst/s, eta {eta / 60:.1f} min)",
                  end="", flush=True)
    print("\r" + " " * 70 + "\r", end="", flush=True)
    return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Score one checkpoint under all protocols")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--testset", type=str, default=None,
                   help=".pt from scripts/lei_testset.py; else generate with --test-seed")
    p.add_argument("--test-seed", type=int, default=2024)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first LIMIT instances (quick runs)")
    p.add_argument("--ref", type=str, default=None,
                   help="same-instance reference .pt (per-instance gaps)")
    p.add_argument("--modes", type=str, nargs="+", default=list(ALL_MODES),
                   choices=ALL_MODES)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="RNG for sample_mean")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy = build_policy(ckpt["model_config"]).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()                                             # BatchNorm running stats
    n_customers = int(ckpt["n_customers"])

    if args.testset:
        data = torch.load(args.testset, weights_only=False)
        coords, demand = data["coords"].to(device), data["demand"].to(device)
        source = f"{args.testset} ({coords.shape[0]} instances)"
    else:
        coords, demand, _ = generate_cvrp(
            args.test_size, n_customers, seed=args.test_seed, device=device
        )
        source = f"generated seed={args.test_seed} ({args.test_size} instances)"
    if args.limit is not None:
        coords, demand = coords[:args.limit], demand[:args.limit]
        source += f", first {coords.shape[0]}"
    if coords.shape[1] != n_customers + 1:
        raise ValueError(f"test set has N={coords.shape[1]}, ckpt n={n_customers}")

    ref = None
    if args.ref:
        ref, _ = load_reference(args.ref)
        if args.limit is not None:
            ref = ref[:args.limit]
        if ref.shape[0] != coords.shape[0]:
            raise ValueError(f"reference has {ref.shape[0]} costs, test set {coords.shape[0]}")

    opt = PUBLISHED_OPT[n_customers]
    print(f"checkpoint : {args.ckpt}  (device {device})", flush=True)
    print(f"test set   : {source}")
    print(f"published  : {opt} (Kool Gurobi; Lei's gap = mean/{opt} - 1)")
    if ref is not None:
        print(f"reference  : {args.ref} mean {ref.mean():.4f}")
    header = f"{'mode':<14}{'mean':>9}{'std':>8}{'gap_pub%':>10}"
    if ref is not None:
        header += f"{'RoM_ref%':>10}{'MoR_ref%':>10}"
    print(header + f"{'time_s':>9}")

    torch.manual_seed(args.seed)
    for mode in args.modes:
        t0 = time.time()
        cost = score(policy, coords, demand, mode, args.eval_batch_size)   # (B,)
        m = cost.mean().item()
        row = f"{mode:<14}{m:>9.4f}{cost.std().item():>8.4f}{100 * (m / opt - 1):>10.2f}"
        if ref is not None:
            rom = 100 * (m / ref.mean().item() - 1)                  # ratio of means
            mor = (100 * (cost - ref) / ref).mean().item()           # mean of ratios
            row += f"{rom:>10.2f}{mor:>10.2f}"
        print(row + f"{time.time() - t0:>9.1f}", flush=True)


if __name__ == "__main__":
    main()