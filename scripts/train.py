"""Train a CVRP policy with POMO.

Example:
    python scripts/train.py --n-customers 20 --steps 2000 --batch-size 32
    # larger effective batch on a small GPU (32 x 4 = 128) via accumulation:
    python scripts/train.py --n-customers 20 --steps 2500 --batch-size 32 --accum-steps 4

Saves the best (lowest validation cost) checkpoint to --out. Checkpoints store
both the weights and the model config, so eval can rebuild the exact model.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data.generator import generate_cvrp
from src.eval.evaluate import evaluate
from src.models.factory import build_policy, model_config_from
from src.train.pomo import train_step_accum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CVRP policy with POMO")
    # problem
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--capacity", type=int, default=None, help="demand units; None=lookup")
    # optimisation (Kwon defaults)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="grad accumulation; effective batch = batch-size * accum-steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=None, help="off by default")
    p.add_argument("--n-starts", type=int, default=None, help="None = all customers")
    # architecture (locked)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hidden-edge-dim", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--norm", type=str, default="batch", choices=["batch", "layer", "none"],
                   help="encoder input norm; layer/none avoid BatchNorm train/eval mismatch")
    p.add_argument("--attn", type=str, default="scalar", choices=["scalar", "channel"],
                   help="encoder attention granularity; channel = Lei-faithful diagnostic")
    # run
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--val-seed", type=int, default=9999)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", type=str, default="checkpoints/cvrp_best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model_config = model_config_from(
        input_edge_dim=1,
        hidden_dim=args.hidden_dim,
        hidden_edge_dim=args.hidden_edge_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        norm=args.norm,
        attn=args.attn,
    )
    policy = build_policy(model_config).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Fixed validation set (same instances throughout training).
    val_coords, val_demand, val_cap = generate_cvrp(
        args.val_size, args.n_customers, args.capacity, seed=args.val_seed, device=device
    )

    # One POMO mini-batch of fresh instances; accumulation runs several per step.
    def sample_batch() -> tuple[torch.Tensor, torch.Tensor, float]:
        return generate_cvrp(
            args.batch_size, args.n_customers, args.capacity, device=device
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    eff_batch = args.batch_size * args.accum_steps
    print(f"effective batch : {eff_batch} ({args.batch_size} x {args.accum_steps} accum)")

    for step in range(1, args.steps + 1):
        metrics = train_step_accum(
            policy, optimizer, sample_batch, args.accum_steps,
            args.n_starts, args.grad_clip,
        )

        if args.log_every and step % args.log_every == 0:
            print(
                f"step {step:5d}/{args.steps} | loss {metrics['loss']:+.4f} "
                f"| train_cost {metrics['mean_cost']:.4f} "
                f"| {step / (time.time() - t0):.1f} it/s"
            )

        if step % args.eval_every == 0 or step == args.steps:
            val_cost = evaluate(
                policy, val_coords, val_demand, val_cap, args.n_starts
            ).mean().item()
            tag = ""
            if val_cost < best_val:
                best_val = val_cost
                torch.save(
                    {"model_state": policy.state_dict(), "model_config": model_config,
                     "n_customers": args.n_customers, "capacity": args.capacity,
                     "val_cost": val_cost},
                    out_path,
                )
                tag = "  <- saved best"
            print(f"  [eval] step {step:5d} | val_cost {val_cost:.4f} "
                  f"| best {best_val:.4f}{tag}")

    print(f"done. best val_cost {best_val:.4f}. checkpoint: {out_path}")


if __name__ == "__main__":
    main()