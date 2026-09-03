"""Train a CVRPTW (variant A) policy with POMO.

Example:
    python scripts/train_cvrptw.py --n-customers 20 --steps 2000 --lr 3e-4 --lr-schedule cosine

Mirrors scripts/train.py but uses the CVRPTW env/policy and the calibrated dials
(capacity 20, shift_end 4.0, tw_half_width 0.5, service 0.2). Saves the best
(lowest validation cost) checkpoint with the CVRPTW instance dials so eval can
rebuild identical instances. Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.models.factory import DEFAULT_CVRPTW_CONFIG, build_cvrptw_policy
from src.train.pomo_cvrptw import multistart_greedy_cost_cvrptw, train_step_cvrptw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CVRPTW policy with POMO (variant A)")
    # problem
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--capacity", type=int, default=20, help="calibrated CVRPTW capacity")
    p.add_argument("--shift-end", type=float, default=4.0)
    p.add_argument("--tw-half-width", type=float, default=0.5)
    p.add_argument("--service-time", type=float, default=0.2)
    # optimisation (certified CVRP recipe)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--n-starts", type=int, default=None, help="None = all customers")
    # run
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--val-seed", type=int, default=9999)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", type=str, default="checkpoints/cvrptwA_best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model_config = {**DEFAULT_CVRPTW_CONFIG}
    policy = build_cvrptw_policy(model_config).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.min_lr
        )

    def sample_batch():
        return generate_cvrptw(
            args.batch_size, args.n_customers, args.capacity,
            service_time=args.service_time, shift_end=args.shift_end,
            tw_half_width=args.tw_half_width, device=device,
        )

    val = generate_cvrptw(
        args.val_size, args.n_customers, args.capacity,
        service_time=args.service_time, shift_end=args.shift_end,
        tw_half_width=args.tw_half_width, seed=args.val_seed, device=device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    print(f"CVRPTW variant A | cap {args.capacity} shift {args.shift_end} "
          f"tw_half {args.tw_half_width} | effective batch "
          f"{args.batch_size * args.accum_steps}")

    for step in range(1, args.steps + 1):
        metrics = train_step_cvrptw(
            policy, optimizer, sample_batch, args.accum_steps, args.n_starts, args.grad_clip
        )
        if scheduler is not None:
            scheduler.step()

        if args.log_every and step % args.log_every == 0:
            print(f"step {step:5d}/{args.steps} | loss {metrics['loss']:+.4f} "
                  f"| train_cost {metrics['mean_cost']:.4f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| {step / (time.time() - t0):.1f} it/s")

        if step % args.eval_every == 0 or step == args.steps:
            val_cost = multistart_greedy_cost_cvrptw(policy, val, args.n_starts).mean().item()
            tag = ""
            if val_cost < best_val:
                best_val = val_cost
                torch.save(
                    {"model_state": policy.state_dict(), "model_config": model_config,
                     "n_customers": args.n_customers, "capacity": args.capacity,
                     "shift_end": args.shift_end, "tw_half_width": args.tw_half_width,
                     "service_time": args.service_time, "val_cost": val_cost},
                    out_path,
                )
                tag = "  <- saved best"
            print(f"  [eval] step {step:5d} | val_cost {val_cost:.4f} | best {best_val:.4f}{tag}")

    print(f"done. best val_cost {best_val:.4f}. checkpoint: {out_path}")


if __name__ == "__main__":
    main()