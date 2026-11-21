"""Learning-rate schedules for scripts/train_stoch.py: cosine, optionally after a linear warm-up.

Warm-up exists for one measured failure: training M0 FROM SCRATCH with the hybrid reward
collapsed in 3 of 6 runs, every time within the first 100-300 steps (validation frozen at
22.39, pre-tanh scores saturated). A linear warm-up keeps the first updates small. With
`warmup_steps == 0` the old CosineAnnealingLR is returned unchanged, so every earlier run
is reproduced exactly.
"""

from __future__ import annotations

import math

import torch


def warmup_cosine_factor(step: int, total: int, warmup: int, base_lr: float,
                         min_lr: float) -> float:
    """Multiplier on `base_lr` after `step` completed optimizer steps.

    lr(step) = min(1, (step + 1) / warmup) * [min_lr + (base_lr - min_lr) * cos_term],
    cos_term = (1 + cos(pi * step / total)) / 2. The warm-up multiplies the cosine, so
    after it the schedule is the ordinary cosine from base_lr to min_lr over `total`.
    """
    if total < 1 or base_lr <= 0:
        raise ValueError(f"need total >= 1 and base_lr > 0, got {total}, {base_lr}")
    s = min(max(step, 0), total)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * s / total))
    lr = min_lr + (base_lr - min_lr) * cos_term
    warm = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    return warm * lr / base_lr


def build_scheduler(optimizer: torch.optim.Optimizer, schedule: str, steps: int,
                    min_lr: float, warmup_steps: int = 0
                    ) -> torch.optim.lr_scheduler.LRScheduler | None:
    """The scheduler train_stoch steps once per optimizer step.

    schedule 'none' -> None (warm-up refused); 'cosine' with warmup_steps == 0 -> the
    original CosineAnnealingLR; 'cosine' with warmup_steps > 0 -> LambdaLR, one lambda per
    param group so each group (e.g. a gate with its own lr) is scaled from its own base.
    """
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError(f"warmup_steps must be in [0, steps), got {warmup_steps} for {steps}")
    if schedule == "none":
        if warmup_steps:
            raise ValueError("--warmup-steps needs --lr-schedule cosine")
        return None
    if schedule != "cosine":
        raise ValueError(f"unknown schedule {schedule!r}")
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps,
                                                          eta_min=min_lr)
    lambdas = []
    for group in optimizer.param_groups:
        base = float(group["lr"])
        floor = min(min_lr, base)
        lambdas.append(lambda s, b=base, f=floor:
                       warmup_cosine_factor(s, steps, warmup_steps, b, f))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
