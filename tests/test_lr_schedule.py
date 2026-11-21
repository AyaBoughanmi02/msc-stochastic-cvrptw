"""Tests for src/train/lr_schedule.py (warm-up + cosine)."""

from __future__ import annotations

import math

import pytest
import torch

from src.train.lr_schedule import build_scheduler, warmup_cosine_factor


def _opt(lrs: list[float]) -> torch.optim.Optimizer:
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1, device="cpu"))], "lr": lr}
              for lr in lrs]
    return torch.optim.Adam(groups)


def _run(opt: torch.optim.Optimizer, sched, n: int) -> list[list[float]]:
    """lr of every group used at optimizer steps 1..n. -> n x groups."""
    seen = []
    for _ in range(n):
        seen.append([g["lr"] for g in opt.param_groups])
        opt.step()
        sched.step()
    return seen


def test_no_warmup_is_the_old_cosine() -> None:
    opt = _opt([3e-4])
    sched = build_scheduler(opt, "cosine", 100, 1e-6, 0)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_warmup_rises_linearly_then_follows_cosine() -> None:
    total, warm, base, floor = 1000, 100, 3e-4, 1e-6
    opt = _opt([base])
    lrs = [r[0] for r in _run(opt, build_scheduler(opt, "cosine", total, floor, warm), total)]
    assert lrs[0] == pytest.approx(base / warm * (floor + (base - floor)) / base * 1.0, rel=1e-6)
    assert lrs[49] == pytest.approx(0.5 * lrs[99], rel=0.02)        # linear in the warm-up
    cos = lambda s: floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * s / total))
    for s in (100, 500, 999):
        assert lrs[s] == pytest.approx(cos(s), rel=1e-6)          # plain cosine after it
    assert max(lrs) <= base * (1 + 1e-9)


def test_each_group_scaled_from_its_own_base() -> None:
    opt = _opt([3e-4, 1e-2])
    lrs = _run(opt, build_scheduler(opt, "cosine", 200, 1e-6, 20), 30)
    assert lrs[25][1] / lrs[25][0] == pytest.approx(
        warmup_cosine_factor(25, 200, 20, 1e-2, 1e-6) * 1e-2
        / (warmup_cosine_factor(25, 200, 20, 3e-4, 1e-6) * 3e-4), rel=1e-9)
    assert lrs[0][1] == pytest.approx(1e-2 / 20, rel=1e-3)


@pytest.mark.parametrize("schedule,steps,warm", [("cosine", 100, -1), ("cosine", 100, 100),
                                                 ("none", 100, 10)])
def test_invalid_arguments_raise(schedule: str, steps: int, warm: int) -> None:
    with pytest.raises(ValueError):
        build_scheduler(_opt([3e-4]), schedule, steps, 1e-6, warm)
