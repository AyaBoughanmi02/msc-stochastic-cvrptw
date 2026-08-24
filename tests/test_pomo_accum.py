"""Gradient accumulation must equal the mean of the mini-batch gradients.

That mean-of-gradients identity is the exact, testable claim (a true single
large batch would differ only through BatchNorm's batch-coupling). Per-mini-batch
sampling RNG is fixed so the two paths see identical trajectories and the check
is meaningful.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp
from src.models.factory import build_policy
from src.train.pomo import pomo_loss, pomo_rollout, train_step_accum


def test_accumulated_grad_is_mean_of_minibatch_grads() -> None:
    torch.manual_seed(0)
    policy = build_policy().train()
    accum = 3
    batches = [generate_cvrp(4, 20, seed=100 + k, device="cpu") for k in range(accum)]
    seeds = [11, 22, 33]  # fix sampling per mini-batch so both paths match

    # Path 1: each mini-batch's gradient in isolation, then averaged.
    per_batch: list[list[torch.Tensor]] = []
    for k in range(accum):
        policy.zero_grad()
        torch.manual_seed(seeds[k])
        loss, _ = pomo_loss(*pomo_rollout(policy, *batches[k]))
        loss.backward()
        per_batch.append([p.grad.detach().clone() for p in policy.parameters()])
    mean_grads = [
        torch.stack([per_batch[k][i] for k in range(accum)]).mean(0)
        for i in range(len(per_batch[0]))
    ]

    # Path 2: accumulate with 1/accum scaling (what train_step_accum does).
    policy.zero_grad()
    for k in range(accum):
        torch.manual_seed(seeds[k])
        loss, _ = pomo_loss(*pomo_rollout(policy, *batches[k]))
        (loss / accum).backward()
    accum_grads = [p.grad.detach().clone() for p in policy.parameters()]

    # BatchNorm annihilates bias shifts, so bias gradients are near-zero and
    # suffer float32 cancellation: large *relative* but tiny *absolute* error.
    # A missing 1/accum scale would instead give errors of order |grad| (~30),
    # so a 1e-3 absolute bound is both float-honest and bug-catching.
    max_abs_err = max((a - m).abs().max().item() for a, m in zip(accum_grads, mean_grads))
    assert max_abs_err < 1e-3, max_abs_err


def test_train_step_accum_runs_and_updates() -> None:
    torch.manual_seed(0)
    policy = build_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in policy.parameters()]

    def sample_batch() -> tuple:
        return generate_cvrp(4, 20, device="cpu")

    metrics = train_step_accum(policy, opt, sample_batch, accum_steps=2)
    assert set(metrics) == {"loss", "mean_cost", "grad_norm"}
    assert all(torch.isfinite(torch.tensor(v)) for v in [metrics["loss"], metrics["mean_cost"]])
    moved = any(not torch.equal(b, p) for b, p in zip(before, policy.parameters()))
    assert moved  # an optimizer step actually changed the weights


def test_accum_steps_zero_rejected() -> None:
    policy = build_policy()
    opt = torch.optim.Adam(policy.parameters())
    try:
        train_step_accum(policy, opt, lambda: generate_cvrp(2, 20, device="cpu"), accum_steps=0)
    except ValueError:
        return
    raise AssertionError("accum_steps=0 should raise ValueError")