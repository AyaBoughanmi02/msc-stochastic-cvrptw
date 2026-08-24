"""Instance chunking in evaluate must be result-preserving.

`eval_batch_size` exists only to bound peak GPU memory; per-instance costs are
independent, so chunked evaluation must equal the single-pass result. Greedy
decode is deterministic, so the check is exact equality, not approximate.
"""

from __future__ import annotations

import torch

from src.data.generator import generate_cvrp
from src.eval.evaluate import evaluate
from src.models.factory import build_policy


def _policy() -> object:
    torch.manual_seed(0)
    return build_policy().eval()


def test_chunking_matches_full_no_augment() -> None:
    policy = _policy()
    coords, demand, cap = generate_cvrp(12, 20, seed=2024, device="cpu")
    full = evaluate(policy, coords, demand, cap, augment=False, eval_batch_size=None)
    chunk = evaluate(policy, coords, demand, cap, augment=False, eval_batch_size=4)
    assert full.shape == (12,)
    assert torch.equal(full, chunk)


def test_chunking_matches_full_augment() -> None:
    policy = _policy()
    coords, demand, cap = generate_cvrp(6, 20, seed=2024, device="cpu")
    full = evaluate(policy, coords, demand, cap, augment=True, eval_batch_size=None)
    chunk = evaluate(policy, coords, demand, cap, augment=True, eval_batch_size=2)
    assert torch.equal(full, chunk)


def test_batch_size_ge_B_is_noop() -> None:
    policy = _policy()
    coords, demand, cap = generate_cvrp(5, 20, seed=2024, device="cpu")
    a = evaluate(policy, coords, demand, cap, augment=False, eval_batch_size=None)
    b = evaluate(policy, coords, demand, cap, augment=False, eval_batch_size=999)
    assert torch.equal(a, b)
