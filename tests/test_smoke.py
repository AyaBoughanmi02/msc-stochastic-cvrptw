"""Smoke tests for environment sanity. Run with: pytest tests/test_smoke.py"""
import torch


def test_torch_imports() -> None:
    """Confirm torch is installed and version is as expected."""
    assert torch.__version__.startswith("2."), f"unexpected torch version: {torch.__version__}"


def test_torch_basic_op() -> None:
    """Confirm basic tensor operations work on CPU."""
    # x: (2, 3), y: (3, 4), z = x @ y: (2, 4)
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = x @ y
    assert z.shape == (2, 4), f"expected (2, 4), got {tuple(z.shape)}"