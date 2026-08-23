"""Tests for reference loading and the eval alignment guard.

The guard is the thing standing between "correct optimality gap" and "same-shape
but silently-wrong gap", so its pass/raise behaviour is pinned explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.eval.reference import assert_reference_alignment, load_reference

_INSTANCES = {"n_customers": 20, "test_seed": 2024, "test_size": 1}


def _write_reference(dirpath: Path, costs: list[float], instances: dict) -> Path:
    pt = dirpath / "ref.pt"
    torch.save(torch.tensor(costs, dtype=torch.float32), pt)
    (dirpath / "ref.json").write_text(json.dumps({"instances": instances}))
    return pt


def test_load_reference_with_sidecar(tmp_path: Path) -> None:
    pt = _write_reference(tmp_path, [4.0, 5.0, 6.0], _INSTANCES)
    costs, sidecar = load_reference(pt)
    assert costs.shape == (3,) and costs.dtype == torch.float32
    assert sidecar is not None
    assert sidecar["instances"]["n_customers"] == 20


def test_load_reference_without_sidecar(tmp_path: Path) -> None:
    pt = tmp_path / "bare.pt"                        # no accompanying .json
    torch.save(torch.tensor([1.0, 2.0]), pt)
    costs, sidecar = load_reference(pt)
    assert costs.shape == (2,)
    assert sidecar is None                          # missing sidecar -> None, not error


def test_alignment_passes_on_match(tmp_path: Path) -> None:
    _, sidecar = load_reference(_write_reference(tmp_path, [4.0], _INSTANCES))
    assert sidecar is not None
    assert_reference_alignment(sidecar, 20, 2024, 1)  # matches -> no raise


@pytest.mark.parametrize(
    "n_customers, test_seed, test_size",
    [(50, 2024, 1), (20, 9999, 1), (20, 2024, 999)],  # n / seed / size mismatch
)
def test_alignment_raises_on_mismatch(
    tmp_path: Path, n_customers: int, test_seed: int, test_size: int
) -> None:
    _, sidecar = load_reference(_write_reference(tmp_path, [4.0], _INSTANCES))
    assert sidecar is not None
    with pytest.raises(ValueError, match="instance mismatch"):
        assert_reference_alignment(sidecar, n_customers, test_seed, test_size)