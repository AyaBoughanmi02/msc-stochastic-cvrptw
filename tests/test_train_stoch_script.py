"""Smoke test for scripts/train_stoch.py.

Not a training test -- six steps on tiny instances proves nothing about learning. What it
does prove is that the entry point is WIRED: the simulator dials reach the Edge Bank, the
periodic gate-chain assertions run without firing, the OOD readout is produced, and the
checkpoint carries everything needed to rebuild both the model and the exact simulator it
was trained under. Those are the failures that would otherwise surface an hour into a real
run, or worse, only when a checkpoint turns out to be unreproducible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_stoch_runs_and_saves_a_reproducible_checkpoint(tmp_path) -> None:
    out = tmp_path / "smoke.pt"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [
            sys.executable, "scripts/train_stoch.py",
            "--n-customers", "8", "--steps", "4", "--batch-size", "4",
            "--val-size", "8", "--val-scenarios", "1", "--n-starts", "3",
            "--eval-every", "2", "--log-every", "2", "--gate-check-every", "2",
            "--eval-batch-size", "4", "--beta-var", "0.3", "--ood-beta-var", "0.6",
            "--tardiness-weight", "2.0", "--device", "cpu", "--out", str(out),
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "[eval]" in result.stdout
    assert "OOD@0.6" in result.stdout            # the shifted readout is produced
    assert out.exists()

    blob = torch.load(out, map_location="cpu", weights_only=False)
    # Everything needed to rebuild the model AND the simulator it was trained under.
    for key in ("model_state", "model_config", "sim_config", "tardiness_weight",
                "val_scenario_seeds", "val_stats", "ood_stats", "val_cost"):
        assert key in blob, key
    assert blob["model_config"]["input_edge_dim"] == 2
    assert blob["model_config"]["edge_feat_dim"] == 2
    assert blob["sim_config"]["beta_var"] == 0.3
    assert blob["sim_config"]["shift_end"] == blob["shift_end"]
    assert blob["ood_stats"] is not None

    # The saved config must rebuild the architecture and load strictly.
    from src.models.factory import build_cvrptw_stoch_policy

    policy = build_cvrptw_stoch_policy(blob["model_config"])
    missing, unexpected = policy.load_state_dict(blob["model_state"], strict=True)
    assert not missing and not unexpected


def test_train_stoch_refuses_an_unreachable_ripple(tmp_path) -> None:
    """alpha=0.5 puts a far-corner customer out of reach at peak congestion."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [
            sys.executable, "scripts/train_stoch.py",
            "--n-customers", "8", "--steps", "1", "--batch-size", "2",
            "--val-size", "2", "--alpha", "0.5", "--device", "cpu",
            "--out", str(tmp_path / "never.pt"),
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode != 0
    assert "reachability floor" in (result.stderr + result.stdout)


def test_calibrate_tardiness_runs_and_reports_a_weight(tmp_path) -> None:
    """The calibration entry point must run and print a usable recommendation."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [
            sys.executable, "scripts/calibrate_tardiness.py",
            "--n-customers", "20", "--batch-size", "16",
            "--beta-var", "0.3", "--device", "cpu",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "penalty share by weight" in result.stdout
    assert "--tardiness-weight" in result.stdout      # a concrete suggestion was printed