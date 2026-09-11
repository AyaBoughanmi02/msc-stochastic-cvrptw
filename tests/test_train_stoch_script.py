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


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run a script in a subprocess and fail with a READABLE message if it misbehaves.

    A script that is empty or missing its `if __name__ == "__main__"` block exits 0 and
    prints nothing, which otherwise surfaces as an opaque `assert "..." in ''`. Check the
    file is complete first, and always report captured output on failure.
    """
    path = REPO_ROOT / script
    assert path.exists(), f"{script} does not exist"
    source = path.read_text(encoding="utf-8")
    assert source.strip(), f"{script} is empty -- the file was not saved correctly"
    assert 'if __name__ == "__main__":' in source, (
        f"{script} has no `if __name__ == \"__main__\":` block, so running it does "
        f"nothing and exits 0 silently -- the file is probably truncated"
    )

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    # -u: unbuffered, so output is captured even if the process is killed by the timeout.
    result = subprocess.run(
        [sys.executable, "-u", script, *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    detail = (
        f"\n--- exit code {result.returncode} ---"
        f"\n--- stdout ---\n{result.stdout[-3000:]}"
        f"\n--- stderr ---\n{result.stderr[-3000:]}"
    )
    assert result.returncode == 0, detail
    assert result.stdout.strip(), f"{script} produced no output at all.{detail}"
    return result


def test_train_stoch_runs_and_saves_a_reproducible_checkpoint(tmp_path) -> None:
    out = tmp_path / "smoke.pt"
    result = _run_script(
        "scripts/train_stoch.py",
        "--n-customers", "8", "--steps", "4", "--batch-size", "4",
        "--val-size", "8", "--val-scenarios", "1", "--n-starts", "3",
        "--eval-every", "2", "--log-every", "2", "--gate-check-every", "2",
        "--eval-batch-size", "4", "--beta-var", "0.3", "--ood-beta-var", "0.6",
        "--tardiness-weight", "2.0", "--device", "cpu", "--out", str(out),
    )
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
            sys.executable, "-u", "scripts/train_stoch.py",
            "--n-customers", "8", "--steps", "1", "--batch-size", "2",
            "--val-size", "2", "--alpha", "0.5", "--device", "cpu",
            "--out", str(tmp_path / "never.pt"),
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode != 0, "an unreachable ripple should abort the run"
    assert "reachability floor" in (result.stderr + result.stdout)


def test_calibrate_tardiness_runs_and_reports_a_weight(tmp_path) -> None:
    """The calibration entry point must run and print a usable recommendation."""
    result = _run_script(
        "scripts/calibrate_tardiness.py",
        "--n-customers", "20", "--batch-size", "16",
        "--beta-var", "0.3", "--device", "cpu",
    )
    assert "penalty share by weight" in result.stdout, result.stdout[-3000:]
    assert "--tardiness-weight" in result.stdout, result.stdout[-3000:]


def test_check_trained_reports_the_full_verification(tmp_path) -> None:
    """Train a tiny checkpoint, then verify the post-training checker reads it back.

    Covers the handoff that would otherwise only be exercised after a real 800-step run:
    the checkpoint must carry enough to rebuild the policy AND the exact simulator, and the
    checker must produce the gate-chain verdict, the ID/OOD table and the level-2
    calibration from it alone.
    """
    ckpt = tmp_path / "tiny_b.pt"
    _run_script(
        "scripts/train_stoch.py",
        "--n-customers", "8", "--steps", "2", "--batch-size", "4",
        "--val-size", "4", "--val-scenarios", "1", "--n-starts", "3",
        "--eval-every", "2", "--log-every", "2", "--gate-check-every", "0",
        "--eval-batch-size", "4", "--beta-var", "0.3", "--tardiness-weight", "2",
        "--device", "cpu", "--out", str(ckpt),
    )
    result = _run_script(
        "scripts/check_trained.py",
        "--checkpoint", str(ckpt), "--test-size", "8", "--scenarios", "1",
        "--n-starts", "3", "--eval-batch-size", "4", "--ood-beta-var", "0.6",
        "--no-augment", "--skip-saturation", "--device", "cpu",
    )
    assert "gate chain: PASS" in result.stdout
    assert "ID / OOD" in result.stdout
    assert "(ID)" in result.stdout                      # the training-variance row
    assert "calibration under the TRAINED policy" in result.stdout
    assert "penalty share at the trained weight" in result.stdout


def test_check_trained_reads_a_variant_d_checkpoint(tmp_path) -> None:
    """The variant must be inferred from the SAVED WEIGHTS, not assumed.

    A variant-D checkpoint carries the gate scalar; rebuilding it as variant B fails
    load_state_dict(strict=True) with an 'Unexpected key gate_w'.
    """
    ckpt = tmp_path / "tiny_d.pt"
    _run_script(
        "scripts/train_stoch.py", "--variant", "d",
        "--n-customers", "8", "--steps", "2", "--batch-size", "4",
        "--val-size", "4", "--val-scenarios", "1", "--n-starts", "3",
        "--eval-every", "2", "--log-every", "2", "--gate-check-every", "0",
        "--eval-batch-size", "4", "--beta-var", "0.3", "--tardiness-weight", "2",
        "--device", "cpu", "--out", str(ckpt),
    )
    result = _run_script(
        "scripts/check_trained.py",
        "--checkpoint", str(ckpt), "--test-size", "8", "--scenarios", "1",
        "--n-starts", "3", "--eval-batch-size", "4", "--ood-beta-var", "0.6",
        "--no-augment", "--skip-saturation", "--device", "cpu",
    )
    assert "(variant D)" in result.stdout
    assert "gate chain: PASS" in result.stdout
    assert "softplus_w" in result.stdout          # the gate scalar is reported


def test_check_trained_rejects_a_variant_a_checkpoint(tmp_path) -> None:
    """A deterministic checkpoint has no simulator, so it cannot be verified this way."""
    import torch

    from src.models.factory import DEFAULT_CVRPTW_CONFIG, build_cvrptw_policy

    path = tmp_path / "variant_a.pt"
    torch.save(
        {"model_state": build_cvrptw_policy().state_dict(),
         "model_config": DEFAULT_CVRPTW_CONFIG},
        path,
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-u", "scripts/check_trained.py", "--checkpoint", str(path),
         "--device", "cpu"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode != 0
    assert "sim_config" in result.stderr