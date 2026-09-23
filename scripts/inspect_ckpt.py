"""Print the training configuration stored in checkpoints, for comparability checks.

Usage:
    python -m scripts.inspect_ckpt checkpoints/B_fix_s1234.pt checkpoints/B_fix15k_s2024.pt
    python -m scripts.inspect_ckpt checkpoints/*_fix*.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

# Keys whose values must match across checkpoints before they may be compared.
COMPARABILITY_KEYS = (
    "variant",
    "n_customers",
    "capacity",
    "tardiness_weight",
    "beta_var_range",
    "shift_end",
    "tw_half_width",
    "service_time",
    "reward_mode",
)


def describe(path: Path) -> dict[str, Any]:
    """Load a checkpoint and return its configuration fields (no tensors)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    row: dict[str, Any] = {"name": path.stem}
    for key in COMPARABILITY_KEYS:
        row[key] = blob.get(key)
    sim = blob.get("sim_config", {}) or {}
    for key in ("alpha", "eta", "beta_var", "field_scale", "zeta", "risk_freq", "n_bins", "period"):
        row[f"sim.{key}"] = sim.get(key)
    metrics = blob.get("policy_metrics", {}) or {}
    row["softplus_w"] = metrics.get("softplus_w")
    row["val_cost"] = blob.get("val_cost")
    # Step count is only present if the checkpoint was written after the --save-every patch.
    for key in ("steps", "step", "global_step"):
        if key in blob:
            row["steps"] = blob[key]
            break
    else:
        row["steps"] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [describe(p) for p in args.paths]
    fields = list(rows[0].keys())

    # Column width from the widest value in each column.
    widths = {f: max(len(f), *(len(str(r.get(f))) for r in rows)) for f in fields}
    print(" | ".join(f.ljust(widths[f]) for f in fields))
    print("-+-".join("-" * widths[f] for f in fields))
    for row in rows:
        print(" | ".join(str(row.get(f)).ljust(widths[f]) for f in fields))

    # Flag any field that is not constant across the given checkpoints.
    if len(rows) > 1:
        differing = [
            f
            for f in fields
            if f not in {"name", "variant", "softplus_w", "val_cost", "steps", "sim.beta_var"}
            and len({str(r.get(f)) for r in rows}) > 1
        ]
        print()
        if differing:
            print("NOT COMPARABLE -- these fields differ:", ", ".join(differing))
        else:
            print("comparable: every training-setting field matches")


if __name__ == "__main__":
    main()