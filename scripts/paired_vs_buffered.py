"""Paired comparison of neural arms against a BUFFERED rolling-horizon PyVRP (X3b).

`scripts/pyvrp_rolling.py` pairs every neural arm against plain RH (k = 0) only. This
script reads the per-instance file it writes and pairs each arm against any RH variant
stored there (e.g. "RH k=0.5"), with the same paired test (`paired_difference`) and the
same seed-averaging as `pyvrp_reference.paired_line`.

No new computation: it only reads `<out>_per_instance.pt`. Keys used (written by
scripts/pyvrp_rolling.py):
    f"{beta:g}/{rh_name}/total"                      -> (n_instances,) RH cost
    f"{beta:g}/{arm}/s{si}/planned_distance"         -> (n_instances,) neural cost, seed si

Example (PowerShell):
    python -u -m scripts.paired_vs_buffered runs\\pyvrp_rolling_k05_per_instance.pt `
      --arm M0_ftH --rh "RH" "RH k=0.5" 2>&1 | Tee-Object -FilePath runs\\paired_vs_buffered.log
"""

from __future__ import annotations

import argparse
import re

import torch
from torch import Tensor

from scripts.compare_variants import paired_difference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="neural arm vs (buffered) RH, paired")
    p.add_argument("per_instance", type=str, help="<out>_per_instance.pt of pyvrp_rolling")
    p.add_argument("--arm", type=str, nargs="+", required=True, help="neural arm name(s)")
    p.add_argument("--rh", type=str, nargs="+", default=["RH"],
                   help='RH variants to compare against, e.g. "RH" "RH k=0.5"')
    return p.parse_args()


def _betas(keys: list[str]) -> list[str]:
    """Beta labels present in the file, in numeric order."""
    return sorted({k.split("/")[0] for k in keys}, key=float)


def _arm_seeds(per_inst: dict[str, Tensor], beta: str, arm: str) -> list[Tensor]:
    """Per-seed planned_distance costs of `arm` at `beta`, ordered by seed index."""
    pat = re.compile(rf"^{re.escape(beta)}/{re.escape(arm)}/s(\d+)/planned_distance$")
    found = sorted(((int(m.group(1)), v) for k, v in per_inst.items()
                    if (m := pat.match(k))), key=lambda t: t[0])
    if not found:
        raise KeyError(f"no planned_distance entries for arm {arm!r} at beta {beta} "
                       "(was the run made with --skip-neural?)")
    return [v.double() for _, v in found]                            # each (n_instances,)


def main() -> None:
    args = parse_args()
    per_inst: dict[str, Tensor] = torch.load(args.per_instance, map_location="cpu")
    keys = list(per_inst.keys())
    print(f"file: {args.per_instance}")
    for beta in _betas(keys):
        print(f"\n=== beta {beta} ===")
        for rh in args.rh:
            key = f"{beta}/{rh}/total"
            if key not in per_inst:
                raise KeyError(f"{key!r} not in file; available RH keys: "
                               f"{sorted(k for k in keys if k.endswith('/total'))}")
            ref = per_inst[key].double()                             # (n_instances,)
            print(f"  reference {rh!r}: mean {float(ref.mean()):.4f}")
            for arm in args.arm:
                seeds = _arm_seeds(per_inst, beta, arm)
                cost = torch.stack(seeds).mean(dim=0)                # (n_instances,)
                p = paired_difference(cost, ref)
                per_seed = [float((s - ref).mean()) for s in seeds]
                agree = all(x < 0 for x in per_seed) or all(x > 0 for x in per_seed)
                print(f"    {arm} - {rh}: {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
                      f" | arm {float(cost.mean()):.4f} | per seed "
                      + " ".join(f"{x:+.4f}" for x in per_seed)
                      + f" | agree {'yes' if agree else 'NO'}")


if __name__ == "__main__":
    main()