"""E4: did the hybrid reward learn faster during fine-tuning? Read from the existing logs.

No GPU, no new runs. Every fine-tuning run already validated every 100 steps on the
SAME 256 validation instances (seed 9999) and the SAME 2 days, at beta 0.3, choosing the
best start on the realised day (the trainer's monitor, not the paper's test protocol).
All four arms of a seed start from the same checkpoint (M0_w1_15k_s<seed>_last.pt), so
their curves are paired.

Usage (repo root, .venv active):
    python -u -m scripts.e4_val_curves 2>&1 | Tee-Object -FilePath runs\\e4_val_curves.log

For every seed and arm it prints the mean validation cost over steps 100-1000 (early),
over all steps (area under the curve / steps) and at the last step, then each arm minus
the hybrid arm. Descriptive only: three seeds, one beta, a 256-instance monitor.

READING (fixed before running):
  R4a  report "the hybrid reward learned faster than the sampled one" only if the EARLY
       mean of M0_ftS minus M0_ftH is positive on all three seeds; and say in the paper
       that this is a training monitor, not the test protocol. Otherwise no claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from scripts.make_fig_valcurves import parse_curve

RUNS: Final[Path] = Path("runs")
SEEDS: Final[tuple[int, ...]] = (2024, 7, 1234)
ARMS: Final[tuple[str, ...]] = ("M0_ftH", "M0_ftS", "M0_ftS4", "M0_ftA")  # ref first
EARLY_END: Final[int] = 1000


def summarise(steps: list[int], vals: list[float]) -> dict[str, float]:
    """Early mean, whole-run mean and final validation cost of one curve."""
    early = [v for s, v in zip(steps, vals) if s <= EARLY_END]
    if not early:
        raise ValueError(f"no validation point at or before step {EARLY_END}")
    return {"early": sum(early) / len(early), "whole": sum(vals) / len(vals),
            "final": vals[-1], "last_step": float(steps[-1])}


def main() -> None:
    table: dict[str, dict[int, dict[str, float]]] = {}
    for arm in ARMS:
        table[arm] = {}
        for seed in SEEDS:
            path = RUNS / f"train_{arm}_s{seed}.log"
            steps, vals = parse_curve(path)
            table[arm][seed] = summarise(steps, vals)
    last = {table[a][s]["last_step"] for a in ARMS for s in SEEDS}
    if len(last) != 1:
        raise RuntimeError(f"runs end at different steps: {sorted(last)} (budget mismatch)")

    print(f"validation cost (beta 0.3, 256 instances, 2 days); early = steps 100-{EARLY_END}\n")
    print(f"{'arm':>8} {'seed':>5} {'early':>8} {'whole':>8} {'final':>8}")
    for arm in ARMS:
        for seed in SEEDS:
            r = table[arm][seed]
            print(f"{arm:>8} {seed:>5} {r['early']:>8.4f} {r['whole']:>8.4f} {r['final']:>8.4f}")

    ref = ARMS[0]
    print(f"\neach arm minus {ref} (positive: the hybrid arm is cheaper)")
    print(f"{'arm':>8} {'window':>6}  per seed {SEEDS}            mean   positive")
    out: dict = {"per_run": {a: {str(s): v for s, v in d.items()} for a, d in table.items()},
                 "diff": {}}
    for arm in ARMS[1:]:
        out["diff"][arm] = {}
        for win in ("early", "whole", "final"):
            d = [table[arm][s][win] - table[ref][s][win] for s in SEEDS]
            m = sum(d) / len(d)
            pos = sum(x > 0 for x in d)
            out["diff"][arm][win] = {"per_seed": d, "mean": m, "positive": pos}
            print(f"{arm:>8} {win:>6}  " + " ".join(f"{x:+.4f}" for x in d)
                  + f"   {m:+.4f}   {pos}/{len(d)}")

    r4a = all(x > 0 for x in out["diff"]["M0_ftS"]["early"]["per_seed"])
    print(f"\nREADING (fixed before running)\n  R4a hybrid learned faster than sampled "
          f"(early window, all seeds): {r4a}")
    (RUNS / "e4_val_curves.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("  wrote runs/e4_val_curves.json")


if __name__ == "__main__":
    main()
