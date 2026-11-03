"""Figure 4.3: validation cost during training of M0 from scratch, with the
analytic reward, the hybrid reward, and the hybrid reward with gradient
clipping. Every run is drawn, including the ones that failed.

Reads only the training logs ``runs/train_*.log`` (PowerShell Tee-Object writes
UTF-16, the trainer may write UTF-8; both are handled). It parses the lines

    [eval] step   100 | val_cost 11.2953 (dist ...) | best ...

Usage (repo root, .venv active)::

    python -u -m scripts.make_fig_valcurves 2>&1 | Tee-Object -FilePath runs\\make_fig_valcurves.log

Output: ``figures/F4_3_valcurves.pdf`` (and a .png preview).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # no display needed; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

EVAL_RE: Final[re.Pattern[str]] = re.compile(r"\[eval\]\s+step\s+(\d+)\s+\|\s+val_cost\s+([0-9.]+)")
# A run counts as failed if its LAST validation cost is above this value.
# Healthy runs end near 10-11; every failed run ends at about 22.39.
FAILED_ABOVE: Final[float] = 20.0

# (panel title, [(log file, label)]); labels are the seeds.
PANELS: Final[tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = (
    ("analytic reward", (
        ("train_M0_w1_15k_s2024.log", "2024"),
        ("train_M0_w1_15k_s7.log", "7"),
        ("train_M0_w1_15k_s1234.log", "1234"),
        ("train_M0_w1_15k_s42.log", "42"),
        ("train_M0_w1_15k_s99.log", "99"),
    )),
    ("hybrid reward", (
        ("train_M0H_w1_15k_s2024.log", "2024"),
        ("train_M0H_w1_15k_s7.log", "7"),
        ("train_M0H_w1_15k_s1234.log", "1234"),
        ("train_M0H_w1_15k_s42_interrupted.log", "42 (interrupted)"),
        ("train_M0H_w1_15k_s42_rerun_collapsed.log", "42 (re-run)"),
        ("train_M0H_w1_15k_s99.log", "99"),
    )),
    ("hybrid reward, clipped", (
        ("train_M0Hc_w1_15k_s2024.log", "2024"),
        ("train_M0Hc_w1_15k_s7.log", "7"),
        ("train_M0Hc_w1_15k_s1234.log", "1234"),
        ("train_M0Hc_w1_15k_s99.log", "99"),
    )),
)
HEALTHY_COLOR: Final[str] = "#2a78d6"
FAILED_COLOR: Final[str] = "#e34948"


def read_text(path: Path) -> str:
    """Read a log written either as UTF-16 (with BOM) or as UTF-8."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def parse_curve(path: Path) -> tuple[list[int], list[float]]:
    """Return (steps, val_costs) from one training log. Raises if none is found."""
    steps: list[int] = []
    vals: list[float] = []
    for m in EVAL_RE.finditer(read_text(path)):
        steps.append(int(m.group(1)))
        vals.append(float(m.group(2)))
    if not steps:
        raise ValueError(f"{path}: no '[eval] step ... | val_cost ...' line found")
    return steps, vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures/F4_3_valcurves.pdf"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.9), sharey=True)
    for ax, (title, runs) in zip(axes, PANELS):
        n_failed = 0
        for name, seed in runs:
            steps, vals = parse_curve(args.runs / name)
            failed = vals[-1] > FAILED_ABOVE
            n_failed += failed
            print(f"{title:24s} seed {seed:18s} evals {len(vals):3d} | first {vals[0]:.4f}"
                  f" | last step {steps[-1]:5d} val {vals[-1]:.4f} | {'FAILED' if failed else 'ok'}")
            ax.plot(steps, vals, lw=1.3 if failed else 1.0,
                    ls="--" if failed else "-",
                    color=FAILED_COLOR if failed else HEALTHY_COLOR, alpha=0.9)
        ax.set_title(f"{title}\n{len(runs) - n_failed} of {len(runs)} trained", fontsize=9)
        ax.set_xlabel("training step", fontsize=9)
        ax.set_xlim(0, 15000)
        ax.set_xticks([0, 5000, 10000, 15000], ["0", "5k", "10k", "15k"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("validation cost", fontsize=9)
    handles = [plt.Line2D([], [], color=HEALTHY_COLOR, lw=1.0, label="trained normally"),
               plt.Line2D([], [], color=FAILED_COLOR, lw=1.3, ls="--", label="failed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()