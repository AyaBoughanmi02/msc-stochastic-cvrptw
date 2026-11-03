"""Figure 4.2: test cost against the noise level for each way of entering the
variance, after 20,000 training steps with the analytic reward (left) and with
the hybrid reward (right).

Reads only JSON files written by ``scripts.apriori_study`` (rule
planned_distance, seed-averaged means); it imports nothing from the project.

Usage (repo root, .venv active)::

    python -u -m scripts.make_fig_converge 2>&1 | Tee-Object -FilePath runs\\make_fig_converge.log

Output: ``figures/F4_2_converge.pdf`` (and a .png preview).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, NamedTuple

import matplotlib

matplotlib.use("Agg")  # no display needed; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

RULE: Final[str] = "planned_distance"
BETAS: Final[tuple[float, ...]] = (0.3, 0.6, 0.9)


class Series(NamedTuple):
    """One line of the figure: which JSON, which arm in it, how to draw it."""

    json_name: str   # file in --runs
    arm: str         # arm name as written by apriori_study
    label: str       # label in the thesis
    panel: int       # 0 = analytic reward, 1 = hybrid reward
    color: str
    marker: str


# Colours: validated categorical palette, slots 1-4; the same model keeps the
# same colour and marker in both panels.
SERIES: Final[tuple[Series, ...]] = (
    Series("apriori_ft_m0.json", "M0ftA", "M0", 0, "#52514e", "o"),
    Series("apriori_ft_m.json", "M_ftA", "M", 0, "#2a78d6", "s"),
    Series("apriori_b_ftA.json", "B_ftA", "B", 0, "#eb6834", "^"),
    Series("apriori_ft_m0.json", "M0ftH", "M0", 1, "#52514e", "o"),
    Series("apriori_ft_m.json", "M_ftH", "M", 1, "#2a78d6", "s"),
    Series("apriori_b_ftH.json", "B_ftH", "B", 1, "#eb6834", "^"),
    Series("apriori_vgtr_ftH.json", "VGTR_ftH", "R", 1, "#1baf7a", "D"),
)
# The same checkpoints appear in several runs; their means must agree exactly
# (evaluation is deterministic), otherwise the runs are not comparable.
CROSS_CHECKS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("apriori_ft_m0.json", "M0ftH", "apriori_b_ftH.json", "M0_ftH"),
    ("apriori_ft_m0.json", "M0ftH", "apriori_vgtr_ftH.json", "M0_ftH"),
    ("apriori_ft_m0.json", "M0ftA", "apriori_b_ftA.json", "M0_ftA"),
    ("apriori_b_ftA.json", "B_ftH", "apriori_b_ftH.json", "B_ftH"),
)


def load(runs: Path, name: str, arm: str) -> list[float]:
    """Return the planned-distance cost of ``arm`` at each beta in BETAS.

    Raises KeyError / ValueError if the file layout or the arm is not as expected.
    """
    data = json.loads((runs / name).read_text(encoding="utf-8"))
    if arm not in data["arms"]:
        raise ValueError(f"{name}: arm '{arm}' not in {data['arms']}")
    by_beta = {float(r["beta"]): float(r["cost"][RULE])
               for r in data["rows"] if r["arm"] == arm}
    missing = [b for b in BETAS if b not in by_beta]
    if missing:
        raise ValueError(f"{name}/{arm}: no rows for beta {missing}")
    return [by_beta[b] for b in BETAS]  # (3,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures/F4_2_converge.pdf"))
    args = parser.parse_args()

    for f1, a1, f2, a2 in CROSS_CHECKS:
        c1, c2 = load(args.runs, f1, a1), load(args.runs, f2, a2)
        if any(abs(x - y) > 1e-9 for x, y in zip(c1, c2)):
            raise ValueError(f"{f1}/{a1} {c1} != {f2}/{a2} {c2}")
        print(f"check ok: {f1}/{a1} == {f2}/{a2}")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1), sharey=True)
    for s in SERIES:
        y = load(args.runs, s.json_name, s.arm)
        print(f"panel {s.panel} {s.label:6s} ({s.json_name}/{s.arm}): "
              + " / ".join(f"{v:.4f}" for v in y))
        axes[s.panel].plot(BETAS, y, color=s.color, marker=s.marker, lw=2.0,
                           ms=5, label=s.label)
    for ax, title in zip(axes, ("analytic reward", "hybrid reward")):
        ax.set_title(title, fontsize=10)
        ax.set_xticks(BETAS, [f"{b:g}" for b in BETAS], fontsize=9)
        ax.set_xlabel(r"noise level $\beta$", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean test cost", fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()  # superset of panel 0
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()