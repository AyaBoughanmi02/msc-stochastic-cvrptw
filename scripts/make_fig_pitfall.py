"""Figure 4.1: cost difference to M0 of five models, under hindsight selection
and under the planned-distance rule, one panel per noise level.

Each value is (mean cost of the model) - (mean cost of M0) under the same rule,
on the same 256 instances and 4 days, averaged over 3 seeds. It equals the mean
of the paired differences; the confidence intervals of the planned-distance
values are in section 4 of runs/apriori_15k_pd.log (not plotted).

Reads only the JSON written by ``scripts.apriori_study`` (run R1, see
``runs/apriori_15k_pd.log``); it imports nothing from the project code.

Usage (repo root, .venv active)::

    python -u -m scripts.make_fig_pitfall 2>&1 | Tee-Object -FilePath runs\\make_fig_pitfall.log

Output: ``figures/F4_1_pitfall.pdf`` (and a .png preview next to it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # no display needed; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# Arm names as written by apriori_study -> label used in the thesis (macros.tex).
LABELS: Final[dict[str, str]] = {
    "M0": "M0",
    "M": "M",
    "Const": "Const",
    "Shuffle": "Shuffle",
    "VGTR0": "R0",
    "B": "B",
}
# Two highlighted models; the others are context in light grey. M0 is the zero line.
# Colours from the validated categorical palette (slots 1, 2).
STYLE: Final[dict[str, dict[str, object]]] = {
    "M": {"color": "#2a78d6", "marker": "s", "ls": "-", "lw": 2.0, "z": 4},
    "VGTR0": {"color": "#eb6834", "marker": "D", "ls": "--", "lw": 2.0, "z": 5},
}
CONTEXT_STYLE: Final[dict[str, object]] = {
    "color": "#b5b4ae", "marker": "o", "ls": "-", "lw": 1.2, "z": 1,
}
RULES: Final[tuple[str, str]] = ("hindsight", "planned_distance")
RULE_LABELS: Final[tuple[str, str]] = ("hindsight", "planned\ndistance")


def load_costs(path: Path) -> dict[str, dict[float, dict[str, float]]]:
    """Return ``costs[arm][beta][rule]`` (seed-averaged mean cost).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError / ValueError: if the JSON does not have the expected layout,
            so a changed output format fails loudly instead of plotting nonsense.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("arms", "rules", "rows"):
        if key not in data:
            raise KeyError(f"{path}: missing top-level key '{key}'")
    for rule in RULES:
        if rule not in data["rules"]:
            raise ValueError(f"{path}: rule '{rule}' not in {data['rules']}")
    missing = set(LABELS) - set(data["arms"])
    if missing:
        raise ValueError(f"{path}: arms {sorted(missing)} not in {data['arms']}")

    costs: dict[str, dict[float, dict[str, float]]] = {}
    for row in data["rows"]:
        arm, beta = str(row["arm"]), float(row["beta"])
        costs.setdefault(arm, {})[beta] = {r: float(row["cost"][r]) for r in RULES}
    return costs


def plot(costs: dict[str, dict[float, dict[str, float]]], out: Path) -> None:
    """Draw one slope panel per beta and save ``out`` (.pdf) plus a .png."""
    betas = sorted(costs["M0"])
    if len(betas) != 3:
        raise ValueError(f"expected 3 noise levels, got {betas}")

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True)
    for ax, beta in zip(axes, betas):
        ax.axhline(0.0, color="#52514e", lw=1.4, zorder=2)
        # M0 is the reference line; label it on the left, away from the data ends.
        ax.text(-0.22, 0.0, "M0", va="bottom", ha="left", fontsize=9,
                color="#0b0b0b")
        # Context models first so the highlighted ones are drawn on top.
        order = [a for a in LABELS if a not in STYLE and a != "M0"] + list(STYLE)
        for arm in order:
            st = STYLE.get(arm, CONTEXT_STYLE)
            # (2,) difference to M0 under hindsight, under planned distance
            y = [costs[arm][beta][r] - costs["M0"][beta][r] for r in RULES]
            ax.plot(
                [0, 1], y,
                color=st["color"], marker=st["marker"], ls=st["ls"],
                lw=st["lw"], ms=5, zorder=st["z"],
            )
            if arm in STYLE:  # direct labels only for the three highlighted models
                ax.annotate(
                    LABELS[arm], (1, y[1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color="#0b0b0b",
                )
        ax.set_title(rf"$\beta = {beta:g}$", fontsize=10)
        ax.set_xticks([0, 1], RULE_LABELS, fontsize=9)
        ax.set_xlim(-0.25, 1.55)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cost difference to M0", fontsize=9)

    # Shared legend: the three highlighted models + "other models".
    handles = [
        plt.Line2D([], [], color=STYLE[a]["color"], marker=STYLE[a]["marker"],
                   ls=STYLE[a]["ls"], lw=2.0, ms=5, label=LABELS[a])
        for a in STYLE
    ]
    handles.append(plt.Line2D([], [], **{k: v for k, v in CONTEXT_STYLE.items()
                                         if k in ("color", "marker", "ls")},
                              lw=1.2, ms=5, label="Const, Shuffle, B"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=Path("runs/apriori_15k_pd.json"))
    parser.add_argument("--out", type=Path, default=Path("figures/F4_1_pitfall.pdf"))
    args = parser.parse_args()

    costs = load_costs(args.json)
    for arm in LABELS:  # echo the plotted numbers so the log is the source
        for beta, c in sorted(costs[arm].items()):
            m0 = costs["M0"][beta]
            print(f"{arm:8s} beta {beta:.1f} | hindsight {c['hindsight']:.4f}"
                  f" ({c['hindsight'] - m0['hindsight']:+.4f} vs M0)"
                  f" | planned_distance {c['planned_distance']:.4f}"
                  f" ({c['planned_distance'] - m0['planned_distance']:+.4f} vs M0)")
    plot(costs, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()