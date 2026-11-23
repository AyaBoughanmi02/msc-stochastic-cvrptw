"""Draw the three result figures of Chapter 3 from the numbers in chapter4.tex.

Every value below is copied from a table of chapter4.tex (version of 11 Nov);
the table and row are given next to each block. Nothing is recomputed.
Outputs: F4_rules.pdf (was Table 3.2), F4_network.pdf (was Table 3.3),
F4_classical.pdf (was Table 3.6), plus PNG previews.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42,
})
INK2 = "#52514e"
GRID = "#e4e3df"
BETAS = ["0.3", "0.6", "0.9"]
# Noise levels are ordered, so one hue, light -> dark, plus a shape each.
BETA_COL = ["#86b4ea", "#2a78d6", "#0e3d7a"]
BETA_MK = ["o", "s", "^"]
WIDTH = 6.3  # inches, about \linewidth


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Table 3.3 (tab:network): difference with M0, 95% CI, three noise levels
#    and the degradation. (value, lo, hi); significant = CI excludes 0.
# ---------------------------------------------------------------------------
network = {
    "B (input)":              [(-0.09, -0.21, 0.03), (0.32, 0.14, 0.51), (0.66, 0.33, 0.99), (0.75, 0.41, 1.09)],
    "M (training gate)":      [(-0.32, -0.47, -0.18), (-0.31, -0.52, -0.09), (-0.34, -0.61, -0.06), (-0.01, -0.23, 0.21)],
    "R (risk term at test)":  [(-0.10, -0.26, 0.06), (-0.11, -0.33, 0.11), (0.11, -0.22, 0.43), (0.20, -0.12, 0.53)],
    "R0 (training only)":     [(0.07, -0.13, 0.27), (0.20, -0.10, 0.51), (0.02, -0.37, 0.40), (-0.06, -0.36, 0.25)],
    "Const (constant gate)":  [(-0.29, -0.42, -0.15), (-0.15, -0.37, 0.07), (-0.16, -0.45, 0.14), (0.13, -0.12, 0.38)],
    "Shuffle (permuted gate)": [(0.01, -0.11, 0.12), (-0.00, -0.16, 0.16), (-0.07, -0.29, 0.15), (-0.07, -0.27, 0.13)],
    "Const-Dec (decoder only)": [(-0.04, -0.15, 0.06), (-0.03, -0.19, 0.12), (-0.04, -0.25, 0.18), (0.01, -0.17, 0.18)],
}


def forest() -> None:
    names = list(network)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(WIDTH, 3.7), sharey=True,
        gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.08})
    n = len(names)
    for r, name in enumerate(names):
        y0 = n - 1 - r
        for b in range(3):
            v, lo, hi = network[name][b]
            y = y0 + (1 - b) * 0.24
            sig = lo > 0 or hi < 0
            ax1.plot([lo, hi], [y, y], color=BETA_COL[b], lw=1.4, solid_capstyle="round")
            ax1.plot(v, y, BETA_MK[b], ms=6, color=BETA_COL[b],
                     mfc=BETA_COL[b] if sig else "white", mew=1.4)
        v, lo, hi = network[name][3]
        sig = lo > 0 or hi < 0
        ax2.plot([lo, hi], [y0, y0], color=INK2, lw=1.4, solid_capstyle="round")
        ax2.plot(v, y0, "D", ms=5.5, color=INK2, mfc=INK2 if sig else "white", mew=1.4)
        if r % 2 == 0:
            for ax in (ax1, ax2):
                ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f4f3f0", zorder=0, lw=0)
    for ax in (ax1, ax2):
        ax.axvline(0, color="#0b0b0b", lw=0.9, zorder=1)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, n - 0.4)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(names[::-1])
    ax1.set_xlabel(r"Cost minus cost of M0  (negative = cheaper)")
    ax2.set_xlabel("Degradation minus that of M0")
    ax1.set_title("(a) Cost at each noise level", loc="left")
    ax2.set_title("(b) Degradation (0.3 to 0.9)", loc="left")
    handles = [Line2D([], [], marker=BETA_MK[b], color=BETA_COL[b], mfc=BETA_COL[b], lw=1.4,
                      label=fr"$\beta={BETAS[b]}$") for b in range(3)]
    handles += [Line2D([], [], marker="o", color=INK2, mfc=INK2, lw=0, label="significant"),
                Line2D([], [], marker="o", color=INK2, mfc="white", lw=0, label="not significant")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.55, -0.11), handletextpad=0.4, columnspacing=1.2)
    save(fig, "F4_network")


# ---------------------------------------------------------------------------
# 2. Table 3.6 (tab:classical): mean test cost and paired differences.
# ---------------------------------------------------------------------------
classical = {  # name: (costs at 0.3, 0.6, 0.9), color, linestyle, width
    "Frozen plan":              ((14.31, 17.23, 18.91), "#9a9893", (0, (4, 2)), 1.6),
    "Frozen plan, buffer":      ((12.59, 15.53, 18.35), "#9a9893", (0, (1, 1.6)), 1.8),
    "Rolling horizon":          ((12.09, 13.82, 15.14), "#2a78d6", "-", 1.6),
    "Rolling horizon, buffer":  ((11.85, 13.80, 15.40), "#2a78d6", (0, (4, 2)), 1.6),
    "M0 (ftH)":                 ((11.56, 13.30, 14.70), "#eb6834", "-", 2.4),
}
diffs = {  # M0ftH minus ...: (value, lo, hi) at 0.3, 0.6, 0.9
    "rolling horizon":         [(-0.52, -0.80, -0.25), (-0.52, -0.90, -0.13), (-0.44, -0.94, 0.06)],
    "rolling horizon, buffer": [(-0.28, -0.61, 0.05), (-0.50, -0.88, -0.12), (-0.70, -1.16, -0.24)],
}


def classical_fig() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.9),
                                   gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.42})
    x = [0, 1, 2]
    for name, (c, col, ls, lw) in classical.items():
        ax1.plot(x, c, color=col, ls=ls, lw=lw, marker="o", ms=4.5,
                 mfc=col, mec="white", mew=0.8, zorder=3 if "M0" in name else 2)
    ax1.set_xticks(x, [fr"$\beta={b}$" for b in BETAS])
    ax1.set_xlim(-0.15, 2.15)
    ax1.set_ylabel("Mean test cost")
    ax1.yaxis.grid(True, color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Mean cost", loc="left")

    cols = ["#2a78d6", "#0e3d7a"]
    mks = ["o", "s"]
    for k, (name, rows) in enumerate(diffs.items()):
        for b, (v, lo, hi) in enumerate(rows):
            xx = b + (k - 0.5) * 0.22
            sig = lo > 0 or hi < 0
            ax2.plot([xx, xx], [lo, hi], color=cols[k], lw=1.4, solid_capstyle="round")
            ax2.plot(xx, v, mks[k], ms=6, color=cols[k], mfc=cols[k] if sig else "white", mew=1.4)
    ax2.axhline(0, color="#0b0b0b", lw=0.9)
    ax2.set_xticks(x, [fr"$\beta={b}$" for b in BETAS])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylabel("M0 (ftH) minus re-planner")
    ax2.yaxis.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Paired difference, 95% CI", loc="left")
    handles = [Line2D([], [], marker=mks[k], color=cols[k], lw=1.4, label=f"vs {n}")
               for k, n in enumerate(diffs)]
    handles += [Line2D([], [], marker="o", color=INK2, mfc="white", lw=0, label="not significant")]
    ax2.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.0, -0.02), handletextpad=0.3)
    ax2.set_ylim(-1.75, 0.35)
    h1 = [Line2D([], [], color=col, ls=ls, lw=lw, marker="o", ms=4, mec="white", label=name)
          for name, (_, col, ls, lw) in classical.items()]
    fig.legend(handles=h1, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.17), columnspacing=1.4)
    save(fig, "F4_classical")


# ---------------------------------------------------------------------------
# 3. Table 3.2 (tab:rule-costs): cost under each rule minus cost under
#    PLANNED DISTANCE, for the six analytic models. Columns of the table:
#    hindsight, planned, expected, saa, average.
# ---------------------------------------------------------------------------
rule_costs = {
    "M0":      [(10.37, 11.99, 12.02, 12.11, 12.70), (11.05, 13.77, 14.00, 13.97, 14.51), (11.55, 15.12, 15.48, 15.57, 15.86)],
    "M":       [(9.81, 11.66, 11.81, 11.84, 12.39), (10.35, 13.46, 13.80, 13.81, 14.24), (10.74, 14.78, 15.29, 15.25, 15.60)],
    "Const":   [(9.76, 11.70, 11.83, 11.86, 12.38), (10.30, 13.62, 13.79, 13.68, 14.25), (10.69, 14.96, 15.23, 15.10, 15.61)],
    "Shuffle": [(10.37, 11.99, 11.93, 12.10, 12.70), (11.05, 13.77, 13.94, 14.07, 14.51), (11.55, 15.05, 15.53, 15.51, 15.87)],
    "R0":      [(9.35, 12.06, 11.99, 12.14, 12.76), (9.73, 13.97, 13.96, 14.21, 14.66), (9.99, 15.13, 15.09, 15.24, 15.85)],
    "B":       [(10.26, 11.90, 11.90, 12.08, 12.57), (11.57, 14.09, 14.40, 14.37, 14.95), (12.70, 15.78, 16.08, 16.26, 16.81)],
}
RULES = [("expected", 2, "#eb6834", "o"), ("SAA", 3, "#1baf7a", "s"), ("average candidate", 4, "#6b6a66", "^")]


def rules_fig() -> None:
    models = list(rule_costs)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    lo_h = []
    for b, ax in enumerate(axes):
        for m_i, m in enumerate(models):
            row = rule_costs[m][b]
            lo_h.append(row[0] - row[1])
            for k, (lab, col_i, col, mk) in enumerate(RULES):
                ax.plot(m_i + (k - 1) * 0.22, row[col_i] - row[1], mk, ms=5.5,
                        color=col, mec="white", mew=0.6)
        ax.axhline(0, color="#0b0b0b", lw=1.0)
        ax.set_xticks(range(len(models)), models, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(fr"$\beta={BETAS[b]}$", loc="left")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Cost minus cost under\nplanned distance (line at 0)")
    handles = [Line2D([], [], marker=mk, color=col, lw=0, ms=6, label=lab)
               for lab, _, col, mk in RULES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.16))
    save(fig, "F4_rules")
    print("hindsight minus planned: %.2f to %.2f" % (max(lo_h), min(lo_h)))


if __name__ == "__main__":
    forest()
    classical_fig()
    rules_fig()