"""Figure 3.3 (Chapter 3): one fixed route driven on many random days, under the three rewards.

Inference only: no training, no checkpoint, no GPU (runs on the CPU in a few seconds).
It drives the fixed route of tests/test_hybrid_reward.py (10 customers, 4 vehicles,
beta = 0.6, omega = 1) through CVRPTWStochEnv, exactly as that test does, and

  (a) plots, for 50 sampled days, the arrival time minus the window closing l_j at every
      customer of the route (above 0 = late), together with the same quantity on the
      mean clock used by the analytic reward;
  (b) plots, over 20,000 sampled days, the histogram of the lateness charged per day by the
      sampled and by the hybrid reward, with the analytic reward's single value.

It also PRINTS the numbers of Table 3.5 (means and standard deviations over the days), so
this script's log is the source of that table.

Example (PowerShell, repo root, .venv active):
    python -u -m scripts.make_fig_route_days 2>&1 | Tee-Object -FilePath runs\\make_fig_route_days.log

Output: figures/F3_3_route_days.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")                     # file output only, no window
import matplotlib.pyplot as plt           # noqa: E402
import torch                              # noqa: E402
from torch import Tensor                  # noqa: E402

from src.data.cvrptw_generator import generate_cvrptw          # noqa: E402
from src.data.simulator import SimulatorConfig, build_edge_bank  # noqa: E402
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv           # noqa: E402

# Same instance, simulator and route as tests/test_hybrid_reward.py (keep them in sync).
SIM: Final = SimulatorConfig(alpha=0.3, beta_var=0.6, eta=0.6, omega=1.0)
ROUTE: Final[list[int]] = [3, 5, 0, 1, 2, 0, 4, 6, 7, 0, 8, 9, 10, 0]   # 10 customers, 4 trips
INSTANCE_SEED: Final[int] = 3
N_CUSTOMERS: Final[int] = 10
CAPACITY: Final[int] = 20
TARDINESS_WEIGHT: Final[float] = 2.0

BLUE: Final[str] = "#2a6fdb"     # sampled
ORANGE: Final[str] = "#eb6834"   # hybrid
_FIG_W, _FIG_H = 6.3, 2.5        # full text width, two panels


class _LoggingEnv(CVRPTWStochEnv):
    """CVRPTWStochEnv that also records the realized arrival time and l_j at every step."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.arrivals: list[Tensor] = []   # each (B,)
        self.latest: list[Tensor] = []     # each (B,)

    def _on_arrival(self, action: Tensor, arrival: Tensor) -> None:
        idx = torch.arange(self.batch_size, device=self.coords.device)
        self.arrivals.append(arrival.clone())                       # (B,)
        self.latest.append(self.time_windows[idx, action, 1].clone())  # (B,)
        super()._on_arrival(action, arrival)


def drive(mode: str, days: int, seed: int) -> tuple[dict[str, Tensor], Tensor]:
    """Drive ROUTE on `days` independent days. Returns cost components and a - l, (days, T)."""
    data = generate_cvrptw(1, N_CUSTOMERS, CAPACITY, seed=INSTANCE_SEED)
    rep = lambda t: t.repeat(days, *([1] * (t.dim() - 1)))       # noqa: E731
    bank = build_edge_bank(data.coords, SIM).repeat(days, 1, 1, 1, 1)   # (D, N, N, K, 2)
    n = data.coords.size(1)
    noise = torch.randn(days, n, n, SIM.n_bins,
                        generator=torch.Generator().manual_seed(seed))  # (D, N, N, K)
    env = _LoggingEnv(rep(data.coords), rep(data.demand), data.capacity,
                      rep(data.time_windows), rep(data.service_time), data.shift_end,
                      bank=bank, sim_cfg=SIM, noise=noise,
                      tardiness_weight=TARDINESS_WEIGHT, reward_mode=mode)
    actions = torch.tensor([ROUTE]).repeat(days, 1)                # (D, T)
    env.reset()
    for t in range(actions.size(1)):
        env.step(actions[:, t])
    gap = torch.stack(env.arrivals, 1) - torch.stack(env.latest, 1)  # (D, T)
    return env.cost_components(actions), gap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 3.3: one route on many days")
    p.add_argument("--days", type=int, default=20_000, help="days for panel (b) and Table 3.5")
    p.add_argument("--shown-days", type=int, default=50, help="days drawn in panel (a)")
    p.add_argument("--seed", type=int, default=0, help="noise seed (0 = the test's seed)")
    p.add_argument("--out", type=str, default="figures/F3_3_route_days.pdf")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sampled, gap_s = drive("sampled", args.days, args.seed)
    hybrid, _ = drive("hybrid", args.days, args.seed)
    analytic, gap_a = drive("analytic", 1, args.seed)

    # ---- Table 3.5 numbers (this log is their source)
    print(f"route {ROUTE} | instance seed {INSTANCE_SEED} | {N_CUSTOMERS} customers | "
          f"beta {SIM.beta_var} | omega {SIM.omega} | {args.days} days | noise seed {args.seed}")
    for key in ("tardiness", "late_count"):
        s, h, a = sampled[key], hybrid[key], analytic[key]
        print(f"{key:10s} sampled mean {s.mean():.4f} sd {s.std():.4f} | "
              f"hybrid mean {h.mean():.4f} sd {h.std():.4f} | analytic {a.item():.4f} | "
              f"sd reduction {1 - h.std() / s.std():.1%} | variance removed "
              f"{1 - h.var() / s.var():.1%}")
    print(f"distance   {sampled['distance'][0].item():.4f} (identical in all modes: "
          f"{bool(torch.equal(sampled['distance'], hybrid['distance']))})")

    # ---- Figure
    is_cust = torch.tensor(ROUTE) != 0                              # (T,)
    cust_pos = [i for i, c in enumerate(ROUTE) if c != 0]           # step index of each customer
    x = list(range(1, len(cust_pos) + 1))                           # customer order on the route
    trips: list[list[int]] = []                                     # x positions per trip
    cur: list[int] = []
    for i, c in enumerate(ROUTE):
        if c != 0:
            cur.append(cust_pos.index(i) + 1)
        elif cur:
            trips.append(cur)
            cur = []

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H),
                                     gridspec_kw={"width_ratios": [1.25, 1]})
    g_s = gap_s[: args.shown_days][:, is_cust]                       # (shown, C)
    g_a = gap_a[0, is_cust]                                          # (C,)
    for d in range(g_s.size(0)):
        for tr in trips:                                             # no line across a depot
            ax_a.plot(tr, g_s[d, [t - 1 for t in tr]].tolist(), color=BLUE, lw=0.6, alpha=0.25)
    for tr in trips:
        ax_a.plot(tr, g_a[[t - 1 for t in tr]].tolist(), color="black", lw=1.6, ls="--",
                  marker="o", ms=3)
    ax_a.axhline(0.0, color="#888888", lw=0.8)
    ax_a.text(3.0, 0.08, "window closes", fontsize=7, ha="center", va="bottom",
              color="#555555", bbox={"fc": "white", "ec": "none", "pad": 0.5})
    for tr in trips[1:]:
        ax_a.axvline(tr[0] - 0.5, color="#cccccc", lw=0.8, ls=":")
    ax_a.set_xticks(x)
    ax_a.set_xlabel("customer, in route order (dotted: new vehicle)", fontsize=8)
    ax_a.set_ylabel(r"arrival $-\ \ell_j$", fontsize=9)
    ax_a.tick_params(labelsize=8)
    ax_a.plot([], [], color=BLUE, lw=1.0, label=f"{args.shown_days} sampled days")
    ax_a.plot([], [], color="black", lw=1.6, ls="--", label="mean clock (analytic)")
    ax_a.legend(fontsize=7, loc="upper left", frameon=False)

    s_t = sampled["tardiness"].numpy()
    h_t = hybrid["tardiness"].numpy()
    hi = float(torch.quantile(sampled["tardiness"], 0.99))         # cut the long right tail
    bins = [hi * i / 40 for i in range(41)]
    print(f"panel (b) x-axis cut at the 99th percentile of the sampled charge: {hi:.2f} "
          f"(max {s_t.max():.2f})")
    ax_b.hist(s_t, bins=bins, density=True, histtype="step", color=BLUE, lw=1.4,
              label="sampled")
    ax_b.hist(h_t, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.4,
              label="hybrid")
    ax_b.axvline(float(analytic["tardiness"]), color="black", lw=1.4, ls="--",
                 label="analytic")
    ax_b.axvline(float(s_t.mean()), color="#555555", lw=0.8, ls=":", label="true mean")
    ax_b.set_xlim(0.0, hi)
    ax_b.set_xlabel("lateness charged per day", fontsize=8)
    ax_b.set_ylabel("density", fontsize=9)
    ax_b.tick_params(labelsize=8)
    ax_b.legend(fontsize=7, frameon=False)
    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)
    ax_a.set_title("(a)", fontsize=9)
    ax_b.set_title("(b)", fontsize=9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()