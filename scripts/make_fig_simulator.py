"""Figure 3.1 (Chapter 3): the congestion ripple, and how omega decouples variance from distance.

Inference only: no training, no GPU. It rebuilds the Edge Bank of the test instances with the
EXACT simulator settings stored in a trained checkpoint, then

  (a) plots mu_ij(tau_k) / d_ij over the shift for three roads at different distances from
      the centre of the instance (the rush hour reaches outer roads later);
  (b) plots the day-averaged variance against the day-averaged mean for every road, at
      omega = 0 and omega = 1, with the Pearson correlation in the legend.

It also PRINTS what the thesis text needs: the simulator constants (phase_k = c_phi,
field_scale = s_g, eta, risk_freq, ...), a check of the mean-preservation identity (3.16),
and exactly how the correlation is computed (which roads, how many instances).

Example (PowerShell, repo root, .venv active):
    python -u -m scripts.make_fig_simulator --ckpt checkpoints\\M0_w1_15k_s2024_last.pt `
        2>&1 | Tee-Object -FilePath runs\\make_fig_simulator.log

Outputs: figures/F1a_ripple.pdf, figures/F1b_corr.pdf.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # file output only, no window
import matplotlib.pyplot as plt           # noqa: E402
import torch                              # noqa: E402
from torch import Tensor                  # noqa: E402

from src.data.cvrptw_generator import generate_cvrptw, pairwise_distances  # noqa: E402
from src.data.simulator import (          # noqa: E402
    MU,
    SIGMA2,
    SimulatorConfig,
    build_edge_bank,
    horizon_average,
)

# Single-column width of a LaTeX page with 2.5 cm margins is about 6.3 in; two panels side by side.
_FIG_W, _FIG_H = 3.1, 2.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 3.1: simulator ripple and omega")
    p.add_argument("--ckpt", type=str, required=True,
                   help="any trained checkpoint; only its simulator settings are read")
    p.add_argument("--instances", type=int, default=256)
    p.add_argument("--seed", type=int, default=4242, help="4242 = the test instances")
    p.add_argument("--beta", type=float, default=0.3,
                   help="noise level for the plotted variances (the correlation does not "
                        "depend on it: every variance is multiplied by the same beta)")
    p.add_argument("--scatter-instances", type=int, default=16,
                   help="instances shown in panel (b); the correlation uses all of them")
    p.add_argument("--out-dir", type=str, default="figures")
    return p.parse_args()


def load_checkpoint_settings(path: str) -> tuple[SimulatorConfig, dict]:
    """Rebuild the SimulatorConfig stored in a checkpoint. Crashes if it is missing."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    raw = blob.get("sim_config")
    if not isinstance(raw, dict):
        raise KeyError(f"{path} has no 'sim_config' dict (found {type(raw).__name__})")
    for key in ("n_customers", "capacity", "service_time", "shift_end", "tw_half_width"):
        if key not in blob:
            raise KeyError(f"{path} has no '{key}' entry")
    return SimulatorConfig(**raw), blob


def edge_radius(coords: Tensor) -> Tensor:
    """Distance from each road's midpoint to the instance centroid. (B, N, 2) -> (B, N, N).

    Same definition as `build_edge_bank` (rho_ij in the thesis).
    """
    centre = coords.mean(dim=1)                                        # (B, 2)
    midpoint = 0.5 * (coords.unsqueeze(2) + coords.unsqueeze(1))       # (B, N, N, 2)
    return (midpoint - centre[:, None, None, :]).norm(dim=-1)          # (B, N, N)


def upper_pairs(x: Tensor) -> Tensor:
    """Values on the roads i < j, flattened over the batch. (B, N, N) -> (B * N(N-1)/2,).

    The bank is symmetric, so i < j keeps each road once; the diagonal (self-loops,
    mu = sigma^2 = 0) is excluded.
    """
    n = x.size(1)
    iu = torch.triu_indices(n, n, offset=1, device=x.device)          # (2, N(N-1)/2)
    return x[:, iu[0], iu[1]].reshape(-1)


def pearson(x: Tensor, y: Tensor) -> float:
    """Pearson correlation of two 1-D tensors, in float64."""
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError(f"need two 1-D tensors of equal length, got {x.shape}, {y.shape}")
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    return float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()))


def main() -> None:
    args = parse_args()
    cfg, blob = load_checkpoint_settings(args.ckpt)
    cfg = replace(cfg, beta_var=args.beta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("simulator settings read from", args.ckpt)
    for key, val in asdict(cfg).items():
        print(f"  {key:>12} = {val}")
    print("  (beta_var overridden for this figure; omega set to 0 and 1 below)\n")

    data = generate_cvrptw(
        args.instances, int(blob["n_customers"]), blob["capacity"],
        service_time=float(blob["service_time"]), shift_end=float(blob["shift_end"]),
        tw_half_width=float(blob["tw_half_width"]), seed=args.seed, device="cpu",
    )
    coords = data.coords.double()                                       # (B, N, 2)
    d = pairwise_distances(coords)                                      # (B, N, N)

    banks = {w: build_edge_bank(coords, replace(cfg, omega=w)) for w in (0.0, 0.5, 1.0)}
    avg = {w: horizon_average(b) for w, b in banks.items()}             # (B, N, N, 2)

    # Check of (3.16): the day-average of the mean equals the distance.
    err = float((avg[1.0][..., MU] - d).abs().max())
    print(f"mean preservation (3.16): max |mean_k mu - d| = {err:.2e} "
          f"(is_mean_preserving = {cfg.is_mean_preserving})")

    # Check that at omega = 1 the variance does not depend on the time bin.
    s2 = banks[1.0][..., SIGMA2]                                        # (B, N, N, K)
    spread = float((s2.amax(dim=-1) - s2.amin(dim=-1)).abs().max())
    print(f"omega = 1: max over roads of (max_k - min_k) sigma^2 = {spread:.2e}\n")

    mu_bar = upper_pairs(avg[0.0][..., MU])
    corr = {w: pearson(mu_bar, upper_pairs(avg[w][..., SIGMA2])) for w in (0.0, 0.5, 1.0)}
    n_roads = mu_bar.numel()
    print("CORRELATION corr(mu_bar, sigma_bar^2): Pearson, pooled over all roads i < j "
          f"(depot included) of {args.instances} instances (seed {args.seed}) = "
          f"{n_roads} roads; day averages over the {cfg.n_bins} bins")
    for w, c in corr.items():
        print(f"  omega = {w:g}: {c:.4f}")

    # ---------------------------------------------------------------- panel (a)
    rho = upper_pairs(edge_radius(coords[:1]))                          # roads of instance 0
    ratio = (banks[1.0][0, ..., MU] / d[0].clamp_min(1e-12).unsqueeze(-1))  # (N, N, K)
    n = coords.size(1)
    iu = torch.triu_indices(n, n, offset=1)
    order = torch.argsort(rho)
    picks = [int(order[int(q * (len(order) - 1))]) for q in (0.05, 0.5, 0.95)]
    tau = (torch.arange(cfg.n_bins, dtype=torch.float64) + 0.5) * cfg.bin_width

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for k, style in zip(picks, ("-", "--", ":")):
        i, j = int(iu[0, k]), int(iu[1, k])
        ax.plot(tau.numpy(), ratio[i, j].numpy(), style, marker="o", markersize=2,
                label=rf"$\rho_{{ij}}={float(rho[k]):.2f}$")
    ax.axhline(1.0, color="grey", linewidth=0.6)
    ax.set_xlabel(r"time of day $\tau$ (time units)")
    ax.set_ylabel(r"$\mu_{ij}(\tau)\,/\,d_{ij}$")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "F1a_ripple.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- panel (b)
    m = min(args.scatter_instances, args.instances)
    x = upper_pairs(avg[0.0][:m, ..., MU])
    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H), sharex=True)
    for ax, w in zip(axes, (0.0, 1.0)):
        y = upper_pairs(avg[w][:m, ..., SIGMA2])
        ax.scatter(x.numpy(), y.numpy(), s=1, alpha=0.4, rasterized=True,
                   label=f"corr = {corr[w]:.2f}")
        ax.set_title(rf"$\omega={w:g}$", fontsize=9)
        ax.set_xlabel(r"$\bar\mu_{ij}$")
        ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=0)
    axes[0].set_ylabel(r"$\bar\sigma^2_{ij}$")
    fig.tight_layout()
    fig.savefig(out_dir / "F1b_corr.pdf", dpi=300)
    plt.close(fig)

    print(f"\nwrote {out_dir / 'F1a_ripple.pdf'} and {out_dir / 'F1b_corr.pdf'}")
    if not math.isfinite(corr[0.0]) or not math.isfinite(corr[1.0]):
        raise RuntimeError("non-finite correlation")


if __name__ == "__main__":
    main()