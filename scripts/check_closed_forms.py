#Table C.1 (Appendix C): the closed forms of the training rewards against Monte Carlo.

from __future__ import annotations

import argparse
import math
from typing import Final

import torch
from torch import Tensor

from src.data.simulator import apply_noise
from src.envs.cvrptw_stoch_env import _lognormal_lateness

# (mean, variance, threshold): threshold = slack K (lognormal) or deadline l_j (normal).
LOGNORMAL_ROWS: Final[list[tuple[float, float, float]]] = [
    (0.50, 0.05, 0.40), (0.50, 0.15, 0.70), (0.30, 0.10, 0.10), (1.00, 0.40, 1.50),
]
NORMAL_ROWS: Final[list[tuple[float, float, float]]] = [(1.00, 0.09, 1.20), (1.00, 0.25, 0.80)]


def _phi(z: Tensor) -> Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def main() -> None:
    p = argparse.ArgumentParser(description="Table C.1: closed forms vs Monte Carlo")
    p.add_argument("--draws", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()
    gen = torch.Generator().manual_seed(args.seed)
    dt = torch.float64
    print(f"{args.draws} draws, seed {args.seed}, float64")
    print(f"{'dist':9s} {'mean':>5s} {'var':>5s} {'thr':>5s} | "
          f"{'E cf':>7s} {'E mc':>7s} | {'P cf':>7s} {'P mc':>7s}")
    worst = 0.0
    for mu, var, k in LOGNORMAL_ROWS:
        m_, v_, k_ = (torch.tensor([x], dtype=dt) for x in (mu, var, k))
        e_cf, p_cf = _lognormal_lateness(m_, v_, k_)                       # (1,) each
        z = torch.randn(args.draws, generator=gen, dtype=dt)               # (draws,)
        x = apply_noise(m_, v_, z)                                         # (draws,)
        e_mc, p_mc = (x - k).clamp_min(0).mean(), (x > k).to(dt).mean()
        worst = max(worst, abs(float(e_cf) - float(e_mc)), abs(float(p_cf) - float(p_mc)))
        print(f"lognormal {mu:5.2f} {var:5.2f} {k:5.2f} | {float(e_cf):7.4f} {float(e_mc):7.4f} | "
              f"{float(p_cf):7.4f} {float(p_mc):7.4f}")
    for mu, var, ell in NORMAL_ROWS:
        sd = math.sqrt(var)
        z0 = torch.tensor((mu - ell) / sd, dtype=dt)
        e_cf = sd * _phi(z0) + (mu - ell) * _cdf(z0)                       # eq. (3.40)
        p_cf = _cdf(z0)
        a = mu + sd * torch.randn(args.draws, generator=gen, dtype=dt)     # (draws,)
        e_mc, p_mc = (a - ell).clamp_min(0).mean(), (a > ell).to(dt).mean()
        worst = max(worst, abs(float(e_cf) - float(e_mc)), abs(float(p_cf) - float(p_mc)))
        print(f"normal    {mu:5.2f} {var:5.2f} {ell:5.2f} | {float(e_cf):7.4f} {float(e_mc):7.4f} | "
              f"{float(p_cf):7.4f} {float(p_mc):7.4f}")
    print(f"largest |closed form - MC| = {worst:.4f}")


if __name__ == "__main__":
    main()