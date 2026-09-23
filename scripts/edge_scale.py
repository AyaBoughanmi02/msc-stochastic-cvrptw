"""How large is the variance gate's penalty against the other terms in the pointer logit?

Three questions, one table:

  1. IS sigma^2 INFORMATIVE? The bank sets sigma^2 = beta_var * r * mu^2, so variance is
     mostly a squared-distance restatement unless the reliability field r decouples them.
     `corr(mu, sigma^2)` measures exactly how much of the variance signal is redundant
     with geometry. Close to 1 means a policy can look risk-aware by avoiding long edges,
     and no variance-specific inductive bias can help.

  2. IS THE GATE NUMERICALLY MEANINGFUL? The pointer ranks candidates by
     u_j = compat_j + w_edge . (mu, sigma^2)_j - lambda * sigma^2_j. What matters is not
     the absolute size of each term but its SPREAD ACROSS CANDIDATES -- a term that is
     nearly constant over j cannot re-rank anything. So the spread of each term is
     reported side by side.

  3. HOW BIG IS THE FREE PATHWAY? For variants B/C/D the decoder's w_edge has a second
     column multiplying sigma^2 with no sign constraint, competing with the gate. Its
     spread is reported next to the gate's, which is the dual-pathway problem in numbers.
     Variant D-prime has no such column and prints "--".

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from torch import Tensor

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import MU, SIGMA2, build_edge_bank, gather_from, horizon_average
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.eval.checkpoints import load_stoch_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scale of the variance gate vs the other logit terms")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--zeta", type=float, nargs="+", default=None,
                   help="also report at these idiosyncratic-risk strengths (the "
                        "checkpoint's own zeta is always included first)")
    p.add_argument("--gate-sweep", type=float, nargs="+", default=[4.0, 6.0])
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--no-rollout", action="store_true",
                   help="skip the reference measurement of the pointer's own logit spread")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def offdiag(x: Tensor) -> Tensor:
    """Flatten a (B, N, N) edge quantity, dropping the self-loop diagonal. -> (M,)."""
    n = x.size(-1)
    eye = torch.eye(n, dtype=torch.bool, device=x.device)          # (N, N)
    return x[:, ~eye].reshape(-1)                                  # (B*(N*N-N),)


def correlation(a: Tensor, b: Tensor) -> float:
    """Pearson correlation of two flat tensors."""
    a, b = a.double(), b.double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float(a.dot(b) / denom) if float(denom) > 0.0 else float("nan")


def candidate_spread(term: Tensor) -> float:
    """Mean over rows of the standard deviation ACROSS candidates. term: (B, N) -> float.

    A term with a large mean but no spread shifts every logit equally and, after the
    softmax, changes nothing. Re-ranking power lives in the spread.
    """
    return float(term.std(dim=-1, unbiased=True).mean())


def logit_spread(policy, env) -> float | None:
    """Mean spread of the pointer's tanh ARGUMENT across feasible candidates, per step.

    Uses the decoder's existing `record_diagnostics` slot, wrapping `forward` so every
    decode step is captured rather than only the last. The wrapper is removed afterwards;
    nothing about the policy is left modified.
    """
    decoder = policy.decoder
    spreads: list[float] = []
    original = decoder.forward
    was_recording = decoder.record_diagnostics
    decoder.record_diagnostics = True

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        u = decoder._last_u                                        # (B, N) or None
        if u is not None:
            mask = kwargs.get("mask")
            if mask is None and len(args) > 4:
                mask = args[4]                                     # positional `mask`
            if mask is not None:
                u = u.masked_fill(mask, float("nan"))
            # Rows with fewer than two feasible candidates have no meaningful spread.
            sd = torch.nanmean(
                torch.nan_to_num(
                    (u - u.nanmean(dim=-1, keepdim=True)).square(), nan=0.0
                ).sum(dim=-1, keepdim=True).sqrt()
            )
            if torch.isfinite(sd):
                spreads.append(float(sd))
        return out

    decoder.forward = spy
    try:
        with torch.no_grad():
            policy(env, decode_type="greedy")
    finally:
        decoder.forward = original
        decoder.record_diagnostics = was_recording
        decoder._last_u = None
    return sum(spreads) / len(spreads) if spreads else None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaded = load_stoch_policy(args.checkpoint, device)
    blob = loaded.blob

    data = generate_cvrptw(
        args.size, blob["n_customers"], blob["capacity"],
        service_time=blob["service_time"], shift_end=blob["shift_end"],
        tw_half_width=blob["tw_half_width"], seed=args.seed, device=device,
    )

    w_edge = getattr(loaded.policy.decoder, "w_edge", None)
    w = None if w_edge is None else w_edge.weight.detach()         # (1, edge_feat_dim)
    has_free_sigma = w is not None and w.size(1) > SIGMA2
    lam = (float(loaded.policy.gate_strength().detach())
           if hasattr(loaded.policy, "gate_strength") else None)

    zetas = [float(loaded.sim_cfg.zeta)]
    for z in (args.zeta or []):
        if float(z) not in zetas:
            zetas.append(float(z))

    print(f"{args.checkpoint}  (variant {loaded.variant})")
    print(f"  decoder w_edge {None if w is None else w.tolist()} | learned softplus(w) "
          f"{'n/a' if lam is None else f'{lam:.4f}'}")
    print(f"  {args.size} instances (seed {args.seed}); decoder terms read at the depot, "
          f"bin 0\n")

    header = (f"{'zeta':>6} {'beta':>6} {'corr(mu,s2)':>12} {'mean s2':>10} "
              f"{'sd_j mu-term':>13} {'sd_j s2-term':>13}"
              + "".join(f"{f'sd_j gate@{g:g}':>15}" for g in args.gate_sweep))
    print(header)
    print("-" * len(header))

    for zeta in zetas:
        for beta in args.beta_var:
            cfg = replace(loaded.sim_cfg, beta_var=beta, zeta=zeta)
            bank = build_edge_bank(data.coords, cfg)                # (B,N,N,K,2)
            hz = horizon_average(bank)                              # (B,N,N,2)
            corr = correlation(offdiag(hz[..., MU]), offdiag(hz[..., SIGMA2]))

            # What the decoder sees at the first step: edges leaving the depot, bin 0.
            depot = torch.zeros(args.size, dtype=torch.long, device=device)
            moments = gather_from(bank, depot, depot, None)         # (B, N, 2)
            mu, sigma2 = moments[..., MU], moments[..., SIGMA2]     # (B, N) each

            mu_term = (w[0, MU] * mu) if w is not None else torch.zeros_like(mu)
            s2_term = (w[0, SIGMA2] * sigma2) if has_free_sigma else None
            gates = [candidate_spread(g * sigma2) for g in args.gate_sweep]

            print(f"{zeta:>6.2f} {beta:>6.2f} {corr:>12.4f} "
                  f"{float(sigma2.mean()):>10.4f} "
                  f"{candidate_spread(mu_term):>13.4f} "
                  f"{('--' if s2_term is None else f'{candidate_spread(s2_term):.4f}'):>13} "
                  + "".join(f"{g:>15.4f}" for g in gates))

    if not args.no_rollout:
        cfg = replace(loaded.sim_cfg, beta_var=args.beta_var[0], zeta=zetas[0])
        bank = build_edge_bank(data.coords, cfg)
        env = CVRPTWStochEnv(
            data.coords, data.demand, data.capacity, data.time_windows,
            data.service_time, data.shift_end, bank, cfg,
            tardiness_weight=float(blob["tardiness_weight"]), reward_mode="analytic",
        )
        try:
            ref = logit_spread(loaded.policy, env)
        except Exception as exc:                                    # diagnostics only
            print(f"\n  (logit-spread probe failed: {exc})")
            ref = None
        if ref is not None:
            print(f"\n  REFERENCE: spread of the pointer's tanh argument across feasible "
                  f"candidates = {ref:.4f}")
            print(f"  (at zeta={zetas[0]:g}, beta={args.beta_var[0]:g}. A gate spread far "
                  f"below this cannot re-rank; far above it saturates the tanh.)")

    print("\n  corr(mu, sigma^2) near 1 => variance is a restatement of distance and no")
    print("  variance-specific bias can help. Raising zeta is what breaks that shortcut.")


if __name__ == "__main__":
    main()