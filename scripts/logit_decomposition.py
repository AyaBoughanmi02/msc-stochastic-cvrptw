"""What drives the pointer's decision? Decompose the tanh argument, measure saturation.

The pointer scores candidate j (PointerDecoder, edge_bias_mode='pre_tanh'):

    u_j     = compat_j + w_edge . e_j(tau)  -  gate_j
    logit_j = 10 * tanh(u_j)              -> mask -> softmax

Questions (no training, existing checkpoints):
  Q1 DOMINANCE   Which term decides? Spread ACROSS OPEN CANDIDATES of compat, edge bias
                 and gate, and how often argmax(u) equals argmax(compat) / argmax(edge).
  Q2 SATURATION  Is the tanh flat where it matters? Share of open candidates with
                 |u| > 2 (tanh' < 0.07), and whether the TWO best candidates are both
                 saturated (then the policy cannot tell them apart).
  Q3 GATE AS A TEMPERATURE  For gated checkpoints, the SAME states are scored again
                 with the gate removed: entropy change, KL and greedy-choice flips. A
                 constant gate (DpConst) can only act through the tanh, so dH != 0 with
                 flip ~ 0 means "temperature, not ranking".

Readings fixed before running:
  * compat spread >> edge and gate spreads, argmax(u) == argmax(compat) in most steps
    => compat drives the decision; the extra terms refine it (Q1 answered "no").
  * top-2 saturated in a large share of steps => the tanh clip muzzles the decision
    itself; otherwise saturation only touches already-losing candidates (harmless).
  * DpConst: |dH| clearly > 0 with a low flip rate => the decoder gate is a training-
    time temperature; dH < 0 sharper, dH > 0 softer.

Rollouts use the TRAINING condition: analytic env (mean clock), sampling, all starts,
no augmentation, BatchNorm in eval mode. Every gated checkpoint runs at its LEARNED
lambda; the gate-off pass is computed on the same state inside the same step.

Example (CPU, while the GPU trains):
    python -u -m scripts.logit_decomposition --device cpu \
        --arm M0 ckpt_s2024 ckpt_s7 ckpt_s1234 --arm M ... --arm DpConst ... --arm B ... \
        --out runs/logit_decomp 2>&1 | Tee-Object -FilePath runs/logit_decomp.console.log
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from scripts.bn_mechanism import _tee_console
from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SIGMA2
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import load_stoch_policy
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env

SAT = 2.0          # |u| above which 10*tanh(u) is flat: tanh'(2) = 0.07
_KEYS = ("std_compat", "std_edge", "std_edge_sigma2", "std_gate", "std_u", "mean_u",
         "sat_open", "top2_sat", "slope_top", "h_norm", "agree_compat", "agree_edge",
         "gate_const", "d_entropy", "kl", "flip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decompose the pointer logit; measure saturation")
    p.add_argument("--arm", nargs="+", action="append", required=True,
                   metavar=("NAME", "CKPT"), help="arm name then its checkpoints (one per seed)")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--decode", type=str, default="sampling", choices=["sampling", "greedy"])
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--out", type=str, default="runs/logit_decomp")
    add_budget_argument(p)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- statistics


def _mstd(x: Tensor, keep: Tensor, k: Tensor) -> Tensor:
    """Population std of x over the open candidates of each row. (R, N) -> (R,)."""
    mean = (x * keep).sum(dim=1) / k
    return ((((x - mean.unsqueeze(1)) ** 2) * keep).sum(dim=1) / k).sqrt()


def _margmax(x: Tensor, keep: Tensor) -> Tensor:
    """argmax over open candidates. (R, N) -> (R,)."""
    return x.masked_fill(~keep, float("-inf")).argmax(dim=1)


def _entropy(log_p: Tensor, keep: Tensor) -> Tensor:
    """Policy entropy over open candidates (nats). (R, N) -> (R,)."""
    return -torch.where(keep, log_p.exp() * log_p, torch.zeros_like(log_p)).sum(dim=1)


class StepStats:
    """Accumulates per-row statistics over every decode step of every rollout."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {k: 0.0 for k in _KEYS}
        self.seen: set[str] = set()
        self.rows = 0
        self.gated_rows = 0

    def add(self, name: str, values: Tensor) -> None:
        self.sums[name] += float(values.sum())
        self.seen.add(name)

    def result(self) -> dict[str, float | None]:
        gated_only = {"std_gate", "gate_const", "d_entropy", "kl", "flip"}
        out: dict[str, float | None] = {}
        for k, v in self.sums.items():
            n = self.gated_rows if k in gated_only else self.rows
            out[k] = v / n if n and k in self.seen else None
        out["rows"] = float(self.rows)
        return out


def _make_hook(stats: StepStats, sigma2_col: int | None):
    """Forward hook on PointerDecoder: decompose u for every row with >= 2 open nodes."""

    def hook(module, args, kwargs, output: Tensor) -> None:
        if module.edge_bias_mode != "pre_tanh":
            raise ValueError("decomposition assumes edge_bias_mode='pre_tanh'")
        mask: Tensor = args[4] if len(args) > 4 else kwargs["mask"]          # (R, N)
        keep = ~mask
        rows = keep.sum(dim=1) >= 2
        if not bool(rows.any()):
            return
        u: Tensor = module._last_u                                           # (R, N)
        edge_feat: Tensor | None = kwargs.get("edge_feat")                   # (R, N, F)
        gate: Tensor | None = kwargs.get("gate_penalty")                     # (R, N)
        zeros = torch.zeros_like(u)
        edge = module.w_edge(edge_feat).squeeze(-1) if edge_feat is not None else zeros
        g = gate if gate is not None else zeros
        compat = u - edge + g                                                # exact

        keep, u, edge, g, compat = keep[rows], u[rows], edge[rows], g[rows], compat[rows]
        log_p = output[rows]
        k = keep.sum(dim=1).to(u.dtype)                                      # (r,)
        stats.rows += int(rows.sum())
        stats.add("std_compat", _mstd(compat, keep, k))
        stats.add("std_edge", _mstd(edge, keep, k))
        if sigma2_col is not None and edge_feat is not None:
            part = module.w_edge.weight[0, sigma2_col] * edge_feat[rows][..., sigma2_col]
            stats.add("std_edge_sigma2", _mstd(part, keep, k))
        stats.add("std_u", _mstd(u, keep, k))
        stats.add("mean_u", (u * keep).sum(dim=1) / k)
        stats.add("sat_open", ((u.abs() > SAT) & keep).sum(dim=1) / k)
        top2 = u.masked_fill(~keep, float("-inf")).topk(2, dim=1).values    # (r, 2)
        stats.add("top2_sat", (top2.abs() > SAT).all(dim=1).to(u.dtype))
        stats.add("slope_top", torch.cosh(top2[:, 0]).pow(-2))
        stats.add("h_norm", _entropy(log_p, keep) / k.log())
        best = _margmax(u, keep)
        stats.add("agree_compat", (best == _margmax(compat, keep)).to(u.dtype))
        stats.add("agree_edge", (best == _margmax(edge, keep)).to(u.dtype))

        if gate is not None:
            # Same state, gate removed: forward() bypasses hooks, so no recursion.
            log_p0 = module.forward(*args, **{**kwargs, "gate_penalty": None})[rows]
            stats.gated_rows += int(rows.sum())
            stats.add("std_gate", _mstd(g, keep, k))
            spread = (g.masked_fill(~keep, float("-inf")).amax(dim=1)
                      - g.masked_fill(~keep, float("inf")).amin(dim=1))
            stats.add("gate_const", (spread < 1e-6).to(u.dtype))
            h, h0 = _entropy(log_p, keep), _entropy(log_p0, keep)
            stats.add("d_entropy", (h - h0) / k.log())
            stats.add("kl", torch.where(keep, log_p.exp() * (log_p - log_p0),
                                        torch.zeros_like(log_p)).sum(dim=1))
            stats.add("flip", (best != _margmax(log_p0, keep)).to(u.dtype))

    return hook


# --------------------------------------------------------------------------- driver


def _slice(data: CVRPTWData, sl: slice) -> CVRPTWData:
    return CVRPTWData(
        coords=data.coords[sl], demand=data.demand[sl], capacity=data.capacity,
        time_windows=data.time_windows[sl], service_time=data.service_time[sl],
        shift_end=data.shift_end,
    )


@torch.no_grad()
def measure(loaded, data: CVRPTWData, beta: float, args) -> dict[str, float | None]:
    """All statistics for one checkpoint at one variance level."""
    policy = loaded.policy
    sim = replace(loaded.sim_cfg, beta_var=beta)
    weight = float(loaded.blob["tardiness_weight"])
    width = policy.decoder.edge_feat_dim
    stats = StepStats()
    handle = policy.decoder.register_forward_hook(
        _make_hook(stats, SIGMA2 if width > SIGMA2 else None), with_kwargs=True
    )
    policy.decoder.record_diagnostics = True
    try:
        b, n, _ = data.coords.shape
        s = n - 1
        for c, i in enumerate(range(0, b, args.eval_batch_size)):
            part = _slice(data, slice(i, i + args.eval_batch_size))
            env = build_stoch_env(part, sim, s, weight, reward_mode="analytic")
            starts = build_starts(part.coords.size(0), s, data.coords.device)
            torch.manual_seed(args.sample_seed + c)        # same draws for every arm
            policy(env, decode_type=args.decode, force_first=starts)
    finally:
        policy.decoder.record_diagnostics = False
        handle.remove()
    return stats.result()


def main() -> None:
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)

    arms: list[tuple[str, list]] = []
    for spec in args.arm:
        if len(spec) < 2:
            raise ValueError(f"--arm needs a name and at least one checkpoint: {spec}")
        arms.append((spec[0], [(p, load_stoch_policy(p, device)) for p in spec[1:]]))
    assert_matched_budget([(p, l.blob) for _, ls in arms for p, l in ls],
                          allow=args.allow_budget_mismatch)
    ref = arms[0][1][0][1].blob
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    print(f"{args.test_size} instances (seed {args.test_seed}) | analytic env | "
          f"{args.decode} | all starts | gated arms at their LEARNED lambda")
    for name, ls in arms:
        lams = [float(l.policy.gate_strength()) if hasattr(l.policy, "gate_w") else None
                for _, l in ls]
        print(f"  {name}: variant {ls[0][1].variant} | learned lambda "
              + ", ".join("-" if x is None else f"{x:.3f}" for x in lams))

    results: dict = {}
    t0 = time.time()
    for name, ls in arms:
        results[name] = {}
        for beta in args.beta_var:
            per_seed = [measure(l, data, beta, args) for _, l in ls]
            results[name][str(beta)] = per_seed
            print(f"  {name} beta {beta:g} done | {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    def avg(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else None

    def fmt(x: float | None, spec: str) -> str:
        return "-" if x is None else format(x, spec)

    print("\n1. WHICH TERM DECIDES? spread across open candidates (seed-averaged)")
    head = (f"{'arm':>10} {'beta':>5} {'compat':>7} {'edge':>7} {'e.s2':>7} {'gate':>7} "
            f"{'u':>7} {'mean u':>7} {'=compat':>8} {'=edge':>7}")
    print(head + "\n" + "-" * len(head))
    for name, by_beta in results.items():
        for beta, rows in by_beta.items():
            print(f"{name:>10} {beta:>5} {fmt(avg(rows, 'std_compat'), '7.3f')} "
                  f"{fmt(avg(rows, 'std_edge'), '7.3f')} "
                  f"{fmt(avg(rows, 'std_edge_sigma2'), '7.3f')} "
                  f"{fmt(avg(rows, 'std_gate'), '7.3f')} {fmt(avg(rows, 'std_u'), '7.3f')} "
                  f"{fmt(avg(rows, 'mean_u'), '+7.2f')} {fmt(avg(rows, 'agree_compat'), '8.1%')} "
                  f"{fmt(avg(rows, 'agree_edge'), '7.1%')}")

    print("\n2. SATURATION (|u| > 2: tanh' < 0.07)")
    head = (f"{'arm':>10} {'beta':>5} {'open sat':>9} {'top-2 sat':>10} "
            f"{'slope@top':>10} {'H/Hmax':>7}")
    print(head + "\n" + "-" * len(head))
    for name, by_beta in results.items():
        for beta, rows in by_beta.items():
            print(f"{name:>10} {beta:>5} {fmt(avg(rows, 'sat_open'), '9.1%')} "
                  f"{fmt(avg(rows, 'top2_sat'), '10.1%')} "
                  f"{fmt(avg(rows, 'slope_top'), '10.3f')} {fmt(avg(rows, 'h_norm'), '7.3f')}")

    print("\n3. THE GATE ON THE SAME STATES (gated arms; gate on minus gate off)")
    head = (f"{'arm':>10} {'beta':>5} {'gate const':>11} {'dH/Hmax':>9} {'KL':>8} "
            f"{'flip':>7}   per-seed dH/Hmax")
    print(head + "\n" + "-" * len(head))
    for name, by_beta in results.items():
        for beta, rows in by_beta.items():
            if avg(rows, "d_entropy") is None:
                continue
            seeds = " ".join(fmt(r["d_entropy"], "+.4f") for r in rows)
            print(f"{name:>10} {beta:>5} {fmt(avg(rows, 'gate_const'), '11.1%')} "
                  f"{fmt(avg(rows, 'd_entropy'), '+9.4f')} {fmt(avg(rows, 'kl'), '8.4f')} "
                  f"{fmt(avg(rows, 'flip'), '7.1%')}   {seeds}")

    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    print(f"\n  wrote {args.out}.json")


if __name__ == "__main__":
    main()