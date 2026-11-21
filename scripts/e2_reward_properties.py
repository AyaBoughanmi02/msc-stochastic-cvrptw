"""E2: how the three training rewards behave on the routes of a trained policy.

Example (PowerShell, one line):
    python -u -m scripts.e2_reward_properties
        --ckpt checkpoints\\M0_ftH_s2024_last.pt checkpoints\\M0_ftH_s7_last.pt
               checkpoints\\M0_ftH_s1234_last.pt
        --out runs\\e2_reward_properties

Inference only (the gradient part calls backward, but no optimizer and no weight change).
For every checkpoint (one per seed) and every noise level beta:

PART A -- COST LEVEL, greedy routes, `--days` days per instance (seeds 20000+d: disjoint
from the test days 0-3 and the selection days 10000+).
  On each day the policy drives its routes in reward_mode='hybrid' (the REAL clock). The
  SAME actions are then replayed in a 'sampled' env on the SAME noise, so C^H and C^S are
  charged on identical routes: their difference is the reward alone, not the routing.
  A third rollout in 'analytic' mode (mean clock, no noise) gives C^A once per start.
  (A1) unbiasedness   mean over days of C^H - C^S            (Prop. 1: zero)
  (A2) cost variance  Var_days(C^H) / Var_days(C^S) per (instance, start)
  (A3) advantage      same ratio for the POMO advantage C_s - mean_s C_s (what multiplies
                      the score in the gradient)
  (A4) analytic bias  C^A - mean_days C^S: what the analytic reward reports about the
                      policy minus what the policy actually costs on the realised clock

PART B -- GRADIENT LEVEL, sampling decode (the training estimator), `--grad-instances`
instances, `--grad-days` days. Day d uses noise seed 20000+d AND action-sampling seed
30000+d, so within a day the hybrid and sampled estimators see the same trajectories:
the hybrid rollout's actions are replayed in a sampled env, and both POMO losses are
backpropagated through the SAME log-probabilities. The analytic estimator uses the same
sampling seed on its own (mean-clock) rollout.
  (B1) total variance trace Cov_days(g^H) / trace Cov_days(g^S), bootstrap CI over days
  (B2) direction      cos(mean g^A, mean g^S) and cos(mean g^H, mean g^S), with the
                      split-half cos(mean g^S over days 1..D/2, over days D/2+1..D) as
                      the ceiling a noisy mean can reach

READING (fixed before running; Section 6.1 of the paper):
  R2a  if the CI of (A1) excludes 0 at any beta: implementation error, stop and fix.
  R2b  claim "the hybrid reward reduces the variance of the cost on policy routes" only if
       the CI of the mean log ratio (A2) lies below 0 at EVERY beta; otherwise report the
       ratio without the claim.
  R2c  same rule for the gradient: the bootstrap CI of (B1) below 1 at every beta.

Outputs: <out>.log (console copy) and <out>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from scripts.bn_mechanism import _tee_console
from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig, draw_noise
from src.eval.checkpoints import load_stoch_policy
from src.train.pomo import build_starts
from src.train.pomo_stoch import build_stoch_env

DAY_SEED_OFFSET = 20_000       # E2 days: never a test day (0-3) or a selection day (10000+)
SAMPLING_SEED_OFFSET = 30_000  # action-sampling seed of gradient day d


# ------------------------------------------------------------------------------ args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2: reward properties on policy routes")
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="trained checkpoints (_last), one per seed")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--test-size", type=int, default=64,
                   help="instances for part A: the FIRST n of the test set (seed 4242)")
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--days", type=int, default=256, help="days per instance, part A")
    p.add_argument("--grad-instances", type=int, default=8, help="instances, part B")
    p.add_argument("--grad-days", type=int, default=64, help="days, part B")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--skip-grad", action="store_true", help="run part A only")
    p.add_argument("--out", type=str, default="runs/e2_reward_properties")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- helpers
def _day_noise(data: CVRPTWData, cfg: SimulatorConfig, day: int) -> Tensor:
    """The realised day `day` for every instance of `data`. (B, N, N, K)."""
    b, n, _ = data.coords.shape
    dev = data.coords.device
    gen = torch.Generator(device=dev).manual_seed(DAY_SEED_OFFSET + day)
    return draw_noise((b, n, n, cfg.n_bins), device=dev, dtype=data.coords.dtype,
                      generator=gen)


def _replay(data: CVRPTWData, cfg: SimulatorConfig, weight: float, n_starts: int,
            noise: Tensor | None, mode: str, actions: Tensor) -> dict[str, Tensor]:
    """Drive the given actions (B*S, T) in a fresh env; return its cost components."""
    b = data.coords.size(0)
    env = build_stoch_env(data, cfg, n_starts, weight, noise=noise, reward_mode=mode)
    env.reset(build_starts(b, n_starts, data.coords.device))
    for t in range(actions.size(1)):
        env.step(actions[:, t])
    return env.cost_components(actions)                              # (B*S,) each


def _rollout(policy, data: CVRPTWData, cfg: SimulatorConfig, weight: float,
             n_starts: int, noise: Tensor | None, mode: str, decode: str
             ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """One policy rollout. Returns actions (B*S, T), sum_logp (B*S,), cost components."""
    b = data.coords.size(0)
    env = build_stoch_env(data, cfg, n_starts, weight, noise=noise, reward_mode=mode)
    actions, sum_logp, cost = policy(env, decode_type=decode,
                                     force_first=build_starts(b, n_starts,
                                                              data.coords.device))
    parts = env.cost_components(actions)
    if not torch.equal(cost, parts["total"]):                        # crash-not-silent
        raise RuntimeError("policy cost != env.cost_components()['total']")
    return actions, sum_logp, parts


def _ci(per_instance: Tensor) -> dict[str, float]:
    """Mean and 95% normal-approximation CI over instances. per_instance: (I,)."""
    x = per_instance.double()
    n = x.numel()
    mean = float(x.mean())
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": n}


def _fmt(c: dict[str, float], digits: int = 4) -> str:
    return f"{c['mean']:+.{digits}f} [{c['lo']:+.{digits}f}, {c['hi']:+.{digits}f}]"


# ------------------------------------------------------------------------- part A
@torch.no_grad()
def part_a(policy, data: CVRPTWData, cfg: SimulatorConfig, weight: float, days: int
           ) -> dict:
    """Cost-level statistics on greedy routes. All per-day tensors are (I, S, D)."""
    i_n, n, _ = data.coords.shape
    s_n = n - 1
    c_h = torch.empty(i_n, s_n, days, dtype=torch.float64)
    c_s = torch.empty_like(c_h)
    for d in range(days):
        noise = _day_noise(data, cfg, d)
        actions, _, ph = _rollout(policy, data, cfg, weight, s_n, noise, "hybrid", "greedy")
        ps = _replay(data, cfg, weight, s_n, noise, "sampled", actions)
        if d == 0:
            # Replaying the policy's own actions in a hybrid env must give its own cost:
            # proves the replay drives the same routes on the same day.
            pr = _replay(data, cfg, weight, s_n, noise, "hybrid", actions)
            if not torch.allclose(pr["total"], ph["total"], atol=1e-5):
                raise RuntimeError("replay does not reproduce the hybrid rollout")
            if not torch.allclose(ps["distance"], ph["distance"], atol=1e-5):
                raise RuntimeError("sampled replay drove a different route")
        c_h[:, :, d] = ph["total"].double().view(i_n, s_n).cpu()
        c_s[:, :, d] = ps["total"].double().view(i_n, s_n).cpu()

    _, _, pa = _rollout(policy, data, cfg, weight, s_n, None, "analytic", "greedy")
    c_a = pa["total"].double().view(i_n, s_n).cpu()                  # (I, S)
    late_a = pa["late_count"].double().view(i_n, s_n).cpu()
    tard_a = pa["tardiness"].double().view(i_n, s_n).cpu()
    dist_a = pa["distance"].double().view(i_n, s_n).cpu()

    # A1: unbiasedness, per instance (mean over starts and days).
    a1 = _ci((c_h - c_s).mean(dim=(1, 2)))

    # A2: variance ratio per (instance, start); starts with no sampled variance (never late
    # on any day) carry no information about noise and are excluded and counted.
    var_h = c_h.var(dim=2, unbiased=True)                            # (I, S)
    var_s = c_s.var(dim=2, unbiased=True)
    ok = var_s > 1e-10
    log_ratio = torch.where(ok, (var_h.clamp_min(1e-12) / var_s.clamp_min(1e-12)).log(),
                            torch.zeros_like(var_s))
    per_inst = (log_ratio * ok).sum(dim=1) / ok.sum(dim=1).clamp_min(1)  # (I,)
    has = ok.any(dim=1)
    a2 = _ci(per_inst[has])
    pooled = float(var_h[ok].sum() / var_s[ok].sum())

    # A3: the POMO advantage C_s - mean_s C_s, per day.
    adv_h = c_h - c_h.mean(dim=1, keepdim=True)
    adv_s = c_s - c_s.mean(dim=1, keepdim=True)
    av_h = adv_h.var(dim=2, unbiased=True)
    av_s = adv_s.var(dim=2, unbiased=True)
    ok3 = av_s > 1e-10
    lr3 = torch.where(ok3, (av_h.clamp_min(1e-12) / av_s.clamp_min(1e-12)).log(),
                      torch.zeros_like(av_s))
    a3 = _ci(((lr3 * ok3).sum(dim=1) / ok3.sum(dim=1).clamp_min(1))[ok3.any(dim=1)])
    pooled3 = float(av_h[ok3].sum() / av_s[ok3].sum())

    # A4: analytic bias against the policy's true expected cost on the realised clock.
    true_cost = c_s.mean(dim=2)                                      # (I, S)
    a4 = _ci((c_a - true_cost).mean(dim=1))
    # Its parts are reported in means only (the sampled parts are not stored per day).
    return {
        "A1_hybrid_minus_sampled": a1,
        "A2_log_var_ratio": a2, "A2_ratio_geomean": math.exp(a2["mean"]),
        "A2_ratio_pooled": pooled, "A2_starts_without_variance": int((~ok).sum()),
        "A3_log_adv_var_ratio": a3, "A3_ratio_geomean": math.exp(a3["mean"]),
        "A3_ratio_pooled": pooled3,
        "A4_analytic_minus_true": a4,
        "true_cost_mean": float(true_cost.mean()),
        "analytic_cost_mean": float(c_a.mean()),
        "analytic_parts_mean": {"distance": float(dist_a.mean()),
                                "late_count": float(late_a.mean()),
                                "tardiness": float(tard_a.mean())},
    }


# ------------------------------------------------------------------------- part B
def _pomo_loss(cost: Tensor, sum_logp: Tensor, b: int, s: int) -> Tensor:
    """POMO shared-baseline loss. cost, sum_logp: (B*S,) -> scalar."""
    c = cost.view(b, s)
    adv = (c - c.mean(dim=1, keepdim=True)).detach()                 # (B, S)
    return (adv * sum_logp.view(b, s)).mean()


def _flat_grad(loss: Tensor, params: list[torch.nn.Parameter], retain: bool) -> Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)]).detach().float().cpu()


def part_b(policy, data: CVRPTWData, cfg: SimulatorConfig, weight: float, days: int,
           n_boot: int) -> dict:
    """Gradient-level statistics with sampling decode. Policy stays in eval mode (BN
    frozen) so the three estimators differ only through the reward."""
    b, n, _ = data.coords.shape
    s = n - 1
    params = [p for p in policy.parameters() if p.requires_grad]
    g_h, g_s, g_a = [], [], []
    for d in range(days):
        noise = _day_noise(data, cfg, d)
        torch.manual_seed(SAMPLING_SEED_OFFSET + d)
        actions, logp, ph = _rollout(policy, data, cfg, weight, s, noise, "hybrid", "sampling")
        with torch.no_grad():
            ps = _replay(data, cfg, weight, s, noise, "sampled", actions)
        g_h.append(_flat_grad(_pomo_loss(ph["total"], logp, b, s), params, retain=True))
        g_s.append(_flat_grad(_pomo_loss(ps["total"], logp, b, s), params, retain=False))
        torch.manual_seed(SAMPLING_SEED_OFFSET + d)
        _, logp_a, pa = _rollout(policy, data, cfg, weight, s, None, "analytic", "sampling")
        g_a.append(_flat_grad(_pomo_loss(pa["total"], logp_a, b, s), params, retain=False))
    gh, gs, ga = torch.stack(g_h), torch.stack(g_s), torch.stack(g_a)  # (D, P)

    def trace(g: Tensor) -> Tensor:
        return g.var(dim=0, unbiased=True).sum()

    ratio = float(trace(gh) / trace(gs))
    gen = torch.Generator().manual_seed(0)
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(0, days, (days,), generator=gen)
        boots.append(float(trace(gh[idx]) / trace(gs[idx])))
    boots_t = torch.tensor(boots)
    lo, hi = float(boots_t.quantile(0.025)), float(boots_t.quantile(0.975))

    def cos(u: Tensor, v: Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(u, v, dim=0))

    half = days // 2
    return {
        "B1_trace_ratio": {"mean": ratio, "lo": lo, "hi": hi, "n_days": days},
        "B1_trace_sampled": float(trace(gs)), "B1_trace_hybrid": float(trace(gh)),
        "B1_trace_analytic": float(trace(ga)),
        "B2_cos_analytic_sampled": cos(ga.mean(0), gs.mean(0)),
        "B2_cos_hybrid_sampled": cos(gh.mean(0), gs.mean(0)),
        "B2_cos_sampled_split_half": cos(gs[:half].mean(0), gs[half:].mean(0)),
        "B2_cos_hybrid_split_half": cos(gh[:half].mean(0), gh[half:].mean(0)),
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)

    loaded = [load_stoch_policy(p, device) for p in args.ckpt]
    ref = loaded[0].blob
    weight = float(ref["tardiness_weight"])
    for lp in loaded:
        if float(lp.blob["tardiness_weight"]) != weight:
            raise ValueError("checkpoints disagree on tardiness_weight")
    full = generate_cvrptw(
        max(args.test_size, args.grad_instances), ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )

    def head(k: int) -> CVRPTWData:
        return full._replace(coords=full.coords[:k], demand=full.demand[:k],
                             time_windows=full.time_windows[:k],
                             service_time=full.service_time[:k])

    data_a, data_b = head(args.test_size), head(args.grad_instances)
    print(f"{len(loaded)} checkpoints ({', '.join(lp.variant for lp in loaded)}) | "
          f"part A: {args.test_size} instances x {args.days} days (greedy) | part B: "
          f"{args.grad_instances} instances x {args.grad_days} days (sampling) | "
          f"tardiness_weight {weight}\n")

    results: dict = {"args": vars(args), "runs": []}
    t0 = time.time()
    for path, lp in zip(args.ckpt, loaded):
        for beta in args.beta_var:
            cfg = replace(lp.sim_cfg, beta_var=beta)
            ra = part_a(lp.policy, data_a, cfg, weight, args.days)
            line = (f"{Path(path).name} beta {beta:g} | A1 H-S {_fmt(ra['A1_hybrid_minus_sampled'])}"
                    f" | A2 var ratio {ra['A2_ratio_geomean']:.3f} (pooled "
                    f"{ra['A2_ratio_pooled']:.3f}) log {_fmt(ra['A2_log_var_ratio'], 3)}"
                    f" | A3 adv ratio {ra['A3_ratio_geomean']:.3f} (pooled "
                    f"{ra['A3_ratio_pooled']:.3f})"
                    f" | A4 analytic bias {_fmt(ra['A4_analytic_minus_true'])}")
            rb = None
            if not args.skip_grad:
                rb = part_b(lp.policy, data_b, cfg, weight, args.grad_days, args.bootstrap)
                r = rb["B1_trace_ratio"]
                line += (f" | B1 grad var ratio {r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
                         f" | cos(A,S) {rb['B2_cos_analytic_sampled']:+.3f} cos(H,S) "
                         f"{rb['B2_cos_hybrid_sampled']:+.3f} split-half S "
                         f"{rb['B2_cos_sampled_split_half']:+.3f}")
            print(line + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
            results["runs"].append({"ckpt": path, "beta": beta, "A": ra, "B": rb})

    print("\nREADING (fixed before running)")
    for beta in args.beta_var:
        rows = [r for r in results["runs"] if r["beta"] == beta]
        a1_bad = [r["ckpt"] for r in rows
                  if r["A"]["A1_hybrid_minus_sampled"]["lo"] > 0
                  or r["A"]["A1_hybrid_minus_sampled"]["hi"] < 0]
        a2_all = all(r["A"]["A2_log_var_ratio"]["hi"] < 0 for r in rows)
        b1_all = (None if args.skip_grad
                  else all(r["B"]["B1_trace_ratio"]["hi"] < 1 for r in rows))
        print(f"  beta {beta:g}: R2a {'FAIL ' + str(a1_bad) if a1_bad else 'ok (CI covers 0)'}"
              f" | R2b cost variance reduced on every seed: {a2_all}"
              f" | R2c gradient variance reduced on every seed: {b1_all}")

    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  wrote {args.out}.json")


if __name__ == "__main__":
    main()
