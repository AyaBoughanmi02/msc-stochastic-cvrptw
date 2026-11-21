"""E3: does each training reward rank the POMO starts of an instance like their true cost?

Example (PowerShell, one line):
    python -u -m scripts.e3_ranking_bias
        --ckpt checkpoints\\M0_ftH_s2024_last.pt checkpoints\\M0_ftH_s7_last.pt
               checkpoints\\M0_ftH_s1234_last.pt
        --out runs\\e3_ranking_bias

Why. E2 showed that the analytic reward overstates the policy's cost by 11-34%. A bias
that shifts every start of an instance by the same amount cancels in the POMO advantage
C_s - mean_s C_s, so E2 alone does not show that the bias misleads training. What the
gradient uses is the RANKING of the starts. This script measures that ranking directly.

Inference only. For every checkpoint (one per seed) and every beta, on the first
`--test-size` test instances (seed 4242) and `--days` days per instance (seeds 40000+d:
disjoint from the test days 0-3, the selection days 10000+ and the E2 days 20000+):
  * each day, the policy drives its greedy routes from all starts on the REALISED clock
    (hybrid env); the same actions are replayed on the same day with the sampled charge;
  * the analytic reward drives the policy once on the MEAN clock (no noise): C^A.
Days are split in two halves:
  * TRUTH = mean over the second half of the sampled cost, per (instance, start): the
    policy's expected cost from that start, estimated on days no estimator below sees;
  * ESTIMATORS on the first half: analytic (one value), and the sampled and hybrid
    costs averaged over blocks of K consecutive days, K in --k-days.
Per instance, for each estimator x: (s = start)
  corr    Pearson correlation over starts between x_s - mean_s x and the true advantage
          (averaged over the blocks of K days); 1 = ranks the starts exactly like the truth
  regret  TRUTH[argmin_s x] - min_s TRUTH: the true extra cost of the start the reward
          calls best (averaged over blocks)
The truth is itself estimated from 128 days, which lowers every correlation somewhat; all
estimators are scored against the SAME truth, paired per instance, so their differences
remain meaningful. corr(H, K=128) shows the highest value this truth allows.

READING (fixed before running; would go into Section 6.1 of the paper):
  R3a  claim "one simulated day ranks the starts better than the noiseless analytic
       reward" only if the CI of corr(hybrid, K=1) - corr(analytic) lies above 0 for
       EVERY seed at EVERY beta. If it lies below 0 everywhere, report instead that the
       analytic reward ranks better than one day, so its cost to training is the clock it
       trains on, not its ranking. Otherwise report the numbers without a claim.
  R3b  claim "the lower variance of the hybrid reward ranks the starts better" only if
       the CI of corr(hybrid, K=1) - corr(sampled, K=1) lies above 0 for every seed at
       every beta; otherwise report without the claim.

Outputs: <out>.log (console copy) and <out>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from scripts.bn_mechanism import _tee_console
from scripts.e2_reward_properties import _ci, _fmt, _replay, _rollout
from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SimulatorConfig, draw_noise
from src.eval.checkpoints import load_stoch_policy

DAY_SEED_OFFSET = 40_000   # E3 days: disjoint from test (0-3), selection (10000+), E2 (20000+)


# ------------------------------------------------------------------------------ args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3: ranking bias of the training rewards")
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="trained checkpoints (_last), one per seed")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--test-size", type=int, default=64,
                   help="the FIRST n instances of the test set")
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--days", type=int, default=256,
                   help="days per instance; first half = estimators, second half = truth")
    p.add_argument("--k-days", type=int, nargs="+", default=[1, 4, 16, 128],
                   help="block sizes K for the day-averaged sampled / hybrid estimators")
    p.add_argument("--out", type=str, default="runs/e3_ranking_bias")
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


def _centred_corr(x: Tensor, y: Tensor) -> Tensor:
    """Pearson correlation over the LAST axis. x, y: (..., S) -> (...,); NaN if flat."""
    xc = x - x.mean(dim=-1, keepdim=True)
    yc = y - y.mean(dim=-1, keepdim=True)
    den = (xc.pow(2).sum(-1) * yc.pow(2).sum(-1)).sqrt()
    num = (xc * yc).sum(-1)
    return torch.where(den > 0, num / den.clamp_min(1e-300),
                       torch.full_like(num, float("nan")))


def _regret(x: Tensor, truth: Tensor) -> Tensor:
    """True extra cost of the start x calls best. x: (I, S, nb), truth: (I, S) -> (I, nb)."""
    pick = x.argmin(dim=1)                                           # (I, nb)
    picked = truth.gather(1, pick)                                   # (I, nb)
    return picked - truth.min(dim=1, keepdim=True).values


def _blocks(c: Tensor, k: int) -> Tensor:
    """Average consecutive blocks of k days. c: (I, S, D) -> (I, S, D // k)."""
    i_n, s_n, d = c.shape
    nb = d // k
    if nb < 1:
        raise ValueError(f"block size {k} exceeds the {d} estimation days")
    return c[:, :, : nb * k].reshape(i_n, s_n, nb, k).mean(dim=3)


def _per_instance(metric: Tensor) -> Tensor:
    """Mean over blocks, ignoring NaN. (I, nb) -> (I,)."""
    return torch.nanmean(metric, dim=1)


def _paired(a: Tensor, b: Tensor) -> dict[str, float]:
    """CI of a - b over instances where both are defined. a, b: (I,)."""
    ok = ~(a.isnan() | b.isnan())
    if int(ok.sum()) < 2:
        raise RuntimeError("fewer than 2 instances with a defined correlation")
    return _ci(a[ok] - b[ok])


# ----------------------------------------------------------------------------- core
@torch.no_grad()
def one_run(policy, data: CVRPTWData, cfg: SimulatorConfig, weight: float, days: int,
            k_days: list[int]) -> dict:
    """All E3 statistics for one (checkpoint, beta)."""
    i_n, n, _ = data.coords.shape
    s_n = n - 1
    if days < 4 or days % 2:
        raise ValueError("--days must be even and >= 4")
    c_h = torch.empty(i_n, s_n, days, dtype=torch.float64)
    c_s = torch.empty_like(c_h)
    for d in range(days):
        noise = _day_noise(data, cfg, d)
        actions, _, ph = _rollout(policy, data, cfg, weight, s_n, noise, "hybrid", "greedy")
        ps = _replay(data, cfg, weight, s_n, noise, "sampled", actions)
        if d == 0 and not torch.allclose(ps["distance"], ph["distance"], atol=1e-5):
            raise RuntimeError("sampled replay drove a different route")
        c_h[:, :, d] = ph["total"].double().view(i_n, s_n).cpu()
        c_s[:, :, d] = ps["total"].double().view(i_n, s_n).cpu()
    _, _, pa = _rollout(policy, data, cfg, weight, s_n, None, "analytic", "greedy")
    c_a = pa["total"].double().view(i_n, s_n).cpu()                  # (I, S)

    half = days // 2
    truth = c_s[:, :, half:].mean(dim=2)                             # (I, S)
    est_h, est_s = c_h[:, :, :half], c_s[:, :, :half]                # (I, S, half)

    corr: dict[str, Tensor] = {}
    regret: dict[str, Tensor] = {}
    corr["A"] = _centred_corr(c_a, truth)                            # (I,)
    regret["A"] = _regret(c_a.unsqueeze(2), truth).squeeze(1)        # (I,)
    for k in k_days:
        if k > half:
            continue
        for name, est in (("H", est_h), ("S", est_s)):
            x = _blocks(est, k)                                      # (I, S, nb)
            c = _centred_corr(x.transpose(1, 2), truth.unsqueeze(1))  # (I, nb)
            corr[f"{name}{k}"] = _per_instance(c)
            regret[f"{name}{k}"] = _regret(x, truth).mean(dim=1)

    return {
        "truth_cost_mean": float(truth.mean()),
        "true_adv_sd_mean": float(truth.std(dim=1, unbiased=True).mean()),
        "corr": {k: _ci(v[~v.isnan()]) for k, v in corr.items()},
        "regret": {k: _ci(v) for k, v in regret.items()},
        "corr_undefined": {k: int(v.isnan().sum()) for k, v in corr.items()},
        "R3a_H1_minus_A": _paired(corr["H1"], corr["A"]),
        "R3b_H1_minus_S1": _paired(corr["H1"], corr["S1"]),
        "regret_A_minus_H1": _ci(regret["A"] - regret["H1"]),
        "regret_S1_minus_H1": _ci(regret["S1"] - regret["H1"]),
    }


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    _tee_console(Path(f"{args.out}.log"))
    print("command:", " ".join(sys.argv))
    device = torch.device(args.device)
    if 1 not in args.k_days:
        raise ValueError("--k-days must include 1 (rules R3a and R3b use it)")

    loaded = [load_stoch_policy(p, device) for p in args.ckpt]
    ref = loaded[0].blob
    weight = float(ref["tardiness_weight"])
    for lp in loaded:
        if float(lp.blob["tardiness_weight"]) != weight:
            raise ValueError("checkpoints disagree on tardiness_weight")
    data = generate_cvrptw(
        args.test_size, ref["n_customers"], ref["capacity"],
        service_time=ref["service_time"], shift_end=ref["shift_end"],
        tw_half_width=ref["tw_half_width"], seed=args.test_seed, device=device,
    )
    half = args.days // 2
    print(f"{len(loaded)} checkpoints | {args.test_size} instances x {args.days} days "
          f"({half} for the estimators, {half} for the truth) | K in "
          f"{[k for k in args.k_days if k <= half]} | tardiness_weight {weight}\n")

    results: dict = {"args": vars(args), "runs": []}
    t0 = time.time()
    for path, lp in zip(args.ckpt, loaded):
        for beta in args.beta_var:
            cfg = replace(lp.sim_cfg, beta_var=beta)
            r = one_run(lp.policy, data, cfg, weight, args.days, args.k_days)
            cm = " ".join(f"{k} {v['mean']:+.3f}" for k, v in r["corr"].items())
            rg = " ".join(f"{k} {v['mean']:.3f}" for k, v in r["regret"].items())
            print(f"{Path(path).name} beta {beta:g} | true cost {r['truth_cost_mean']:.3f} "
                  f"(sd over starts {r['true_adv_sd_mean']:.3f})\n"
                  f"    corr   {cm}\n"
                  f"    regret {rg}\n"
                  f"    corr H1-A {_fmt(r['R3a_H1_minus_A'], 3)} | corr H1-S1 "
                  f"{_fmt(r['R3b_H1_minus_S1'], 3)} | regret A-H1 "
                  f"{_fmt(r['regret_A_minus_H1'], 3)} | regret S1-H1 "
                  f"{_fmt(r['regret_S1_minus_H1'], 3)} | {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            results["runs"].append({"ckpt": path, "beta": beta, **r})

    print("\nREADING (fixed before running)")
    for beta in args.beta_var:
        rows = [r for r in results["runs"] if r["beta"] == beta]
        a_up = all(r["R3a_H1_minus_A"]["lo"] > 0 for r in rows)
        a_dn = all(r["R3a_H1_minus_A"]["hi"] < 0 for r in rows)
        b_up = all(r["R3b_H1_minus_S1"]["lo"] > 0 for r in rows)
        r3a = ("one hybrid day ranks better than analytic on every seed" if a_up else
               "analytic ranks better than one day on every seed" if a_dn else
               "mixed: no claim")
        print(f"  beta {beta:g}: R3a {r3a} | R3b hybrid ranks better than sampled on "
              f"every seed: {b_up}")

    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  wrote {args.out}.json")


if __name__ == "__main__":
    main()
