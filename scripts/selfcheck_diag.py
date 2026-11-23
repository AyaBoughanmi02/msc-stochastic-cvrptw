"""Diagnose a failed self-check of scripts/selection_study.py (read-only: writes nothing).

Re-runs exactly the self-check of `selection_study` (same flags, same instances, same days)
and prints, for every (rule, component), the largest absolute and relative difference
between `evaluate_stoch_apriori` and `selection_protocols`, with the instance where it occurs.

Decision rule (fixed 10 Nov 2026, before running this script):
  ROUNDOFF  if every max relative difference <= 1e-5 AND every max absolute difference < 1e-3
            (a different chosen candidate moves an instance cost by >= 1e-2 at these scales)
  MISMATCH  otherwise -> stop the n = 50 study and debug.

Usage: the same flags as the failed run, e.g.
    python -u -m scripts.selfcheck_diag --ckpt checkpoints\\M0_w1_15k_s2024_last.pt
        --n-customers 50 --capacity 20 --beta-var 0.6 --test-size 128 --scenarios 16
        --select-days 64 --self-check 16 --eval-batch-size 1 --no-log-file --out runs\\diag
"""
from __future__ import annotations

from dataclasses import replace

import torch

from scripts.selection_study import _SELF_CHECK_RULES, _ctx, parse_args
from src.data.cvrptw_generator import generate_cvrptw
from src.eval.apriori_selection import _slice, evaluate_stoch_apriori
from src.eval.checkpoints import load_stoch_policy
from src.eval.selection_protocols import apply_rules, collect_candidate_costs

_COMPONENTS = ("cost", "distance", "tardiness", "late_count")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaded = load_stoch_policy(args.ckpt[0], device)
    ref_blob = loaded.blob
    weight = float(ref_blob["tardiness_weight"])
    n_cust = ref_blob["n_customers"] if args.n_customers is None else args.n_customers
    cap = ref_blob["capacity"] if args.capacity is None else args.capacity
    # Same call as selection_study.main, so the first `self_check` instances are identical.
    data = generate_cvrptw(
        args.test_size, n_cust, cap,
        service_time=ref_blob["service_time"], shift_end=ref_blob["shift_end"],
        tw_half_width=ref_blob["tw_half_width"], seed=args.test_seed, device=device,
    )
    n = min(args.self_check, data.coords.shape[0])
    sub = _slice(data, slice(0, n))
    beta = args.beta_var[0]
    cfg = replace(loaded.sim_cfg, beta_var=beta)
    kw = dict(tardiness_weight=weight, augment=not args.no_augment,
              eval_batch_size=args.eval_batch_size, scenario_seeds=tuple(range(args.scenarios)))
    with _ctx(loaded.policy, args):
        ref = evaluate_stoch_apriori(loaded.policy, sub, cfg, n_selection_days=8, **kw)
        cc = collect_candidate_costs(loaded.policy, sub, cfg, n_selection_days=8, **kw)
    new = apply_rules(cc, ks=(8,), ns=(), n_subsets=0)

    print(f"n_customers {n_cust} | {n} instances | beta {beta:g}")
    print(f"{'rule':>18} {'component':>11} {'max|abs|':>10} {'max rel':>10} {'inst':>5} "
          f"{'old':>12} {'new':>12}")
    worst_abs, worst_rel = 0.0, 0.0
    pairs = [(r, r) for r in _SELF_CHECK_RULES] + [("saa", "saa_out_K8")]
    for old_r, new_r in pairs:
        for k in _COMPONENTS:
            a = ref[old_r][k].double().cpu()                  # (n,)
            b = new[new_r][k].double().cpu()                  # (n,)
            diff = (a - b).abs()                              # (n,)
            rel = diff / a.abs().clamp_min(1e-12)             # (n,); 0/0 guarded
            rel = torch.where(diff == 0, torch.zeros_like(rel), rel)
            i = int(diff.argmax())
            worst_abs = max(worst_abs, float(diff.max()))
            worst_rel = max(worst_rel, float(rel.max()))
            print(f"{new_r:>18} {k:>11} {float(diff.max()):10.2e} {float(rel.max()):10.2e} "
                  f"{i:5d} {float(a[i]):12.6f} {float(b[i]):12.6f}")
    verdict = "ROUNDOFF" if (worst_rel <= 1e-5 and worst_abs < 1e-3) else "MISMATCH"
    print(f"\nworst abs {worst_abs:.2e} | worst rel {worst_rel:.2e} -> {verdict}")


if __name__ == "__main__":
    main()