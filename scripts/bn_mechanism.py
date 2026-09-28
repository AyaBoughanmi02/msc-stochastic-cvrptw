"""Stage-5 mechanism study: WHY does the raw sigma^2 feature hurt variant B?

Example (PowerShell, from the repo root; one line):
    python -u -m scripts.bn_mechanism --baseline checkpoints\\B_w1_15k_s2024_last.pt
        --control checkpoints\\Dp_w1_15k_s2024_last.pt --out runs\\bn_mech_w1

Everything is inference-only: nothing retrains, nothing writes to a checkpoint.

THE MECHANISM UNDER TEST (see src/eval/bn_interventions.py for the full argument).
train_stoch draws ONE beta_var per training mini-batch, and bn_edge in train mode
normalises each batch with that batch's own statistics. At evaluation bn_edge uses running
statistics instead. So B was trained on edge embeddings from which part of the day's
variance level was normalised away, and is evaluated on embeddings where it was not.

THREE MEASUREMENTS

  1. DRIFT (no rollout, seconds). For each beta_var on a grid: how far eval-mode bn_edge
     output is from what train-mode BN would have produced on a batch at that beta. M is
     the placebo: its edge feature (mu) does not depend on beta, so its drift cannot
     either. Also reports which beta the running statistics effectively encode.

  2. COUNTERFACTUAL EVALUATION, paired, same instances and same days as the protocol:
       plain             B as trained and evaluated (the published comparison)
       bn_batch          bn_edge normalised with batch statistics AT THE EVALUATED beta,
                         i.e. what training did. At the ID beta this removes ONLY the
                         train/eval discrepancy; out of distribution it ALSO removes the
                         encoder-side scale extrapolation (a 0.9 batch is normalised back
                         into the trained range). Both are SCALE problems.
       dec_ref           the pointer's sigma^2 input shown at the training-midpoint scale;
                         the simulated day is unchanged. Isolates linear extrapolation
                         through w_edge (the decoder has no norm layer).
       bn_batch+dec_ref  both. What remains of B's gap to M after this is NOT a scale
                         effect: it is the information / optimisation residual (a).
     M is evaluated plain and with bn_batch (placebo: must change ~nothing).

  3. VERDICT against thresholds fixed BEFORE the numbers were seen (constants below).

RESUMABLE. Every finished evaluation is cached in <out>_cache.pt; rerun with --resume
after a crash and only the missing ones are computed. Outputs: <out>.log (full console
copy, UTF-8), <out>_drift.csv, <out>_drift.png (if matplotlib is installed),
<out>_summary.json, <out>_cache.pt (per-instance results, for any later analysis).

Run from the repo root so `src` and `scripts` are importable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

import torch
from torch import Tensor

from scripts.compare_variants import forced_gate, paired_difference
from src.data.cvrptw_generator import CVRPTWData, generate_cvrptw
from src.data.simulator import SIGMA2
from src.eval.bn_interventions import (
    batch_stat_edge_norm,
    edge_norm_stats,
    normalisation_drift,
    scaled_decoder_variance,
)
from src.eval.budget import add_budget_argument, assert_matched_budget
from src.eval.checkpoints import LoadedPolicy, load_stoch_policy
from src.eval.evaluate_stoch import evaluate_stoch

B_CONDITIONS = ("plain", "bn_batch", "dec_ref", "bn_batch+dec_ref")
M_CONDITIONS = ("plain", "bn_batch")
_KEYS = ("cost", "distance", "tardiness", "late_count")

# PRE-REGISTERED decision thresholds (fixed 28 Sept 2026, before any result was seen).
# share = fraction of B's excess degradation over M that an intervention removes.
MAIN_SHARE = 0.5    # >= : the scale mechanism is the MAIN explanation
MINOR_SHARE = 0.2   # <  : the scale mechanism is NOT the explanation


# --------------------------------------------------------------------------- logging
class _Tee:
    """Duplicate a text stream into a UTF-8 file, line-buffered, so no output is lost."""

    def __init__(self, stream: TextIO, file: TextIO) -> None:
        self._stream, self._file = stream, file

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return False


def _tee_console(path: Path) -> None:
    """Send stdout AND stderr (tracebacks included) to `path` as well as the console."""
    log = open(path, "a", encoding="utf-8", buffering=1)      # noqa: SIM115 (process-lifetime)
    log.write(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)


# ------------------------------------------------------------------------------ args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-5 mechanism study (inference only)")
    p.add_argument("--baseline", type=str, nargs="+", required=True,
                   help="variant-B checkpoints (_last), one per seed")
    p.add_argument("--control", type=str, nargs="+", required=True,
                   help="mean-only checkpoints (D' _last, evaluated at lambda = 0)")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9],
                   help="evaluation levels; the FIRST is the in-distribution reference")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--n-starts", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--calib-seed", type=int, default=777,
                   help="held-out instances the batch statistics are computed on")
    p.add_argument("--calib-size", type=int, default=256)
    p.add_argument("--drift-grid", type=float, nargs="+",
                   default=[round(0.10 + 0.05 * i, 2) for i in range(19)],  # 0.10 .. 1.00
                   help="beta_var values for the drift curve")
    p.add_argument("--drift-only", action="store_true",
                   help="stop after the drift section (seconds, no rollouts)")
    p.add_argument("--out", type=str, default="runs/bn_mechanism",
                   help="output PREFIX; files <out>.log, <out>_drift.csv, ...")
    p.add_argument("--resume", action="store_true",
                   help="reuse finished evaluations from <out>_cache.pt")
    add_budget_argument(p)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ------------------------------------------------------------------------ validation
def _assert_one_problem(paths: list[str], loaded: list[LoadedPolicy]) -> None:
    """Same reward, same instance dials, same simulator except beta_var. Never mix omega."""
    ref, ref_path = loaded[0], paths[0]
    ref_sim = {k: v for k, v in asdict(ref.sim_cfg).items() if k != "beta_var"}
    for path, l in zip(paths, loaded):
        if float(l.blob["tardiness_weight"]) != float(ref.blob["tardiness_weight"]):
            raise ValueError(f"{path} and {ref_path} disagree on tardiness_weight")
        for key in ("n_customers", "capacity", "shift_end", "tw_half_width",
                    "service_time"):
            if l.blob[key] != ref.blob[key]:
                raise ValueError(f"{path} and {ref_path} disagree on '{key}'")
        sim = {k: v for k, v in asdict(l.sim_cfg).items() if k != "beta_var"}
        if sim != ref_sim:
            diff = {k: (sim[k], ref_sim[k]) for k in sim if sim[k] != ref_sim[k]}
            raise ValueError(
                f"{path} was trained in a different environment than {ref_path}: {diff}. "
                f"Different omega/eta/... is a different problem; never compare across it"
            )
        if l.blob.get("beta_var_range") != ref.blob.get("beta_var_range"):
            raise ValueError(f"{path} and {ref_path} disagree on beta_var_range")


def _beta_ref(loaded: LoadedPolicy) -> float:
    """Midpoint of the trained beta range (or the fixed trained beta)."""
    rng = loaded.blob.get("beta_var_range")
    if rng is None:
        return float(loaded.sim_cfg.beta_var)
    return 0.5 * (float(rng[0]) + float(rng[1]))


def _generate(loaded: LoadedPolicy, size: int, seed: int, device: torch.device) -> CVRPTWData:
    b = loaded.blob
    return generate_cvrptw(
        size, b["n_customers"], b["capacity"], service_time=b["service_time"],
        shift_end=b["shift_end"], tw_half_width=b["tw_half_width"], seed=seed,
        device=device,
    )


# ----------------------------------------------------------------------------- drift
def run_drift(
    args: argparse.Namespace,
    named: list[tuple[str, str, LoadedPolicy]],
    calib: CVRPTWData,
    out: str,
) -> dict[str, float]:
    """Drift curve for every checkpoint; returns the effective beta of each B checkpoint."""
    trained = named[0][2].blob.get("beta_var_range")
    print("\n1. NORMALISATION DRIFT  (eval-mode bn_edge vs what training-mode BN produced "
          "on a batch at the same beta)")
    print("   drift = 0: evaluation sees exactly what training saw.  M is the placebo.")
    rows: list[dict] = []
    effective: dict[str, float] = {}
    for role, path, loaded in named:
        print(f"\n   {role}  {path}")
        print(f"   {'beta':>6} {'drift':>8} {'loc':>8} {'scale':>8}")
        best_beta, best_loc = float("nan"), float("inf")
        for beta in args.drift_grid:
            d = normalisation_drift(loaded.policy, calib, replace(loaded.sim_cfg, beta_var=beta))
            inside = trained is None or trained[0] - 1e-9 <= beta <= trained[1] + 1e-9
            rows.append({"role": role, "checkpoint": path, "beta_var": beta,
                         "in_trained_range": inside, **d})
            print(f"   {beta:>6.2f} {d['drift']:>8.4f} {d['loc_shift']:>8.4f} "
                  f"{d['scale_ratio']:>8.4f}{'' if inside else '   (outside trained range)'}")
            if d["loc_shift"] < best_loc:
                best_beta, best_loc = beta, d["loc_shift"]
        if role == "B":
            effective[path] = best_beta
            edge = best_beta in (args.drift_grid[0], args.drift_grid[-1])
            print(f"   running statistics best match a batch at beta ~ {best_beta:g}"
                  f"{'  (GRID EDGE: extend --drift-grid)' if edge else ''}")

    csv_path = Path(f"{out}_drift.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n   wrote {csv_path}")
    _plot_drift(rows, trained, args.beta_var, Path(f"{out}_drift.png"))
    return effective


def _plot_drift(rows: list[dict], trained, eval_betas: list[float], path: Path) -> None:
    """Drift vs beta, one line per checkpoint, trained range shaded. Skipped without mpl."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   matplotlib not installed; skipping the figure (the CSV has the data)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for ckpt in dict.fromkeys(r["checkpoint"] for r in rows):
        sub = [r for r in rows if r["checkpoint"] == ckpt]
        role = sub[0]["role"]
        ax.plot([r["beta_var"] for r in sub], [r["drift"] for r in sub],
                marker="o", markersize=3, linestyle="-" if role == "B" else "--",
                label=f"{role}: {Path(ckpt).stem}")
    if trained is not None:
        ax.axvspan(float(trained[0]), float(trained[1]), alpha=0.12, color="grey",
                   label="trained beta range")
    for b in eval_betas:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("beta_var at evaluation")
    ax.set_ylabel("bn_edge drift (normalised units)")
    ax.set_title("Edge embedding: evaluation vs what training produced")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"   wrote {path}")


# ------------------------------------------------------------------------ evaluation
def _cache_key(role: str, path: str, cond: str, beta: float) -> str:
    return f"{role}|{path}|{cond}|{beta:g}"


def _save_cache(path: Path, settings: dict, results: dict) -> None:
    """Atomic write: a crash mid-save cannot corrupt the previous cache."""
    tmp = path.with_suffix(".tmp")
    torch.save({"settings": settings, "results": results}, tmp)
    os.replace(tmp, path)


def _evaluate_one(
    args: argparse.Namespace, loaded: LoadedPolicy, data: CVRPTWData,
    calib: CVRPTWData, cond: str, beta: float, weight: float,
) -> dict[str, Tensor]:
    """One paired evaluation under one intervention. Returns per-instance (B,) on CPU."""
    policy = loaded.policy
    cfg = replace(loaded.sim_cfg, beta_var=beta)
    with ExitStack() as stack:
        if hasattr(policy, "gate_w"):                    # M = D' with lambda forced to 0
            stack.enter_context(forced_gate(policy, 0.0))
        if "bn_batch" in cond:
            stack.enter_context(
                batch_stat_edge_norm(policy, edge_norm_stats(policy, calib, cfg))
            )
        if "dec_ref" in cond:
            stack.enter_context(scaled_decoder_variance(policy, _beta_ref(loaded) / beta))
        out = evaluate_stoch(
            policy, data, cfg, tardiness_weight=weight, n_starts=args.n_starts,
            augment=not args.no_augment, eval_batch_size=args.eval_batch_size,
            scenario_seeds=tuple(range(args.scenarios)),
        )
    return {k: out[k].detach().cpu() for k in _KEYS}


def run_evaluations(
    args: argparse.Namespace,
    named: list[tuple[str, str, LoadedPolicy]],
    data: CVRPTWData,
    calib: CVRPTWData,
    weight: float,
    out: str,
) -> dict[str, dict[str, Tensor]]:
    settings = {"test_size": args.test_size, "test_seed": args.test_seed,
                "scenarios": args.scenarios, "n_starts": args.n_starts,
                "augment": not args.no_augment, "calib_seed": args.calib_seed,
                "calib_size": args.calib_size}
    cache_path = Path(f"{out}_cache.pt")
    results: dict[str, dict[str, Tensor]] = {}
    if args.resume and cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        if blob["settings"] != settings:
            raise ValueError(
                f"{cache_path} was written with different settings {blob['settings']}; "
                f"refusing to mix them with {settings}. Use a new --out or drop --resume"
            )
        results = blob["results"]
        print(f"\n   resuming: {len(results)} evaluations already in {cache_path}")

    jobs = [(role, path, loaded, cond, beta)
            for role, path, loaded in named
            for cond in (B_CONDITIONS if role == "B" else M_CONDITIONS)
            for beta in args.beta_var]
    todo = [j for j in jobs if _cache_key(j[0], j[1], j[3], j[4]) not in results]
    print(f"\n2. COUNTERFACTUAL EVALUATION  ({len(jobs)} evaluations, {len(todo)} to run)")
    t0 = time.time()
    for i, (role, path, loaded, cond, beta) in enumerate(todo, start=1):
        res = _evaluate_one(args, loaded, data, calib, cond, beta, weight)
        results[_cache_key(role, path, cond, beta)] = res
        _save_cache(cache_path, settings, results)
        elapsed = time.time() - t0
        print(f"   [{i:>3}/{len(todo)}] {role} {Path(path).stem:<24} {cond:<17} "
              f"beta {beta:<5g} cost {float(res['cost'].mean()):.4f} | "
              f"{elapsed / 60:.1f} min, ~{elapsed / i * (len(todo) - i) / 60:.1f} min left",
              flush=True)
    return results


# -------------------------------------------------------------------------- analysis
def _share(removed: float, total: float) -> float | None:
    """Fraction of `total` removed; None when there is nothing (positive) to explain."""
    return None if total <= 1e-9 else removed / total


def _verdict(share: float | None, agree: bool) -> str:
    if share is None:
        return "no excess degradation to explain"
    if share >= MAIN_SHARE and agree:
        return "MAIN mechanism"
    if share >= MINOR_SHARE:
        return "PARTIAL" + ("" if agree else " (seeds disagree)")
    return "NOT the mechanism"


def analyse(
    args: argparse.Namespace, results: dict[str, dict[str, Tensor]],
    effective: dict[str, float], out: str,
) -> None:
    betas = list(args.beta_var)
    b0, worst = betas[0], betas[-1]

    def per_seed(role: str, cond: str, beta: float, key: str = "cost") -> list[Tensor]:
        paths = args.baseline if role == "B" else args.control
        return [results[_cache_key(role, p, cond, beta)][key].double() for p in paths]

    def avg(role: str, cond: str, beta: float, key: str = "cost") -> Tensor:
        return torch.stack(per_seed(role, cond, beta, key)).mean(dim=0)   # (n_inst,)

    summary: dict = {"settings": vars(args), "effective_beta": effective,
                     "thresholds": {"main": MAIN_SHARE, "minor": MINOR_SHARE}}

    # --- table A: mean cost and its parts
    print("\nA. MEAN COST (seed-averaged) and components of the selected route")
    print(f"   {'beta':>5} {'arm':>4} {'condition':<17} {'cost':>8} {'dist':>8} "
          f"{'tard':>8} {'late':>7}")
    rows_a = []
    for beta in betas:
        for role, conds in (("M", M_CONDITIONS), ("B", B_CONDITIONS)):
            for cond in conds:
                vals = {k: float(avg(role, cond, beta, k).mean()) for k in _KEYS}
                rows_a.append({"beta": beta, "arm": role, "condition": cond, **vals})
                print(f"   {beta:>5g} {role:>4} {cond:<17} {vals['cost']:>8.4f} "
                      f"{vals['distance']:>8.4f} {vals['tardiness']:>8.4f} "
                      f"{vals['late_count']:>7.3f}")
    summary["mean_cost"] = rows_a

    # --- table B: placebo
    print("\nB. PLACEBO  M(bn_batch) - M(plain): must be ~0 (M's edge feature ignores beta)")
    for beta in betas:
        p = paired_difference(avg("M", "bn_batch", beta), avg("M", "plain", beta))
        print(f"   beta {beta:<5g} {p['mean']:>+8.4f}  [{p['lo']:>+8.4f}, {p['hi']:>+8.4f}]"
              f"{'  significant -- INVESTIGATE' if p['significant'] else ''}")

    # --- table C: gap of each B condition to mean-only, and share of it closed
    print("\nC. GAP TO MEAN-ONLY  B(cond) - M(plain), paired; 'closed' = share of the plain "
          "gap removed")
    rows_c = []
    for beta in betas:
        gap_plain = float((avg("B", "plain", beta) - avg("M", "plain", beta)).mean())
        for cond in B_CONDITIONS:
            p = paired_difference(avg("B", cond, beta), avg("M", "plain", beta))
            share = _share(gap_plain - p["mean"], gap_plain) if cond != "plain" else None
            rows_c.append({"beta": beta, "condition": cond, **dict(p), "closed": share})
            closed = f"{share:>7.1%}" if share is not None else " " * 7
            print(f"   beta {beta:<5g} {cond:<17} {p['mean']:>+8.4f} "
                  f"[{p['lo']:>+8.4f}, {p['hi']:>+8.4f}] closed {closed}")
    summary["gap_to_M"] = rows_c

    # --- table D: excess degradation over M, share removed, per-seed sign agreement
    print(f"\nD. EXCESS DEGRADATION over M from beta {b0:g}  "
          f"[(B(beta) - B({b0:g})) - (M(beta) - M({b0:g}))]")
    rows_d = []
    m_deg = {beta: avg("M", "plain", beta) - avg("M", "plain", b0) for beta in betas[1:]}
    for beta in betas[1:]:
        exc_plain = float((avg("B", "plain", beta) - avg("B", "plain", b0) - m_deg[beta]).mean())
        for cond in B_CONDITIONS:
            b_deg = avg("B", cond, beta) - avg("B", cond, b0)
            p = paired_difference(b_deg, m_deg[beta])
            # per-seed: does the intervention REDUCE B's own degradation in every seed?
            seed_eff = [
                float(((c1 - c0) - (p1 - p0)).mean())
                for c1, c0, p1, p0 in zip(per_seed("B", cond, beta), per_seed("B", cond, b0),
                                          per_seed("B", "plain", beta),
                                          per_seed("B", "plain", b0))
            ]
            agree = cond == "plain" or all(e < 0 for e in seed_eff)
            share = _share(exc_plain - p["mean"], exc_plain) if cond != "plain" else None
            rows_d.append({"beta": beta, "condition": cond, **dict(p), "removed": share,
                           "per_seed_effect": seed_eff, "all_seeds_reduce": agree})
            removed = f"{share:>7.1%}" if share is not None else " " * 7
            seeds = ("" if cond == "plain"
                     else "  per-seed " + " ".join(f"{e:+.3f}" for e in seed_eff)
                     + ("  (all reduce)" if agree else "  (NOT all reduce)"))
            print(f"   beta {beta:<5g} {cond:<17} {p['mean']:>+8.4f} "
                  f"[{p['lo']:>+8.4f}, {p['hi']:>+8.4f}] removed {removed}{seeds}")
    summary["excess_degradation"] = rows_d

    # --- verdict
    def row(cond: str) -> dict:
        return next(r for r in rows_d if r["beta"] == worst and r["condition"] == cond)

    id_bn = next(r for r in rows_c if r["beta"] == b0 and r["condition"] == "bn_batch")
    print(f"\n3. VERDICT at beta {worst:g}  (thresholds fixed in advance: MAIN >= "
          f"{MAIN_SHARE:.0%} with every seed agreeing, NOT < {MINOR_SHARE:.0%})")
    verdict = {}
    for cond, label in (("bn_batch", "encoder: BN train/eval + encoder scale"),
                        ("dec_ref", "decoder: linear extrapolation in w_edge"),
                        ("bn_batch+dec_ref", "all scale effects together")):
        r = row(cond)
        verdict[cond] = _verdict(r["removed"], r["all_seeds_reduce"])
        share = f"{r['removed']:.1%}" if r["removed"] is not None else "n/a"
        print(f"   {label:<42} removes {share:>7}  -> {verdict[cond]}")
    both = row("bn_batch+dec_ref")["removed"]
    if both is not None:
        print(f"   residual NOT explained by scale (mechanism (a) or optimisation): "
              f"{1.0 - both:.1%}")
    id_share = id_bn["closed"]
    print(f"   in distribution (beta {b0:g}): bn_batch closes "
          f"{'n/a' if id_share is None else f'{id_share:.1%}'} of B's gap to M "
          f"-- the pure train/eval discrepancy, no extrapolation involved")
    summary["verdict"] = verdict

    path = Path(f"{out}_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n   wrote {path}")


# ------------------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()
    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _tee_console(Path(f"{out}.log"))
    device = torch.device(args.device)
    print("command:", " ".join(sys.argv))

    base = [load_stoch_policy(p, device) for p in args.baseline]
    ctrl = [load_stoch_policy(p, device) for p in args.control]
    for path, l in zip(args.baseline, base):
        if l.variant != "B":
            raise ValueError(f"--baseline {path} is variant {l.variant}, expected B")
    for path, l in zip(args.control, ctrl):
        if l.policy.decoder.edge_feat_dim > SIGMA2:
            raise ValueError(
                f"--control {path} (variant {l.variant}) reads sigma^2 in its decoder; "
                f"the control must be mean-only"
            )
    paths = [*args.baseline, *args.control]
    loaded_all = [*base, *ctrl]
    assert_matched_budget(list(zip(paths, (l.blob for l in loaded_all))),
                          allow=args.allow_budget_mismatch)
    _assert_one_problem(paths, loaded_all)
    if args.calib_seed == args.test_seed:
        raise ValueError("--calib-seed must differ from --test-seed")

    trained = base[0].blob.get("beta_var_range")
    b0 = args.beta_var[0]
    if trained is not None and not trained[0] - 1e-9 <= b0 <= trained[1] + 1e-9:
        raise ValueError(f"first --beta-var {b0} is outside the trained range {trained}; "
                         f"it is the in-distribution reference")
    weight = float(base[0].blob["tardiness_weight"])
    print(f"omega {base[0].sim_cfg.omega} | trained beta range {trained} | "
          f"beta_ref (decoder rescale) {_beta_ref(base[0]):g} | tardiness_weight {weight}")

    named = ([("B", p, l) for p, l in zip(args.baseline, base)]
             + [("M", p, l) for p, l in zip(args.control, ctrl)])
    calib = _generate(base[0], args.calib_size, args.calib_seed, device)
    effective = run_drift(args, named, calib, out)
    if args.drift_only:
        print("\n--drift-only: stopping before the rollouts.")
        return

    data = _generate(base[0], args.test_size, args.test_seed, device)
    results = run_evaluations(args, named, data, calib, weight, out)
    analyse(args, results, effective, out)
    print("\ndone.")


if __name__ == "__main__":
    main()