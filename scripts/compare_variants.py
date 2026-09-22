"""Compare two trained variants with PAIRED statistics, and sweep the gate at inference.

Example:
    python -m scripts.compare_variants --baseline checkpoints/cvrptwB.pt \
        --candidate checkpoints/cvrptwD.pt --beta-var 0.3 0.6 0.9

WHY PAIRED. Mean cost varies enormously from instance to instance (an easy instance might
cost 6, a hard one 14), and that spread dwarfs the difference between two policies. Two
independent averages therefore cannot tell you whether a gap of 0.06 is real. Because both
variants are evaluated here on the SAME instances under the SAME simulated days, the
difference can be taken instance by instance: the instance difficulty cancels, and what is
left is the policy difference. That is a far more sensitive test, and the only honest way
to report a gap this small.

WHAT IS ACTUALLY BEING CLAIMED. The thesis does not claim the gate lowers cost; it claims
the gate lowers the DEGRADATION carried from the training variance to a shifted one. So
the headline test is paired on that quantity directly: for each instance,
(candidate_ood - candidate_id) - (baseline_ood - baseline_id). A confidence interval that
sits entirely below zero supports the claim; one that straddles zero does not.

THE GATE ABLATIONS. With --gate-sweep the candidate's softplus(w) is OVERRIDDEN at inference
time, with no retraining, so the gate's effect is isolated from everything else the
network learned. If forcing it to 0 reproduces the baseline, the gate is doing work at
inference. If nothing moves, the gate's contribution was to shape LEARNING rather than to
act at decision time -- a different finding, and worth knowing which one you have.
--sed-gate-off is the same ablation for variant C, whose gate has no single scalar to
sweep: it switches every SED2AM-style gate off at inference, which reduces C's FUNCTION to
variant B's. A trained C with its gates off is NOT variant B, though: its other weights
were trained with the gates on and adapted to them, exactly as D's did.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import Tensor

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig
from src.eval.checkpoints import load_stoch_policy
from src.eval.evaluate_stoch import evaluate_stoch

_Z95 = 1.959964


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired comparison of two trained variants")
    p.add_argument("--baseline", type=str, required=True, help="e.g. variant-B checkpoint")
    p.add_argument("--candidate", type=str, required=True, help="e.g. variant-D checkpoint")
    p.add_argument("--beta-var", type=float, nargs="+", default=[0.3, 0.6, 0.9],
                   help="variance levels; the FIRST is treated as in-distribution")
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--test-seed", type=int, default=4242,
                   help="fresh instances, not the validation seed used for model selection")
    p.add_argument("--scenarios", type=int, default=4, help="independent days, averaged")
    p.add_argument("--n-starts", type=int, default=None, help="None = all customers")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-augment", action="store_true", help="skip 8x augmentation (faster)")
    p.add_argument("--gate-sweep", type=float, nargs="*", default=None,
                   help="variant D: override softplus(w) at inference, "
                        "e.g. --gate-sweep 0 0.5 2 4 8")
    p.add_argument("--sed-gate-off", action="store_true",
                   help="variant C: also report the candidate with its SED2AM-style gates "
                        "switched off at inference (C's analogue of softplus(w)=0)")
    p.add_argument("--select-gate-on-val", action="store_true",
                   help="choose the sweep value that is best on a HELD-OUT validation set "
                        "at the in-distribution variance only, then report it on the test "
                        "set. Turns the sweep from cherry-picking into a legitimate "
                        "hyperparameter choice")
    p.add_argument("--gate-val-seed", type=int, default=777,
                   help="instances for gate selection; must differ from --test-seed AND "
                        "from the training validation seed")
    p.add_argument("--gate-val-size", type=int, default=128)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class Paired(dict):
    """Mean paired difference with a 95% interval. Keys: mean, lo, hi, se, significant."""


def paired_difference(candidate: Tensor, baseline: Tensor) -> Paired:
    """Per-instance difference candidate - baseline, with a 95% normal-approx interval.

    Negative means the candidate is CHEAPER. `significant` is True when the whole interval
    sits on one side of zero.
    """
    if candidate.shape != baseline.shape:
        raise ValueError(
            f"paired inputs must align: {tuple(candidate.shape)} vs {tuple(baseline.shape)}"
        )
    diff = (candidate - baseline).double()                    # (n_instances,)
    n = diff.numel()
    mean = float(diff.mean())
    se = float(diff.std(unbiased=True)) / math.sqrt(n) if n > 1 else float("inf")
    lo, hi = mean - _Z95 * se, mean + _Z95 * se
    return Paired(mean=mean, lo=lo, hi=hi, se=se, significant=(lo > 0.0 or hi < 0.0))


def _mark(p: Paired) -> str:
    return "  significant" if p["significant"] else "  (not significant)"


@contextmanager
def forced_gate(policy, strength: float):
    """Temporarily override softplus(w) at inference. Restores the method on exit.

    The method is patched rather than the parameter, so a strength of EXACTLY zero is
    reachable (softplus has no finite pre-image of 0) and the learned value is never
    touched -- nothing is written back to the checkpoint.
    """
    original = policy.gate_strength
    value = torch.tensor(float(strength), device=policy.gate_w.device,
                         dtype=policy.gate_w.dtype)
    policy.gate_strength = lambda: value
    try:
        yield
    finally:
        policy.gate_strength = original


@contextmanager
def sed_gates(policy, active: bool):
    """Temporarily switch variant C's SED2AM-style gates on or off. Restores on exit.

    With active=False the encoder gate is skipped and the pointer penalty is None, so the
    computation is variant B's -- see gate-chain link 5.
    """
    was = policy.sed_gate_active
    policy.set_sed_gates(active)
    try:
        yield
    finally:
        policy.set_sed_gates(was)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    base = load_stoch_policy(args.baseline, device)
    cand = load_stoch_policy(args.candidate, device)
    augment = not args.no_augment
    scenario_seeds = tuple(range(args.scenarios))

    if base.blob["tardiness_weight"] != cand.blob["tardiness_weight"]:
        raise ValueError(
            f"the two checkpoints were trained with different tardiness weights "
            f"({base.blob['tardiness_weight']} vs {cand.blob['tardiness_weight']}); the "
            f"reward defines the problem, so the comparison would be uninterpretable"
        )
    weight = float(base.blob["tardiness_weight"])
    for key in ("n_customers", "capacity", "shift_end", "tw_half_width", "service_time"):
        if base.blob[key] != cand.blob[key]:
            raise ValueError(f"checkpoints disagree on '{key}' -- not the same problem")

    data = generate_cvrptw(
        args.test_size, base.blob["n_customers"], base.blob["capacity"],
        service_time=base.blob["service_time"], shift_end=base.blob["shift_end"],
        tw_half_width=base.blob["tw_half_width"], seed=args.test_seed, device=device,
    )

    print(f"baseline  {args.baseline}  (variant {base.variant})")
    print(f"candidate {args.candidate}  (variant {cand.variant})")
    print(f"  tardiness_weight {weight} held FIXED | {args.test_size} instances "
          f"(seed {args.test_seed}) | {args.scenarios} days | augment={augment}\n")

    def evaluate(loaded, beta_var: float, instances=None,
                 strength: float | None = None,
                 gates_off: bool = False) -> dict[str, Tensor]:
        cfg = replace(loaded.sim_cfg, beta_var=beta_var)
        target = data if instances is None else instances

        def run() -> dict[str, Tensor]:
            return evaluate_stoch(
                loaded.policy, target, cfg, tardiness_weight=weight,
                n_starts=args.n_starts, augment=augment,
                eval_batch_size=args.eval_batch_size, scenario_seeds=scenario_seeds,
            )

        if gates_off and hasattr(loaded.policy, "set_sed_gates"):
            with sed_gates(loaded.policy, False):
                return run()
        if strength is None or not hasattr(loaded.policy, "gate_w"):
            return run()
        with forced_gate(loaded.policy, strength):
            return run()

    # ------------------------------------------- gate selection on held-out instances
    selected: float | None = None
    if args.select_gate_on_val:
        if args.gate_sweep is None:
            raise ValueError("--select-gate-on-val requires --gate-sweep")
        if not hasattr(cand.policy, "gate_w"):
            raise ValueError("--select-gate-on-val needs a variant-D candidate")
        if args.gate_val_seed == args.test_seed:
            raise ValueError(
                "--gate-val-seed must differ from --test-seed, or the gate would be "
                "chosen on the very instances it is then reported on"
            )
        val = generate_cvrptw(
            args.gate_val_size, base.blob["n_customers"], base.blob["capacity"],
            service_time=base.blob["service_time"], shift_end=base.blob["shift_end"],
            tw_half_width=base.blob["tw_half_width"], seed=args.gate_val_seed,
            device=device,
        )
        id_beta = args.beta_var[0]
        print(f"GATE SELECTION on {args.gate_val_size} held-out instances "
              f"(seed {args.gate_val_seed}), at the IN-DISTRIBUTION variance "
              f"beta_var={id_beta} ONLY -- no test or shifted data is consulted")
        print(f"{'softplus(w)':>12} {'val cost':>10}")
        print("-" * 23)
        scores = {}
        for strength in args.gate_sweep:
            c = float(evaluate(cand, id_beta, instances=val,
                               strength=strength)["cost"].mean())
            scores[strength] = c
            print(f"{strength:>12.4f} {c:>10.4f}")
        selected = min(scores, key=scores.get)
        print(f"  selected softplus(w) = {selected:g} "
              f"(val cost {scores[selected]:.4f})\n")

    base_cost: dict[float, Tensor] = {}
    cand_cost: dict[float, Tensor] = {}

    # ------------------------------------------------------------- cost, level by level
    header = (f"{'beta_var':>9} {'baseline':>10} {'candidate':>10} {'diff':>9} "
              f"{'95% CI':>20}")
    note = f" | candidate gate forced to {selected:g}" if selected is not None else ""
    print(f"PAIRED COST  (per instance; negative diff = candidate cheaper){note}")
    print(header)
    print("-" * (len(header) + 22))
    for beta_var in args.beta_var:
        b = evaluate(base, beta_var)
        c = evaluate(cand, beta_var, strength=selected)
        base_cost[beta_var], cand_cost[beta_var] = b["cost"], c["cost"]
        p = paired_difference(c["cost"], b["cost"])
        print(f"{beta_var:>9.3f} {float(b['cost'].mean()):>10.4f} "
              f"{float(c['cost'].mean()):>10.4f} {p['mean']:>+9.4f} "
              f"[{p['lo']:>+8.4f}, {p['hi']:>+8.4f}]{_mark(p)}")

    # ------------------------------------------------------- the claim: degradation
    id_beta = args.beta_var[0]
    if len(args.beta_var) > 1:
        print(f"\nPAIRED DEGRADATION from beta_var={id_beta} (THE THESIS CLAIM)")
        print("  per instance: (candidate_ood - candidate_id) - (baseline_ood - baseline_id)")
        print("  negative and significant => the gate shrinks the degradation")
        print(header.replace("baseline", "base degr").replace("candidate", "cand degr"))
        print("-" * (len(header) + 22))
        for beta_var in args.beta_var[1:]:
            b_deg = base_cost[beta_var] - base_cost[id_beta]
            c_deg = cand_cost[beta_var] - cand_cost[id_beta]
            p = paired_difference(c_deg, b_deg)
            print(f"{beta_var:>9.3f} {float(b_deg.mean()):>10.4f} "
                  f"{float(c_deg.mean()):>10.4f} {p['mean']:>+9.4f} "
                  f"[{p['lo']:>+8.4f}, {p['hi']:>+8.4f}]{_mark(p)}")

    # --------------------------------------------------------------- the gate sweep
    if args.gate_sweep is not None:
        if not hasattr(cand.policy, "gate_w"):
            raise ValueError("--gate-sweep needs a variant-D candidate (it has no gate)")
        learned = float(cand.policy.gate_strength().detach())
        print(f"\nGATE SWEEP  (inference-time override of softplus(w); learned = "
              f"{learned:.4f}; no retraining)")
        cols = "".join(f"{f'cost@{bv:g}':>11}" for bv in args.beta_var)
        print(f"{'softplus(w)':>12}{cols}{'vs baseline@ID':>16}")
        print("-" * (12 + 11 * len(args.beta_var) + 16))
        for strength in args.gate_sweep:
            costs = [evaluate(cand, bv, strength=strength)["cost"]
                     for bv in args.beta_var]
            p = paired_difference(costs[0], base_cost[id_beta])
            tag = "  <- learned" if abs(strength - learned) < 1e-3 else ""
            row = "".join(f"{float(c.mean()):>11.4f}" for c in costs)
            print(f"{strength:>12.4f}{row}{p['mean']:>+16.4f}{tag}")
        print("\n  If softplus(w)=0 reproduces the baseline, the gate acts at INFERENCE.")
        print("  If the row is flat, the gate's contribution was to shape LEARNING.")

    # ------------------------------------------------- variant C: gates off ablation
    if args.sed_gate_off:
        if not hasattr(cand.policy, "set_sed_gates"):
            raise ValueError(
                "--sed-gate-off needs a variant-C candidate (it has no SED2AM-style gate)"
            )
        print("\nSED2AM GATE ABLATION  (candidate's gates switched off at inference; "
              "no retraining)")
        cols = "".join(f"{f'cost@{bv:g}':>11}" for bv in args.beta_var)
        print(f"{'candidate':>12}{cols}")
        print("-" * (12 + 11 * len(args.beta_var)))
        for label, off in (("gates on", False), ("gates off", True)):
            costs = [evaluate(cand, bv, gates_off=off)["cost"] for bv in args.beta_var]
            print(f"{label:>12}" + "".join(f"{float(c.mean()):>11.4f}" for c in costs))
        print(f"{'baseline':>12}"
              + "".join(f"{float(base_cost[bv].mean()):>11.4f}" for bv in args.beta_var))
        print("\n  Gates off reduces C's FUNCTION to variant B's, but not its WEIGHTS:")
        print("  they were trained with the gates on. A large gap means the gate acts at")
        print("  inference; costs worse than the baseline mean the network relies on it.")


if __name__ == "__main__":
    main()