"""Train a stochastic CVRPTW policy (variant B, C, D, D-prime, M0 or VGT-R) with POMO.

Example:
    python scripts/train_stoch.py --n-customers 20 --steps 800 --beta-var 0.3 \
        --tardiness-weight 2.0 --ood-beta-var 0.6

Mirrors scripts/train_cvrptw.py: same calibrated instance dials, same certified
optimisation recipe (batch 32, lr 3e-4 + cosine to 1e-6). What it adds is the Edge Bank
-- travel times ripple with the clock and are sampled, so the reward carries a tardiness
penalty -- and three things that exist to keep the experiment honest:

  * `--ood-beta-var` evaluates the SAME checkpoint under a shifted variance every eval
    step, so the out-of-distribution curve is visible during training rather than
    discovered at the end.
  * `--gate-check-every` runs the gate chain (bank off => certified env; zeroed w_edge =>
    variant-A pointer; empty gate slot => variant B), so drift is caught at the step it
    appears.
  * Validation uses FIXED scenario seeds, so every eval and every variant faces the same
    days -- a paired comparison, not a re-roll of the weather.

Run from the repo root so `src` is importable.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.simulator import SimulatorConfig, assert_ripple_reachable
from src.eval.evaluate_stoch import evaluate_stoch
from src.models.cvrptw_sed2am_policy import DEFAULT_SED_GATE_BIAS_INIT
from src.models.factory import (
    DEFAULT_CVRPTW_GATED_CONFIG,
    DEFAULT_CVRPTW_GATEONLY_CONFIG,
    DEFAULT_CVRPTW_MEANONLY_CONFIG,
    DEFAULT_CVRPTW_RISKBIAS_CONFIG,
    DEFAULT_CVRPTW_SED2AM_CONFIG,
    DEFAULT_CVRPTW_STOCH_CONFIG,
    build_cvrptw_gate_control_policy,
    build_cvrptw_gated_policy,
    build_cvrptw_gateonly_policy,
    build_cvrptw_meanonly_policy,
    build_cvrptw_riskbias_policy,
    build_cvrptw_sed2am_policy,
    build_cvrptw_stoch_policy,
)
from src.models.cvrptw_gate_controls import GATE_SIGNALS, softplus_inverse
from src.models.gate_location import GATE_LOCATIONS, restrict_gate
from src.models.gate_checks import assert_gate_chain_inert
from src.train.gate_schedule import GATE_SCHEDULES, gate_scale
from src.train.day_replication import replicate_days
from src.train.pomo_stoch import build_optimizer, train_step_stoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a stochastic CVRPTW policy with POMO")
    # problem (calibrated dials, locked for A/B/C/D)
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--capacity", type=int, default=20, help="calibrated CVRPTW capacity")
    p.add_argument("--shift-end", type=float, default=4.0)
    p.add_argument("--tw-half-width", type=float, default=0.5)
    p.add_argument("--service-time", type=float, default=0.2)
    # simulator (the Edge Bank)
    p.add_argument("--n-bins", type=int, default=24, help="K time bins per shift")
    p.add_argument("--alpha", type=float, default=0.3, help="rush-hour ripple amplitude")
    p.add_argument("--beta-var", type=float, default=0.3,
                   help="variance scale (OOD knob); also the level validation reports at")
    p.add_argument("--beta-var-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="domain randomisation: resample beta_var uniformly in [LO, HI] "
                        "for every training mini-batch. Validation still reports at "
                        "--beta-var, so the numbers stay comparable across runs")
    p.add_argument("--eta", type=float, default=0.6, help="reliability-field strength")
    p.add_argument("--zeta", type=float, default=0.0,
                   help="idiosyncratic RELIABILITY-field strength (multiplicative). 0.0 "
                        "reproduces the previous model bit-exactly; |eta| + zeta must "
                        "stay below 1. NOTE this scales r, which multiplies mu^2, so it "
                        "does NOT decorrelate sigma^2 from edge length -- use --omega")
    p.add_argument("--risk-freq", type=int, default=3,
                   help="spatial frequency of the idiosyncratic risk field (integer)")
    p.add_argument("--omega", type=float, default=0.0,
                   help="idiosyncratic SHARE of the variance, in [0, 1]. 0.0 reproduces "
                        "the geometric model bit-exactly. Higher values replace part of "
                        "the mu^2 variance with an additive, neighbourhood-driven one, "
                        "lowering corr(mu, sigma^2) -- the precondition for a "
                        "variance-specific inductive bias to be able to help at all")
    p.add_argument("--field-scale", type=float, default=0.3)
    p.add_argument("--phase-k", type=float, default=1.0, help="spatial congestion lag")
    p.add_argument("--tardiness-weight", type=float, default=2.0,
                   help="reward penalty per violation; calibrate with the sweep first")
    p.add_argument("--reward-mode", type=str, default="analytic",
                   choices=["analytic", "sampled", "hybrid"],
                   help="analytic = expected tardiness in closed form (learnable); "
                        "sampled = one realised day (measured to be unlearnable under POMO)")
    p.add_argument("--no-crn", action="store_true",
                   help="draw fresh noise per traversal instead of one day per instance")
    p.add_argument("--days-per-instance", type=int, default=1,
                   help="E1 (equal-compute day averaging): fill each mini-batch with "
                        "batch_size / K distinct instances, each repeated on K independent "
                        "days. Every (instance, day) row keeps its own POMO group and shared "
                        "baseline, so the rollouts per step are unchanged. 1 = the usual "
                        "batch, bit-identical to before")
    # optimisation (certified recipe)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--n-starts", type=int, default=None, help="None = all customers")
    # variant
    p.add_argument("--variant", type=str, default="b",
                   choices=["b", "c", "d", "dprime", "m0", "vgtr"],
                   help="b = (mu, sigma^2) edge features, no gate; "
                        "c = b plus a SED2AM-style learned sigmoid gate (the competitor); "
                        "d = b plus the shared monotone variance gate; "
                        "dprime = d with sigma^2 REMOVED from the encoder edge feature and "
                        "from the decoder's w_edge, so the gate is the only path from "
                        "variance to a decision and the monotone constraint actually binds; "
                        "m0 = TRUE mean-only: dprime's architecture with NO gate, so "
                        "sigma^2 influences neither training nor decisions; "
                        "vgtr = dprime plus a learnable lateness-risk bias "
                        "-alpha * P(late at j) in the decoder, kept at evaluation")
    p.add_argument("--risk-init", type=float, default=None,
                   help="variant vgtr: initial risk_w; default softplus^-1(1) so the "
                        "risk-bias strength alpha starts at 1")
    p.add_argument("--risk-signal", type=str, default="sigma2",
                   choices=["sigma2", "const", "shuffle", "fixed"],
                   help="variant vgtr: variance used to build P(late). sigma2 = the true "
                        "variance; const / shuffle / fixed = controls (see "
                        "src/models/cvrptw_riskbias_policy.py)")
    p.add_argument("--gate-init", type=float, default=0.0,
                   help="variants D/D-prime: initial w; softplus(0)=0.693 starts D as ~B")
    p.add_argument("--gate-lr", type=float, default=1e-2,
                   help="variants C, D and D-prime: dedicated lr for the gate parameters. "
                        "At the global 3e-4 the gate could not travel far enough for the "
                        "experiment to be informative in either direction. The SAME value "
                        "is used for every variant, so none is handicapped")
    # mechanism tests for the TRAINING-TIME gate (variant D-prime only; see
    # src/models/cvrptw_gate_controls.py and src/train/gate_schedule.py)
    p.add_argument("--gate-signal", type=str, default="sigma2", choices=list(GATE_SIGNALS),
                   help="D-prime only: what the gate penalises. sigma2 = the true variance "
                        "(plain D-prime); shuffle / const / dist = magnitude-matched "
                        "controls. The environment always uses the true variance")
    p.add_argument("--gate-schedule", type=str, default="constant",
                   choices=list(GATE_SCHEDULES),
                   help="D-prime only: when the gate is active. early = on until "
                        "--gate-off-frac of the steps, then off; anneal = linearly to 0 "
                        "at --gate-off-frac, then off")
    p.add_argument("--gate-off-frac", type=float, default=None,
                   help="fraction of the steps at which an early/anneal gate reaches 0")
    p.add_argument("--gate-fixed", type=float, default=None,
                   help="D-prime only: FREEZE the gate at this lambda = softplus(w) for "
                        "the whole run (dose-response). Exact: w gets no gradient")
    p.add_argument("--gate-location", type=str, default="both", choices=list(GATE_LOCATIONS),
                   help="D-prime only: where the gate acts. both = encoder and decoder "
                        "(plain D-prime); encoder / decoder = the other path switched off. "
                        "See src/models/gate_location.py")
    p.add_argument("--sed-gate-bias-init", type=float, default=DEFAULT_SED_GATE_BIAS_INIT,
                   help="variant C: initial gate bias; sigmoid(2.0)=0.881 starts C as ~B")
    # evaluation
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--val-seed", type=int, default=9999)
    p.add_argument("--val-scenarios", type=int, default=2,
                   help="independent days averaged per eval (fixed seeds => paired)")
    p.add_argument("--val-augment", action="store_true",
                   help="8x augmentation at eval (slower; off during training)")
    p.add_argument("--eval-batch-size", type=int, default=8, help="instance chunk, for 4GB")
    p.add_argument("--ood-beta-var", type=float, default=None,
                   help="also evaluate under this shifted variance each eval step")
    # run
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0,
                   help="also keep a PERMANENT checkpoint every N steps, named "
                        "<out>_step<N>.pt. One long run then yields the whole budget "
                        "curve (cost and gate strength vs training steps) instead of one "
                        "endpoint, so 'does the effect depend on the budget?' is answered "
                        "without retraining at each budget. Rounded up to a multiple of "
                        "--eval-every, since checkpoints are written at evaluations")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--gate-check-every", type=int, default=200,
                   help="0 disables the gate-chain assertions")
    p.add_argument("--out", type=str, default="checkpoints/cvrptwB_best.pt")
    p.add_argument("--init-from", type=str, default=None,
                   help="fine-tune: load this checkpoint's weights before training. Must be "
                        "the same variant, simulator and tardiness_weight. Its step count is "
                        "added to the recorded budget, so matched-budget checks stay valid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    sim_cfg = SimulatorConfig(
        n_bins=args.n_bins, period=args.shift_end, alpha=args.alpha,
        beta_var=args.beta_var, phase_k=args.phase_k, shift_end=args.shift_end,
        eta=args.eta, field_scale=args.field_scale,
        zeta=args.zeta, risk_freq=args.risk_freq, omega=args.omega,
    )
    # Crash-not-silent: a ripple this strong would make far-corner customers unservable,
    # which breaks feasibility-by-construction and would show up as a stalled rollout.
    assert_ripple_reachable(sim_cfg, args.service_time)
    if not sim_cfg.is_mean_preserving:
        raise ValueError(
            f"period={sim_cfg.period} does not span whole cycles over shift_end="
            f"{sim_cfg.shift_end}; the ripple would shift mean travel time and invalidate "
            f"the instance calibration"
        )

    # E1: K days per instance at equal compute. Validated before anything is built so a
    # bad flag cannot fail after the checkpoint has been loaded.
    k_days = args.days_per_instance
    if k_days < 1 or args.batch_size % k_days != 0:
        raise ValueError(f"--days-per-instance must be >= 1 and divide --batch-size "
                         f"({args.batch_size}), got {k_days}")
    if k_days > 1 and args.reward_mode == "analytic":
        raise ValueError("--days-per-instance > 1 is meaningless with the analytic reward "
                         "(it draws no day)")
    if k_days > 1 and args.no_crn:
        raise ValueError("--days-per-instance > 1 needs common random numbers: without "
                         "them each traversal draws fresh noise and a 'day' is undefined")
    if k_days > 1 and args.accum_steps != 1:
        raise ValueError("--days-per-instance > 1 was written for --accum-steps 1")

    gate_mods = (args.gate_signal != "sigma2" or args.gate_schedule != "constant"
                 or args.gate_fixed is not None or args.gate_location != "both")
    if args.risk_signal != "sigma2" and args.variant != "vgtr":
        raise ValueError("--risk-signal applies to --variant vgtr only")
    if gate_mods and args.variant != "dprime":
        raise ValueError("--gate-signal / --gate-schedule / --gate-fixed / --gate-location "
                         "apply to --variant dprime only")
    if args.variant == "dprime":
        gate_init = (softplus_inverse(args.gate_fixed) if args.gate_fixed is not None
                     else args.gate_init)
        model_config = {**DEFAULT_CVRPTW_GATEONLY_CONFIG, "gate_init": gate_init}
        if args.gate_signal != "sigma2":
            model_config["gate_signal"] = args.gate_signal
            policy = build_cvrptw_gate_control_policy(model_config).to(device)
        else:
            policy = build_cvrptw_gateonly_policy(model_config).to(device)
        if args.gate_fixed is not None:
            policy.gate_w.requires_grad_(False)     # exactly frozen: no gradient, no drift
        if args.gate_location != "both":
            model_config["gate_location"] = args.gate_location   # loader re-applies it
            restrict_gate(policy, args.gate_location)
    elif args.variant == "d":
        model_config = {**DEFAULT_CVRPTW_GATED_CONFIG, "gate_init": args.gate_init}
        policy = build_cvrptw_gated_policy(model_config).to(device)
    elif args.variant == "c":
        model_config = {**DEFAULT_CVRPTW_SED2AM_CONFIG,
                        "sed_gate_bias_init": args.sed_gate_bias_init}
        policy = build_cvrptw_sed2am_policy(model_config).to(device)
    elif args.variant == "vgtr":
        model_config = {**DEFAULT_CVRPTW_RISKBIAS_CONFIG, "gate_init": args.gate_init}
        if args.risk_init is not None:
            model_config["risk_init"] = args.risk_init
        model_config["risk_signal"] = args.risk_signal
        policy = build_cvrptw_riskbias_policy(model_config).to(device)
    elif args.variant == "m0":
        model_config = {**DEFAULT_CVRPTW_MEANONLY_CONFIG}
        policy = build_cvrptw_meanonly_policy(model_config).to(device)
    else:
        model_config = {**DEFAULT_CVRPTW_STOCH_CONFIG}
        policy = build_cvrptw_stoch_policy(model_config).to(device)

    init_steps = 0
    if args.init_from is not None:
        # weights_only=False: our own checkpoint dicts hold configs, not just tensors.
        blob = torch.load(args.init_from, map_location=device, weights_only=False)
        if blob.get("variant") != args.variant:
            raise ValueError(f"--init-from is variant {blob.get('variant')!r}, "
                             f"but --variant is {args.variant!r}")
        if float(blob["tardiness_weight"]) != args.tardiness_weight:
            raise ValueError("--init-from was trained with a different tardiness_weight")
        old_sim, new_sim = blob["sim_config"], asdict(sim_cfg)
        changed = sorted(k for k in new_sim if k != "beta_var" and old_sim.get(k) != new_sim[k])
        if changed:
            raise ValueError(f"--init-from used a different simulator: {changed}")
        policy.load_state_dict(blob["model_state"])          # strict: same architecture
        init_steps = int(blob["step"])
        print(f"initialised from {args.init_from} | variant {args.variant} | "
              f"{init_steps} steps | its reward {blob.get('reward_mode')}")

    # The gate scalar gets its own group (higher lr, no weight decay); see build_optimizer.
    optimizer = build_optimizer(
        policy, lr=args.lr, weight_decay=args.weight_decay, gate_lr=args.gate_lr
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.min_lr
        )

    train_gen = torch.Generator(device=device).manual_seed(args.seed)

    # Gate schedule: scale softplus(w) per step. Validated up front so a bad flag cannot
    # kill a night run at the first step.
    gate_scale_now = [1.0]
    if args.gate_schedule != "constant":
        gate_scale(1, args.steps, args.gate_schedule, args.gate_off_frac)   # validates
        _raw_strength = policy.gate_strength

        def _scheduled_strength() -> torch.Tensor:
            return _raw_strength() * gate_scale_now[0]

        policy.gate_strength = _scheduled_strength

    # The gate chain asserts that each stacked term is inert when switched off, and its
    # links were written against B/C/D, whose decoder reads BOTH moments. D-prime's
    # decoder reads mu alone, so the w_edge link no longer describes it. Skipped rather
    # than silently passing a weaker check -- and skipped HERE, before the run starts,
    # so a stale assertion cannot kill a training job at step 200.
    gate_check_every = args.gate_check_every
    if args.variant in ("dprime", "vgtr") and gate_check_every:
        print("  note: gate-chain checks skipped for variant D-prime (the w_edge link "
              "assumes a 2-dim decoder edge map); extend gate_checks.py to restore them")
        gate_check_every = 0

    # Training-time simulator config: one fixed level, or resampled per mini-batch.
    if args.beta_var_range is None:
        train_sim_cfg = sim_cfg
    else:
        lo, hi = args.beta_var_range
        if not 0.0 <= lo < hi:
            raise ValueError(f"--beta-var-range needs 0 <= LO < HI, got {lo} {hi}")
        cfg_gen = torch.Generator().manual_seed(args.seed + 1)

        def train_sim_cfg() -> SimulatorConfig:
            beta = lo + (hi - lo) * float(torch.rand(1, generator=cfg_gen))
            return replace(sim_cfg, beta_var=beta)

    def sample_batch():
        # E1: B/K distinct instances, each repeated K times (K=1: the usual batch, same RNG
        # consumption, so old runs are reproduced). See src/train/day_replication.py.
        base = generate_cvrptw(
            args.batch_size // k_days, args.n_customers, args.capacity,
            service_time=args.service_time, shift_end=args.shift_end,
            tw_half_width=args.tw_half_width, device=device,
        )
        return replicate_days(base, k_days)

    val = generate_cvrptw(
        args.val_size, args.n_customers, args.capacity,
        service_time=args.service_time, shift_end=args.shift_end,
        tw_half_width=args.tw_half_width, seed=args.val_seed, device=device,
    )
    scenario_seeds = tuple(range(args.val_scenarios))   # FIXED: paired across checkpoints
    ood_cfg = (
        replace(sim_cfg, beta_var=args.ood_beta_var)
        if args.ood_beta_var is not None else None
    )

    def validate(cfg: SimulatorConfig) -> dict[str, float]:
        # eval mode is REQUIRED (evaluate_stoch raises otherwise): with BatchNorm still in
        # train mode, an OOD variance shift would be partly absorbed by re-estimated
        # running statistics. Restored afterwards so training resumes unchanged.
        was_training = policy.training
        policy.eval()
        try:
            out = evaluate_stoch(
                policy, val, cfg, tardiness_weight=args.tardiness_weight,
                n_starts=args.n_starts, augment=args.val_augment,
                eval_batch_size=args.eval_batch_size, scenario_seeds=scenario_seeds,
            )
        finally:
            policy.train(was_training)
        return {k: float(v.mean()) for k, v in out.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    last_path = out_path.with_name(out_path.stem + "_last" + out_path.suffix)
    if args.save_every and args.save_every % args.eval_every != 0:
        raise ValueError(
            f"--save-every ({args.save_every}) must be a multiple of --eval-every "
            f"({args.eval_every}): checkpoints are written at evaluation steps"
        )
    t0 = time.time()
    if args.variant == "dprime":
        gate_note = (f" | gate-ONLY variance path | gate_init {gate_init:.4f} "
                     f"gate_lr {args.gate_lr:.0e} | gate signal {args.gate_signal} "
                     f"| schedule {args.gate_schedule}"
                     + (f" (off at {args.gate_off_frac:g})" if args.gate_off_frac else "")
                     + (f" | FIXED lambda {args.gate_fixed:g}"
                        if args.gate_fixed is not None else "")
                     + (f" | gate in the {args.gate_location.upper()} only"
                        if args.gate_location != "both" else ""))
    elif args.variant == "d":
        gate_note = f" | gate_init {args.gate_init} gate_lr {args.gate_lr:.0e}"
    elif args.variant == "vgtr":
        gate_note = (f" | VGT + lateness-risk bias | gate_init {args.gate_init} "
                     f"risk_init {model_config['risk_init']:.4f} (alpha "
                     f"{float(policy.risk_strength().detach()):.3f}) "
                     f"gate_lr {args.gate_lr:.0e} | risk signal {args.risk_signal}")
    elif args.variant == "m0":
        gate_note = " | MEAN-ONLY: no gate, sigma^2 reaches no decision"
    elif args.variant == "c":
        gate_note = (f" | SED2AM-style gate, bias_init {args.sed_gate_bias_init} "
                     f"gate_lr {args.gate_lr:.0e}")
    else:
        gate_note = ""
    if k_days > 1:
        gate_note += (f" | E1: {args.batch_size // k_days} instances x {k_days} days "
                      f"per mini-batch")
    if args.beta_var_range is not None:
        gate_note += (f" | beta_var randomised over "
                      f"[{args.beta_var_range[0]}, {args.beta_var_range[1]}]")
    print(f"variant {args.variant.upper()}{gate_note}")
    print(f"stochastic CVRPTW | seed {args.seed} | cap {args.capacity} "
          f"shift {args.shift_end} "
          f"tw_half {args.tw_half_width} | alpha {args.alpha} beta_var {args.beta_var} "
          f"eta {args.eta} zeta {args.zeta} omega {args.omega} K {args.n_bins} "
          f"| tardiness_weight {args.tardiness_weight} "
          f"| reward {args.reward_mode} | effective batch "
          f"{args.batch_size * args.accum_steps}")

    for step in range(1, args.steps + 1):
        gate_scale_now[0] = gate_scale(step, args.steps, args.gate_schedule,
                                       args.gate_off_frac)
        metrics = train_step_stoch(
            policy, optimizer, sample_batch, train_sim_cfg,
            accum_steps=args.accum_steps, n_starts=args.n_starts,
            tardiness_weight=args.tardiness_weight, grad_clip=args.grad_clip,
            generator=train_gen, common_random_numbers=not args.no_crn,
            reward_mode=args.reward_mode,
        )
        if scheduler is not None:
            scheduler.step()

        if gate_check_every and step % gate_check_every == 0:
            # Mutates nothing; raises if any stacked term stopped being inert when off.
            assert_gate_chain_inert(policy, val, sim_cfg, seed=step)

        if args.log_every and step % args.log_every == 0:
            # softplus_w appears here for variants D/D-prime via the policy_metrics hook.
            extra = "".join(
                f" | {k} {v:.4f}" for k, v in metrics.items()
                if k not in ("loss", "mean_cost", "mean_distance", "mean_tardiness",
                             "mean_violations", "late_rate", "grad_norm")
            )
            if args.gate_schedule != "constant":
                extra += f" | gate_scale {gate_scale_now[0]:.3f}"
            print(f"step {step:5d}/{args.steps} | loss {metrics['loss']:+.4f} "
                  f"| cost {metrics['mean_cost']:.4f} "
                  f"| dist {metrics['mean_distance']:.4f} "
                  f"| tard {metrics['mean_tardiness']:.4f} "
                  f"| viol {metrics['mean_violations']:.3f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| {step / (time.time() - t0):.1f} it/s{extra}")

        if step % args.eval_every == 0 or step == args.steps:
            id_stats = validate(sim_cfg)
            line = (f"  [eval] step {step:5d} | val_cost {id_stats['cost']:.4f} "
                    f"(dist {id_stats['distance']:.4f} tard {id_stats['tardiness']:.4f} "
                    f"late {id_stats['late_count']:.2f})")
            ood_stats = None
            if ood_cfg is not None:
                ood_stats = validate(ood_cfg)
                line += (f" | OOD@{args.ood_beta_var} {ood_stats['cost']:.4f} "
                         f"(tard {ood_stats['tardiness']:.4f})")

            tag = ""
            ckpt = {"model_state": policy.state_dict(), "model_config": model_config,
                    "sim_config": asdict(sim_cfg),
                    "tardiness_weight": args.tardiness_weight,
                    "n_customers": args.n_customers, "capacity": args.capacity,
                    "shift_end": args.shift_end, "tw_half_width": args.tw_half_width,
                    "service_time": args.service_time,
                    "reward_mode": args.reward_mode, "variant": args.variant,
                    "beta_var_range": args.beta_var_range,
                    "gate_lr": args.gate_lr,
                    "gate_signal": args.gate_signal,
                    "gate_location": args.gate_location,
                    "gate_schedule": args.gate_schedule,
                    "gate_off_frac": args.gate_off_frac,
                    "gate_fixed": args.gate_fixed,
                    "risk_signal": args.risk_signal,
                    "days_per_instance": k_days,
                    "policy_metrics": (policy.policy_metrics()
                                       if hasattr(policy, "policy_metrics") else {}),
                    "val_scenario_seeds": scenario_seeds,
                    "val_augment": args.val_augment,
                    "val_stats": id_stats, "ood_stats": ood_stats,
                                        "val_cost": id_stats["cost"], "step": init_steps + step,
                    "init_from": args.init_from, "init_steps": init_steps,
                    "finetune_steps": step}
            # Always keep the LATEST weights too: with a noisy 256-instance validation,
            # "best val" can select an early checkpoint by luck. A fixed-budget final
            # checkpoint is the selection-free alternative for the seed study.
            torch.save(ckpt, last_path)
            if args.save_every and step % args.save_every == 0:
                # A permanent snapshot of THIS budget: never overwritten, so the run can
                # be re-evaluated later at 5k, 10k, ... as if it had been stopped there.
                torch.save(
                    ckpt,
                    out_path.with_name(f"{out_path.stem}_step{step}{out_path.suffix}"),
                )
            if id_stats["cost"] < best_val:
                best_val = id_stats["cost"]
                torch.save(ckpt, out_path)
                tag = "  <- saved best"
            print(line + f" | best {best_val:.4f}{tag}")

    print(f"done. best val_cost {best_val:.4f}. checkpoint: {out_path} (latest: {last_path})")


if __name__ == "__main__":
    main()