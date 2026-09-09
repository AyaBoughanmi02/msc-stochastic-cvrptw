"""Stochastic-CVRPTW calibration diagnostics.

Two questions, both of which have to be answered with numbers before variants B/D are
trained.

1. ARE THE INSTANCES STILL TIME-BINDING once travel times ripple? Stage 2 established
   time_only ~0.275 vs capacity_only ~0.024 with travel == distance. Masking still plans
   on the mean, but the mean now moves with the clock, so the binding numbers have to be
   re-measured rather than assumed. Same decomposition as `calibration_stats`, run on the
   stochastic env.

2. WHAT SHOULD `tardiness_weight` BE? It is a calibration target, not a constant: too
   small and lateness is invisible to the gradient (the problem collapses back to variant
   A); too large and distance stops mattering (the policy degenerates to short, timid
   routes). The dial we can actually read is the PENALTY SHARE,

       share(w) = w * sum(late_count + tardiness) / [ sum(distance) + w * sum(...) ],

   the fraction of total cost that lateness accounts for. Because the rollout itself does
   not depend on `w`, the whole sweep is arithmetic over ONE rollout -- no retraining, no
   re-simulation.

Interpretation of `w`: the penalty for one late arrival is `w * (1 + lateness)`, so `w` is
denominated in DISTANCE UNITS -- it is how much extra travel the policy should be willing
to accept to avoid one violation. `mean_distance_per_customer` is reported so that number
is readable.

CAVEAT, to be stated in the write-up: the Level-1 sweep uses a greedy-feasible proxy that
makes no attempt to avoid lateness, so it OVERSTATES the tardiness a trained policy will
incur. The weight it recommends is therefore a lower bound; re-run `policy_tardiness_stats`
(Level 2) on a trained checkpoint to confirm the share has not collapsed.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SimulatorConfig, build_edge_bank, draw_noise
from src.envs.cvrptw_diagnostics import decompose_infeasibility
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv

# Spread over two orders of magnitude around 1.0 (= "one violation is worth one unit of
# detour"), which is the scale of a whole CVRPTW20 tour divided by ~9 customers.
DEFAULT_WEIGHTS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# A penalty worth 10-30% of total cost: visible to the gradient, not dominant.
DEFAULT_TARGET_SHARE = 0.2


class TardinessProfile(NamedTuple):
    """What one candidate `tardiness_weight` implies for a fixed set of rollouts."""

    weight: float
    mean_penalty: float    # mean over trajectories of w * (late_count + tardiness)
    mean_cost: float       # mean distance + mean penalty
    penalty_share: float   # aggregate penalty / aggregate cost, in [0, 1)


def tardiness_weight_sweep(
    components: dict[str, Tensor],
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> list[TardinessProfile]:
    """Price a fixed set of rollouts under several `tardiness_weight` values.

    components: the dict returned by CVRPTWStochEnv.cost_components (needs 'distance',
                'tardiness', 'late_count'; each (M,) over trajectories).
    Pure arithmetic -- no rollout, no simulation, no RNG.
    """
    missing = {"distance", "tardiness", "late_count"} - set(components)
    if missing:
        raise ValueError(f"components is missing {sorted(missing)}")
    if len(weights) == 0:
        raise ValueError("weights must be non-empty")
    if any(w < 0.0 for w in weights):
        raise ValueError(f"tardiness weights must be >= 0, got {list(weights)}")

    distance = components["distance"]                                  # (M,)
    violations = components["late_count"] + components["tardiness"]    # (M,)
    total_distance = float(distance.sum())
    total_violation = float(violations.sum())

    profiles: list[TardinessProfile] = []
    for w in weights:
        penalty = w * total_violation
        total = total_distance + penalty
        profiles.append(
            TardinessProfile(
                weight=float(w),
                mean_penalty=w * float(violations.mean()),
                mean_cost=float(distance.mean()) + w * float(violations.mean()),
                penalty_share=penalty / total if total > 0.0 else 0.0,
            )
        )
    return profiles


def recommend_tardiness_weight(
    profiles: Sequence[TardinessProfile],
    target_share: float = DEFAULT_TARGET_SHARE,
) -> float:
    """The swept weight whose penalty share lands closest to `target_share`."""
    if len(profiles) == 0:
        raise ValueError("profiles must be non-empty")
    if not 0.0 < target_share < 1.0:
        raise ValueError(f"target_share must be in (0, 1), got {target_share}")
    return min(profiles, key=lambda p: abs(p.penalty_share - target_share)).weight


def weight_grid_saturated(
    profiles: Sequence[TardinessProfile],
    target_share: float = DEFAULT_TARGET_SHARE,
) -> bool:
    """True when the recommendation is pinned to an END of the grid and still misses target.

    A recommendation at the grid boundary is not a recommendation -- it means the target
    share lies outside the swept range and the sweep must be widened. Reporting it as a
    number without this flag is how a badly calibrated weight gets locked in silently.
    """
    by_weight = sorted(profiles, key=lambda p: p.weight)
    lo, hi = by_weight[0], by_weight[-1]
    return hi.penalty_share < target_share or lo.penalty_share > target_share


def _build_env(
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    seed: int,
    common_random_numbers: bool,
) -> CVRPTWStochEnv:
    """One instance-aligned stochastic env (no POMO replication) with a fixed day."""
    dev = data.coords.device
    generator = torch.Generator(device=dev).manual_seed(seed)
    bank = build_edge_bank(data.coords, sim_cfg)                       # (B, N, N, K, 2)
    noise = None
    if common_random_numbers:
        noise = draw_noise(
            bank.shape[:-1], device=dev, dtype=data.coords.dtype, generator=generator
        )                                                              # (B, N, N, K)
    return CVRPTWStochEnv(
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
        bank=bank, sim_cfg=sim_cfg, noise=noise, generator=generator,
        tardiness_weight=0.0,   # the sweep reprices; keep the env's own weight neutral
    )


def stoch_calibration_stats(
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    target_share: float = DEFAULT_TARGET_SHARE,
    seed: int = 0,
    max_steps: int | None = None,
    common_random_numbers: bool = True,
) -> dict[str, object]:
    """Level 1: greedy-feasible proxy rollout under stochastic travel times.

    Reports the time-vs-capacity binding decomposition (does the ripple change what
    Stage 2 measured?) and the tardiness the proxy incurs, then prices that tardiness
    across `weights`.
    """
    env = _build_env(data, sim_cfg, seed, common_random_numbers)
    n = env.num_nodes
    steps = 3 * n if max_steps is None else max_steps
    cust = slice(1, None)

    mask = env.reset()
    tot_uv = time_only = cap_only = both = 0
    actions: list[Tensor] = []
    for _ in range(steps):
        reasons = decompose_infeasibility(env)          # reads the same seam as the mask
        unvisited = ~reasons.visited[:, cust]                          # (B, N-1)
        t_block = (reasons.window_closed | reasons.cant_return)[:, cust] & unvisited
        c_block = reasons.over_capacity[:, cust] & unvisited
        tot_uv += int(unvisited.sum())
        time_only += int((t_block & ~c_block).sum())
        cap_only += int((c_block & ~t_block).sum())
        both += int((t_block & c_block).sum())

        # Greedy proxy: nearest feasible node in PLANNED travel time.
        step_cost = env._travel_from_current().masked_fill(mask, float("inf"))  # (B, N)
        action = step_cost.argmin(dim=1)                               # (B,)
        actions.append(action)
        mask, done = env.step(action)
        if bool(done.all()):
            break

    if not bool(env.visited[:, 1:].all()):              # crash-not-silent
        raise RuntimeError(
            "the greedy-feasible proxy failed to serve every customer within the step "
            "budget; the instance calibration or the ripple amplitude is wrong"
        )

    parts = env.cost_components(torch.stack(actions, dim=1))           # (B,) each
    profiles = tardiness_weight_sweep(parts, weights)
    denom = max(tot_uv, 1)
    return {
        "time_only_rate": time_only / denom,
        "capacity_only_rate": cap_only / denom,
        "both_rate": both / denom,
        "mean_distance": float(parts["distance"].mean()),
        "mean_distance_per_customer": float(parts["distance"].mean()) / (n - 1),
        "mean_tardiness": float(parts["tardiness"].mean()),
        "mean_late_count": float(parts["late_count"].mean()),
        "late_rate": float((parts["late_count"] > 0).to(torch.float32).mean()),
        "profiles": profiles,
        "recommended_weight": recommend_tardiness_weight(profiles, target_share),
        "weight_grid_saturated": weight_grid_saturated(profiles, target_share),
    }


@torch.no_grad()
def measure_pointer_saturation(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    threshold: float = 8.0,
    tight_slack: float = 0.25,
    n_starts: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """How much of the pointer's signal is being flattened by the tanh clip.

    The edge term and the variance gate are added BEFORE the tanh clip, so a candidate
    whose pre-tanh logit u is already deep in the flat region (|u| > threshold) barely
    moves when the gate pushes on it. This measures whether that is actually happening,
    and specifically whether it happens where it would matter -- next to a tight window.

    A candidate counts as NEAR A TIGHT WINDOW when its planned arrival leaves less than
    `tight_slack` before the window closes: latest_j - (clock + mu_cj) <= tight_slack.
    Masked candidates are excluded throughout; their logits are -inf and would make the
    ratio meaningless. The depot is excluded from the tight-window slice only (its window
    spans the whole shift, so it is never tight).

    If `saturated_frac_near_tight_window` is negligible, the pre-tanh placement costs
    nothing and `edge_bias_mode='post_tanh'` need never be tried. If it is high, that is
    evidence the flag matters -- measured, before spending a training run on it.
    """
    from src.train.pomo import build_starts
    from src.train.pomo_stoch import assert_ood_eval_mode, build_stoch_env

    policy.eval()
    assert_ood_eval_mode(policy)
    b, n, _ = data.coords.shape
    starts_n = (n - 1) if n_starts is None else n_starts
    env = build_stoch_env(
        data, sim_cfg, starts_n, tardiness_weight=0.0,
        generator=torch.Generator(device=data.coords.device).manual_seed(seed),
    )

    enc = policy._encoder_inputs(env)
    node_flat, global_emb = policy.encoder(
        enc.x, enc.edge_index, enc.edge_attr, enc.batch
    )
    node_emb = node_flat.view(env.batch_size, n, -1)                  # (B*S, N, H)
    cache = policy.decoder.precompute(node_emb)
    row = torch.arange(env.batch_size, device=data.coords.device)

    n_feasible = n_saturated = n_tight = n_saturated_tight = 0
    abs_u_sum = 0.0
    decoder = policy.decoder
    was_recording = decoder.record_diagnostics
    decoder.record_diagnostics = True
    try:
        mask = env.reset(build_starts(b, starts_n, data.coords.device))
        for _ in range(2 * n):
            edge_feat = policy._decoder_edge_feat(env)
            log_p = decoder(
                cache, global_emb, node_emb[row, env.current_node],
                env.normalized_remaining(), mask,
                context_extra=env.normalized_time(),
                edge_feat=edge_feat,
                gate_penalty=policy._gate_penalty(env, edge_feat),
            )                                                          # (B*S, N)
            u = decoder._last_u                                        # (B*S, N)
            feasible = ~mask                                           # (B*S, N)
            saturated = (u.abs() > threshold) & feasible

            # Slack until the window closes, on PLANNED travel (no RNG consumed).
            slack = env.time_windows[..., 1] - (
                env.time.unsqueeze(1) + env._travel_from_current()
            )                                                          # (B*S, N)
            tight = feasible & (slack <= tight_slack)
            tight[:, 0] = False                                        # the depot is never tight

            n_feasible += int(feasible.sum())
            n_saturated += int(saturated.sum())
            n_tight += int(tight.sum())
            n_saturated_tight += int((saturated & tight).sum())
            abs_u_sum += float(u.abs()[feasible].sum())

            mask, done = env.step(log_p.argmax(dim=-1))
            if bool(done.all()):
                break
    finally:
        decoder.record_diagnostics = was_recording
        decoder._last_u = None

    return {
        "saturated_frac": n_saturated / max(n_feasible, 1),
        "saturated_frac_near_tight_window": n_saturated_tight / max(n_tight, 1),
        "frac_near_tight": n_tight / max(n_feasible, 1),
        "mean_abs_u": abs_u_sum / max(n_feasible, 1),
        "n_feasible": float(n_feasible),
        "n_tight": float(n_tight),
    }


@torch.no_grad()
def policy_tardiness_stats(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    n_starts: int | None = None,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    target_share: float = DEFAULT_TARGET_SHARE,
    seed: int = 0,
) -> dict[str, object]:
    """Level 2: the same pricing under a TRAINED policy's own routing.

    A trained policy avoids lateness that the greedy proxy walks into, so the share here
    will be lower for the same weight. Run this on a checkpoint to check the weight chosen
    at Level 1 has not made the penalty invisible.
    """
    from src.train.pomo_stoch import multistart_greedy_stoch  # local: avoids an import cycle

    out = multistart_greedy_stoch(
        policy, data, sim_cfg, n_starts=n_starts, tardiness_weight=0.0,
        generator=torch.Generator(device=data.coords.device).manual_seed(seed),
    )
    components = {
        "distance": out.distance.reshape(-1),      # (B*S,)
        "tardiness": out.tardiness.reshape(-1),    # (B*S,)
        "late_count": out.late_count.reshape(-1),  # (B*S,)
    }
    profiles = tardiness_weight_sweep(components, weights)
    n_customers = data.coords.size(1) - 1
    return {
        "mean_distance": float(components["distance"].mean()),
        "mean_distance_per_customer": float(components["distance"].mean()) / n_customers,
        "mean_tardiness": float(components["tardiness"].mean()),
        "mean_late_count": float(components["late_count"].mean()),
        "late_rate": float((components["late_count"] > 0).to(torch.float32).mean()),
        "profiles": profiles,
        "recommended_weight": recommend_tardiness_weight(profiles, target_share),
        "weight_grid_saturated": weight_grid_saturated(profiles, target_share),
    }