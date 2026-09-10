"""POMO training for stochastic, time-dependent CVRPTW (variants B/C/D).

Mirrors src/train/pomo_cvrptw.py. Three things change, all of them in how the batch is
assembled -- the loss, the shared baseline and the policy interface are untouched:

  1. The env is a CVRPTWStochEnv, so `cost` is the PENALISED cost
     (distance + tardiness_weight * (late_count + tardiness)). `pomo_loss` is reused
     verbatim because the sign convention is unchanged: lower is still better.
  2. An Edge Bank is built once per batch at INSTANCE granularity and shared across the
     n_starts replicas via `instance_idx`, instead of being replicated with the coords.
     At n=50, S=50 that is ~0.5 MB resident instead of ~770 MB.
  3. A standard-normal noise field of shape (B, N, N, K) is drawn once per batch --
     one realised "day" per instance, shared by all of that instance's starts (common
     random numbers). Every POMO trajectory then faces the same travel times, so the
     shared baseline compares DECISIONS rather than luck. Set
     `common_random_numbers=False` to draw fresh noise per traversal instead; that is
     strictly higher-variance and is kept only so the choice stays testable.

The out-of-distribution experiment is plumbed by passing a DIFFERENT `sim_cfg` to the
evaluation helper than the one used for training (typically a larger `beta_var`, or a
different reliability field). Nothing else needs to change.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SimulatorConfig, build_edge_bank, draw_noise
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.train.pomo import build_starts, pomo_loss


class StochRollout(NamedTuple):
    """Per-trajectory outcome of a multi-start rollout. All (B, S)."""

    cost: Tensor        # penalised cost -- the POMO training signal
    sum_logp: Tensor    # summed log-prob of the sampled actions (has grad)
    distance: Tensor    # tour length only
    tardiness: Tensor   # summed lateness
    late_count: Tensor  # number of late arrivals


def build_stoch_env(
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    n_starts: int,
    tardiness_weight: float,
    generator: torch.Generator | None = None,
    common_random_numbers: bool = True,
    noise: Tensor | None = None,      # (B, N, N, K) pre-drawn field; None = draw one
    reward_mode: str = "sampled",
) -> CVRPTWStochEnv:
    """Assemble a stochastic env with each instance replicated CONTIGUOUSLY n_starts times.

    The geometric tensors are replicated (the policy needs one row per trajectory); the
    Edge Bank is NOT, and neither is the noise field -- both stay at instance granularity
    and are addressed through `instance_idx`.

    `noise` lets the caller supply the realised day instead of drawing one. Evaluation
    uses it to give all 8 augmentations of an instance the SAME day, so "best over
    augmentations" selects the best route rather than the luckiest weather.
    """
    b, n, _ = data.coords.shape
    dev = data.coords.device

    bank = build_edge_bank(data.coords, sim_cfg)                    # (B, N, N, K, 2)
    # 'analytic' never samples a travel time, so a noise field would be dead weight.
    if reward_mode == "analytic":
        noise = None
    elif noise is None and common_random_numbers:
        # One realised day per instance; every start of that instance shares it.
        noise = draw_noise(
            bank.shape[:-1], device=dev, dtype=data.coords.dtype, generator=generator
        )                                                           # (B, N, N, K)

    instance_idx = torch.arange(b, device=dev).repeat_interleave(n_starts)  # (B*S,)
    return CVRPTWStochEnv(
        coords=data.coords.repeat_interleave(n_starts, dim=0),           # (B*S, N, 2)
        demand=data.demand.repeat_interleave(n_starts, dim=0),           # (B*S, N)
        capacity=data.capacity,
        time_windows=data.time_windows.repeat_interleave(n_starts, dim=0),  # (B*S, N, 2)
        service_time=data.service_time.repeat_interleave(n_starts, dim=0),  # (B*S, N)
        shift_end=data.shift_end,
        bank=bank,
        sim_cfg=sim_cfg,
        instance_idx=instance_idx,
        noise=noise,
        generator=generator,
        tardiness_weight=tardiness_weight,
        reward_mode=reward_mode,
    )


def _resolve_n_starts(data: CVRPTWData, n_starts: int | None) -> int:
    """POMO uses every customer as a start (Kwon et al. 2020); validate an override."""
    n_customers = data.coords.size(1) - 1
    if n_starts is None:
        return n_customers
    if not 1 <= n_starts <= n_customers:
        raise ValueError(f"n_starts must be in [1, {n_customers}], got {n_starts}")
    return n_starts


def assert_ood_eval_mode(policy) -> None:
    """Raise unless the policy (and every BatchNorm in it) is in eval mode.

    The encoder normalises its edge embedding with BatchNorm (`bn_edge`). Under an OOD
    variance shift the input `sigma_bar^2` scales by beta_ood/beta_train, and we WANT that
    shift to reach the network: in eval mode BN uses frozen running statistics, so it does.
    In train mode BN would re-estimate its statistics on the shifted batch and partially
    CANCEL the very shift the experiment is measuring -- silently, with no error and a
    plausible-looking number at the end. Hence an assertion, not a comment.
    """
    if policy.training:
        raise RuntimeError(
            "policy is in train mode during greedy decoding; call policy.eval() first. "
            "Under an OOD variance shift, BatchNorm in train mode re-estimates its "
            "statistics on the shifted batch and cancels part of the shift."
        )
    batchnorms = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    offenders = [
        name for name, module in policy.named_modules()
        if isinstance(module, batchnorms) and module.training
    ]
    if offenders:
        raise RuntimeError(
            f"these BatchNorm modules are still in train mode: {offenders}; they would "
            f"re-estimate running statistics on the evaluation distribution"
        )


def pomo_rollout_stoch(
    policy,                                   # CVRPTWPolicy (duck-typed; see module docs)
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    n_starts: int | None = None,
    tardiness_weight: float = 1.0,
    decode_type: str = "sampling",
    generator: torch.Generator | None = None,
    common_random_numbers: bool = True,
    reward_mode: str = "sampled",
) -> StochRollout:
    """Run N multi-start trajectories per instance under stochastic travel times.

    The policy is unchanged from variant A: CVRPTWStochEnv is a CVRPTWEnv, so
    `build_graph_tw`, `normalized_time()` and the rollout loop all work as-is.
    """
    if decode_type == "greedy":
        # Greedy decoding is always evaluation; refuse to run it under train-mode BN.
        assert_ood_eval_mode(policy)
    b = data.coords.size(0)
    s = _resolve_n_starts(data, n_starts)

    env = build_stoch_env(
        data, sim_cfg, s, tardiness_weight,
        generator=generator, common_random_numbers=common_random_numbers,
        reward_mode=reward_mode,
    )
    starts = build_starts(b, s, data.coords.device)                 # (B*S,)
    actions, sum_logp, cost = policy(env, decode_type=decode_type, force_first=starts)
    parts = env.cost_components(actions)                            # (B*S,) each

    if not torch.equal(cost, parts["total"]):                       # crash-not-silent
        raise RuntimeError(
            "policy-returned cost != env.cost_components()['total']; the policy is not "
            "reading the penalised cost from this env"
        )
    return StochRollout(
        cost=cost.reshape(b, s),
        sum_logp=sum_logp.reshape(b, s),
        distance=parts["distance"].reshape(b, s),
        tardiness=parts["tardiness"].reshape(b, s),
        late_count=parts["late_count"].reshape(b, s),
    )


def train_step_stoch(
    policy,
    optimizer: torch.optim.Optimizer,
    sample_batch: Callable[[], CVRPTWData],
    sim_cfg: SimulatorConfig,
    accum_steps: int = 1,
    n_starts: int | None = None,
    tardiness_weight: float = 1.0,
    grad_clip: float | None = None,
    generator: torch.Generator | None = None,
    common_random_numbers: bool = True,
    reward_mode: str = "sampled",
) -> dict[str, float]:
    """One POMO optimizer step over accum_steps stochastic mini-batches.

    Reports the penalised cost AND its parts, because they are not comparable across
    `tardiness_weight` settings: only `mean_distance` is comparable to variant A's
    val_cost, and `mean_tardiness` / `late_rate` are what the weight sweep calibrates.
    """
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    policy.train()
    optimizer.zero_grad()
    # late_rate counts trajectories with ANY violation; under reward_mode='analytic' the
    # violation count is a fractional expectation and is essentially always > 0, so
    # mean_violations is the meaningful figure there.
    totals = {"loss": 0.0, "mean_cost": 0.0, "mean_distance": 0.0,
              "mean_tardiness": 0.0, "mean_violations": 0.0, "late_rate": 0.0}
    for _ in range(accum_steps):
        data = sample_batch()
        out = pomo_rollout_stoch(
            policy, data, sim_cfg, n_starts, tardiness_weight,
            generator=generator, common_random_numbers=common_random_numbers,
            reward_mode=reward_mode,
        )
        loss, mean_cost = pomo_loss(out.cost, out.sum_logp)         # reused unchanged
        (loss / accum_steps).backward()
        totals["loss"] += float(loss.detach())
        totals["mean_cost"] += float(mean_cost.detach())
        totals["mean_distance"] += float(out.distance.mean().detach())
        totals["mean_tardiness"] += float(out.tardiness.mean().detach())
        totals["mean_violations"] += float(out.late_count.mean().detach())
        totals["late_rate"] += float((out.late_count > 0).to(torch.float32).mean())
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    else:
        grad_norm = torch.tensor(float("nan"))
    optimizer.step()
    out_stats = {k: v / accum_steps for k, v in totals.items()}
    out_stats["grad_norm"] = float(grad_norm)
    # Policy-owned scalars (variant D reports softplus(w); B reports nothing). Read AFTER
    # the optimizer step so the logged value is the one the next rollout will use.
    metrics = getattr(policy, "policy_metrics", None)
    if callable(metrics):
        out_stats.update({k: float(v) for k, v in metrics().items()})
    return out_stats


@torch.no_grad()
def multistart_greedy_stoch(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    n_starts: int | None = None,
    tardiness_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> StochRollout:
    """Greedy multi-start rollout under a FIXED noise draw. Returns the full (B, S) outcome.

    Pass a freshly seeded `generator` so that two models being compared face the SAME days
    (a paired comparison). This is also how the OOD experiment is run: train under one
    `sim_cfg` and call this with a shifted `beta_var`.

    Common random numbers are forced ON here. Without them each start would face a
    different day, and "best of N" would partly be a draw for the luckiest weather rather
    than the best route.
    """
    policy.eval()
    return pomo_rollout_stoch(
        policy, data, sim_cfg, n_starts, tardiness_weight,
        decode_type="greedy", generator=generator, common_random_numbers=True,
    )


@torch.no_grad()
def multistart_greedy_cost_stoch(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    n_starts: int | None = None,
    tardiness_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Best penalised cost over N greedy multi-start rollouts per instance. -> (B,).

    CAVEAT for the write-up: taking the min over starts on ONE realised day is optimistic
    -- the selection sees the noise. It is the right analogue of variant A's best-of-N and
    is fine for tracking training, but a headline OOD number should average this over
    several independent days (several `generator` seeds), not report a single draw.
    """
    return multistart_greedy_stoch(
        policy, data, sim_cfg, n_starts, tardiness_weight, generator
    ).cost.min(dim=1).values                                        # (B,)