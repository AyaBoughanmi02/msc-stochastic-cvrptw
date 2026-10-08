"""Batched stochastic, time-dependent CVRPTW environment (variants B/C/D).

Subclass of CVRPTWEnv that swaps the travel-time seam for an Edge-Bank lookup. The
state machine, the clock semantics (reset to 0 on every depot return), the capacity
logic and the mask conventions are inherited UNCHANGED -- only four seam hooks are
overridden. That is the payoff of the Stage-2 seam.

PLAN ON MU, EXECUTE ON A SAMPLE
-------------------------------
Hard masking still forbids everything that is certain: visited, over-capacity,
depot-at-depot, and -- using the PLANNED mean travel time `mu` at the current time bin
-- window-closed and can't-serve-and-return. What it can no longer forbid is a late
arrival, because the clock advances by a SAMPLED travel time that may exceed `mu`. So
variant A's hard time-window constraint softens into a TARDINESS PENALTY in the cost:

    cost = tour_distance + tardiness_weight * (late_count + total_tardiness)

`late_count` counts arrivals after `latest`, `total_tardiness` sums how late they were
(both include the depot, whose window is [0, shift_end] -- so a late return is counted
as a shift overrun by the same mechanism). The OBJECTIVE stays distance; the penalty is
the only thing the uncertainty adds. `tardiness_weight` is a CALIBRATION TARGET, not a
constant: it is swept in a separate diagnostic to find the regime where violations cost
enough to matter but do not dominate the distance term.

Two documented simplifications:
  - The return-leg estimate `_travel_to_depot` is read at the CURRENT bin, not at the
    (unknown) bin the vehicle would actually arrive in. Making it exact would need a
    fixed-point solve per candidate node per step.
  - Masking uses `mu` only; it is risk-neutral by construction. That is deliberate --
    it is what leaves the variance gate as the SOLE difference between variants B and D.

MEMORY / POMO
-------------
POMO replicates each instance `n_starts` times (`repeat_interleave`), so the env batch is
B*S. The Edge Bank is NOT replicated: it stays at instance granularity (B, N, N, K, 2) and
`instance_idx: (B*S,)` maps each trajectory to its instance. At N=50, S=50 that is the
difference between ~0.5 MB and ~770 MB of resident bank.

COMMON RANDOM NUMBERS
---------------------
If `noise` is given -- a pre-drawn standard-normal field of shape (B, N, N, K), i.e. one
realised "day" per instance -- every trajectory of that instance faces the same travel
times, so POMO's shared baseline compares DECISIONS rather than luck. If `noise` is None,
fresh noise is drawn per traversal from `generator` (higher gradient variance; kept as an
option so the choice is testable rather than assumed).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.data.simulator import (
    _MU_EPS,
    MU,
    SIGMA2,
    SimulatorConfig,
    apply_noise,
    bin_of,
    draw_noise,
    gather_edge,
    gather_from,
    gather_to,
)
from src.envs.cvrptw_env import CVRPTWEnv
from src.utils.masking import _TW_EPS, tour_length  # same tolerance the mask uses

_SQRT_2PI = math.sqrt(2.0 * math.pi)
# Floor inside sqrt() so a zero-variance route (beta_var = 0) gives sd -> 0 without NaN.
_VAR_EPS = 1e-12


def _std_normal_pdf(z: Tensor) -> Tensor:
    return torch.exp(-0.5 * z * z) / _SQRT_2PI


def _std_normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z * (0.5 ** 0.5)))


def _lognormal_lateness(mu: Tensor, sigma2: Tensor, slack: Tensor) -> tuple[Tensor, Tensor]:
    """E[(X - slack)^+] and P(X > slack) for X lognormal with mean mu, variance sigma2.

    mu, sigma2, slack: (M,) each. slack = latest - departure clock (may be <= 0).
    Uses the moment-matched parameters of `apply_noise`, so it is the exact expectation of
    what reward_mode='sampled' would charge for this arc:
        s^2 = log1p(sigma2 / mu^2),  m = log(mu) - s^2 / 2
        E[(X - K)^+] = mu * Phi(d1) - K * Phi(d2),  P(X > K) = Phi(d2)
        d1 = (m - log K + s^2) / s,  d2 = d1 - s           (K > 0)
    K <= 0: lateness is certain, E = mu - K, P = 1. sigma2 == 0 (or mu == 0, the idle
    self-loop): deterministic, E = relu(mu - K), P = 1[mu > K].
    """
    mu_safe = mu.clamp_min(_MU_EPS)
    s2 = torch.log1p(sigma2.clamp_min(0.0) / mu_safe.square())            # (M,)
    s = s2.sqrt()
    k_pos = slack.clamp_min(_MU_EPS)
    m = mu_safe.log() - 0.5 * s2
    s_safe = s.clamp_min(_VAR_EPS)
    d1 = (m - k_pos.log() + s2) / s_safe
    d2 = d1 - s_safe
    exp_pos = mu * _std_normal_cdf(d1) - slack * _std_normal_cdf(d2)      # K > 0, s > 0
    p_pos = _std_normal_cdf(d2)

    degenerate = (s <= _VAR_EPS) | (mu <= _MU_EPS)
    exp_det = (mu - slack).clamp_min(0.0)
    p_det = (mu > slack).to(mu.dtype)
    certain = slack <= 0.0
    exp_late = torch.where(degenerate, exp_det, torch.where(certain, mu - slack, exp_pos))
    p_late = torch.where(degenerate, p_det, torch.where(certain, torch.ones_like(mu), p_pos))
    return exp_late.clamp_min(0.0), p_late


class CVRPTWStochEnv(CVRPTWEnv):
    """CVRPTW state machine with stochastic, time-dependent travel times."""

    def __init__(
        self,
        coords: Tensor,                       # (B*S, N, 2)
        demand: Tensor,                       # (B*S, N)
        capacity: Tensor | float,             # (B*S,) or scalar
        time_windows: Tensor,                 # (B*S, N, 2)
        service_time: Tensor,                 # (B*S, N)
        shift_end: float,
        bank: Tensor,                         # (B, N, N, K, 2) instance-level Edge Bank
        sim_cfg: SimulatorConfig,
        instance_idx: Tensor | None = None,   # (B*S,) long into bank; None = identity
        noise: Tensor | None = None,          # (B, N, N, K) pre-drawn CRN field
        generator: torch.Generator | None = None,
        tardiness_weight: float = 1.0,
        reward_mode: str = "sampled",         # 'sampled' (legacy) or 'analytic' (training)
    ) -> None:
        b, n, _ = coords.shape
        dev = coords.device

        if reward_mode not in ("sampled", "analytic", "hybrid"):
            raise ValueError(
                f"unknown reward_mode {reward_mode!r}; use 'sampled' or 'analytic' or 'hybrid'"
            )

        if bank.dim() != 5 or bank.size(-1) != 2:
            raise ValueError(f"bank must be (B, N, N, K, 2), got {tuple(bank.shape)}")
        if bank.size(1) != n or bank.size(2) != n:
            raise ValueError(
                f"bank has {bank.size(1)}x{bank.size(2)} nodes but coords have {n}"
            )
        if bank.size(3) != sim_cfg.n_bins:
            raise ValueError(
                f"bank has {bank.size(3)} time bins but sim_cfg.n_bins={sim_cfg.n_bins}; "
                f"bin_of would index the wrong slice"
            )
        if bank.device != dev:
            raise ValueError(f"bank is on {bank.device} but coords are on {dev}")
        if not math.isclose(sim_cfg.shift_end, float(shift_end), rel_tol=1e-9):
            raise ValueError(
                f"sim_cfg.shift_end={sim_cfg.shift_end} != env shift_end={shift_end}; "
                f"time bins would not line up with the clock"
            )
        if tardiness_weight < 0.0:
            raise ValueError(f"tardiness_weight must be >= 0, got {tardiness_weight}")

        if instance_idx is None:
            if bank.size(0) != b:
                raise ValueError(
                    f"without instance_idx the bank must have one row per trajectory: "
                    f"{bank.size(0)} vs {b}"
                )
            instance_idx = torch.arange(b, device=dev)                 # (B*S,)
        else:
            if instance_idx.shape != (b,):
                raise ValueError(
                    f"instance_idx must be ({b},), got {tuple(instance_idx.shape)}"
                )
            if instance_idx.dtype != torch.long:
                raise ValueError(f"instance_idx must be int64, got {instance_idx.dtype}")
            # Range-checked ONCE here (a device sync); the per-step gathers do not.
            if int(instance_idx.min()) < 0 or int(instance_idx.max()) >= bank.size(0):
                raise ValueError(
                    f"instance_idx out of range for a bank of {bank.size(0)} instances"
                )
        if noise is not None and noise.shape != bank.shape[:-1]:
            raise ValueError(
                f"noise must be {tuple(bank.shape[:-1])} (the bank without its moment "
                f"axis), got {tuple(noise.shape)}"
            )

        # Set before super().__init__: it calls _reset_state(), which we extend.
        self.bank = bank
        self.sim_cfg = sim_cfg
        self.instance_idx = instance_idx
        self.noise = noise
        self.generator = generator
        self.tardiness_weight = float(tardiness_weight)
        self.reward_mode = reward_mode

        super().__init__(coords, demand, capacity, time_windows, service_time, shift_end)

        # Declared for clarity; set in _reset_state.
        self.tardiness: Tensor    # (B*S,) lateness (realised, or expected in analytic mode)
        self.late_count: Tensor   # (B*S,) violations (count, or expected count)
        self.time_var: Tensor     # (B*S,) variance of the arrival clock (analytic mode)

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        super()._reset_state()                                   # visited/capacity/clock
        dev, dtype = self.coords.device, self.coords.dtype
        b = self.batch_size
        self.tardiness = torch.zeros(b, device=dev, dtype=dtype)      # (B*S,)
        self.late_count = torch.zeros(b, device=dev, dtype=dtype)     # (B*S,)
        # Arrival-time variance, accumulated along the route and reset at the depot
        # exactly like the clock. Only used by reward_mode='analytic'.
        self.time_var = torch.zeros(b, device=dev, dtype=dtype)        # (B*S,)
        self._arc_var = torch.zeros(b, device=dev, dtype=dtype)        # (B*S,) scratch
        self._depot_idx = torch.zeros(b, dtype=torch.long, device=dev)  # (B*S,) reused

    def current_bin(self) -> Tensor:
        """Time bin of each trajectory's clock. (B*S,) long, clamped to [0, K-1]."""
        return bin_of(self.time, self.sim_cfg)

    # ------------------------------------------------------------- seam hooks

    def _travel_from_current(self) -> Tensor:
        """PLANNED (mean) travel time from the current node to every node. (B*S, N)."""
        moments = gather_from(
            self.bank, self.current_node, self.current_bin(), self.instance_idx
        )                                                        # (B*S, N, 2)
        return moments[..., MU]                                  # (B*S, N)

    def _travel_to_depot(self) -> Tensor:
        """PLANNED (mean) travel time from every node to the depot. (B*S, N).

        Read at the CURRENT bin (documented simplification; see the module docstring).
        """
        moments = gather_to(
            self.bank, self._depot_idx, self.current_bin(), self.instance_idx
        )                                                        # (B*S, N, 2)
        return moments[..., MU]                                  # (B*S, N)

    def _travel_time(self, action: Tensor) -> Tensor:
        """EXECUTED travel time on the traversed edge: one lognormal sample. (B*S,).

        Drawn at the DEPARTURE bin, with mean `mu` and variance `sigma^2` from the bank.
        Degenerate edges (sigma^2 = 0, i.e. variant A; or the mu = 0 self-loop a finished
        trajectory idles on) return `mu` exactly, so nothing drifts.
        """
        bin_idx = self.current_bin()                             # (B*S,)
        moments = gather_edge(
            self.bank, self.current_node, action, bin_idx, self.instance_idx
        )                                                        # (B*S, 2)
        mu, sigma2 = moments[:, MU], moments[:, SIGMA2]          # (B*S,) each
        self._arc_var = sigma2          # for the analytic charge
        self._arc_mu = mu  # for the hybrid charge

        if self.reward_mode == "analytic":
            # The clock advances on the MEAN, so the route is a deterministic function of
            # the actions. Uncertainty is priced in _on_arrival instead of rolled for.
            return mu                                            # (B*S,)
        if self.noise is not None:
            # One realised day per instance: shared by all of that instance's POMO starts.
            z = self.noise[self.instance_idx, self.current_node, action, bin_idx]  # (B*S,)
        else:
            z = draw_noise(
                (self.batch_size,), device=mu.device, dtype=mu.dtype,
                generator=self.generator,
            )                                                    # (B*S,)
        return apply_noise(mu, sigma2, z)                        # (B*S,)

    def _on_arrival(self, action: Tensor, arrival: Tensor) -> None:
        """Accumulate tardiness for this step. action, arrival: (B*S,).

        Lateness is measured against `latest` of the node just arrived at. Waiting only
        helps when EARLY, so `relu(arrival - latest)` is the whole story. The depot's
        window is [0, shift_end], so a late return is recorded here as a shift overrun.
        The tolerance matches the mask's, so a node the mask judged exactly-on-time is
        never counted late.
        """
        idx = torch.arange(self.batch_size, device=self.coords.device)
        latest = self.time_windows[idx, action, 1]               # (B*S,)

        if self.reward_mode == "analytic":
            # Arrival time is approximated as Normal(arrival, time_var + arc_var): the
            # means were accumulated by the clock, the variances are accumulated here
            # (arc travel times are independent, so variances add). The expected lateness
            # of a Normal above a threshold has a closed form, so BOTH terms below are
            # deterministic functions of the route -- zero sampling noise in the reward.
            var = self.time_var + self._arc_var                  # (B*S,)
            sd = (var + _VAR_EPS).sqrt()                         # (B*S,)
            excess = arrival - latest                            # (B*S,) <= 0 under the mask
            z = excess / sd
            # E[(A - latest)^+] = sd * phi(z) + excess * Phi(z);  P(A > latest) = Phi(z)
            self.tardiness = (
                self.tardiness + sd * _std_normal_pdf(z) + excess * _std_normal_cdf(z)
            )
            self.late_count = self.late_count + _std_normal_cdf(z)
            self.time_var = var
            return

        if self.reward_mode == "hybrid":
            # The clock is REAL (sampled), but this arc is charged its EXPECTED lateness
            # given the departure clock, in closed form for the lognormal travel time.
            # Same expectation as 'sampled' (unbiased), without this arc's noise.
            exp_late, p_late = _lognormal_lateness(
                self._arc_mu, self._arc_var, latest - self.time
            )  # (B*S,) each
            self.tardiness = self.tardiness + exp_late
            self.late_count = self.late_count + p_late
            return

        late = (arrival - latest).clamp_min(0.0)  # (B*S,)
        self.tardiness = self.tardiness + late

        self.late_count = self.late_count + (late > _TW_EPS).to(self.tardiness.dtype)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one step, then reset the arrival-time variance at the depot.

        The clock resets to 0 on every depot return (a fresh vehicle/shift), so the
        uncertainty accumulated along the finished route must reset with it.
        """
        mask, done = super().step(action)
        if self.reward_mode == "analytic":
            self.time_var = torch.where(
                action == 0, torch.zeros_like(self.time_var), self.time_var
            )
        return mask, done

    # ------------------------------------------------------------------ cost

    def cost_components(self, actions: Tensor) -> dict[str, Tensor]:
        """Break the episode cost into its parts. actions: (B*S, T) -> (B*S,) tensors.

        STATEFUL: the tardiness terms come from the rollout that produced `actions`, so
        this is only meaningful at the end of that rollout.

        In reward_mode='analytic' the tardiness terms are EXPECTATIONS, so 'late_count' is
        a fractional expected number of violations rather than an integer count.
        """
        distance = tour_length(self.coords, actions)             # (B*S,)
        penalty = self.tardiness_weight * (self.late_count + self.tardiness)
        return {
            "distance": distance,
            "late_count": self.late_count,
            "tardiness": self.tardiness,
            "penalty": penalty,
            "total": distance + penalty,
        }

    def cost(self, actions: Tensor) -> Tensor:
        """Penalised episode cost (the POMO training signal). (B*S,).

        Overrides the pure tour length so the POMO loss and shared baseline need no
        change: `cost` is still "lower is better", it just now prices lateness.
        """
        return self.cost_components(actions)["total"]

    def normalized_time(self) -> Tensor:
        """Clock / shift_end for the decoder context. (B*S, 1).

        Unlike variant A this can EXCEED 1: a sampled travel time can push the clock past
        the shift. It is deliberately not clamped -- being over the shift is exactly the
        state the policy should be able to see.
        """
        return (self.time / self.shift_end).unsqueeze(1)