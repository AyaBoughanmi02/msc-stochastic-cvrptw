"""Variant VGT-R: VGT plus a slack-aware lateness-risk bias in the decoder.

WHY THIS VARIANT EXISTS
-----------------------
Three earlier results fix its design:

  * Raw sigma^2 as an INPUT hurts (variant B): it drifts out of the trained range under a
    variance shift.
  * A decision-time penalty `-lambda * sigma^2` is MIS-AIMED (scripts/gate_components.py):
    it makes routes longer AND later, because it charges a risky road the same whether or
    not a deadline is near.
  * The training-time gain of VGT is reproduced by a CONSTANT penalty (gate controls), so
    it does not come from risk information.

So no tested form lets variance information help a decision. This variant gives the
decoder the quantity the reward actually prices, in the form ICAM (Zhou et al., 2024)
uses for distance: a learnable non-negative strength times a quantity that already
contains everything deciding its relevance.

    u_j = compat_j + w_edge * mu_j  -  lambda * sigma^2_j  -  alpha * p_late_j
                                       (training-time gate)   (THIS variant, kept at test)

    p_late_j = Phi( (time + mu_ij - latest_j) / sqrt(time_var + sigma^2_ij) )

i.e. the probability of reaching candidate j after its window closes if it is visited
next -- exactly the per-arrival term the analytic reward charges (late_count increment in
CVRPTWStochEnv._on_arrival). alpha = softplus(risk_w) >= 0, so more risk can never make a
candidate MORE attractive.

Properties that motivated the form:
  * bounded in [0, 1] (in [0, 0.5] for every candidate the mask allows, since the mask
    forbids a planned late arrival), so a larger variance cannot push it out of range the
    way raw sigma^2 left the trained range;
  * no BatchNorm on this path;
  * it is large ONLY where risk costs something: near a deadline, or after variance has
    accumulated along the route.

The training-time sigma^2 gate of VGT is kept unchanged (and forced to 0 at evaluation
exactly as for M), so the comparison VGT-R vs M isolates the risk bias.

TRAINING vs EVALUATION (documented, not a bug)
---------------------------------------------
In 'analytic' training mode the clock advances on the MEAN and `env.time_var` carries the
variance accumulated since the last depot visit. In 'sampled' evaluation mode the clock is
the REALISED time, past uncertainty has already resolved, and `time_var` stays 0. Both
give "probability of being late at j given what is known now", so the semantics match;
the training feature is somewhat more spread out than the evaluation one.

CAVEAT: p_late is myopic -- it prices the next arrival only, not the knock-on effect of a
delay on later customers.

CONTROLS (`risk_signal`)
------------------------
Trained with the bias and evaluated WITHOUT it (alpha forced to 0), VGT-R is the best
model measured. To tell whether that comes from the variance INFORMATION in p_late, the
same bias can be built from a control instead of the true sigma^2, everything else equal:

  sigma2   the true variance (VGT-R proper)
  const    every road gets the instance's mean sigma^2: keeps slack, the day's variance
           LEVEL and its accumulation along the route; removes WHICH roads are riskier
  shuffle  the instance's sigma^2 values permuted across roads (symmetric, as for the gate
           controls): keeps the level, scrambles the pattern
  fixed    "slack-only": like const, but rescaled to ONE reference variance level
           (beta_var = FIXED_BETA_REF) whatever the day's actual level. Keeps deadline
           slack and a fixed uncertainty width; removes the day's variance level too.
           If it matches const, the benefit is deadline-aware training, not uncertainty.

For a control, the variance accumulated since the last depot visit is tracked from the
CONTROL signal (track_accumulated_variance), so the true variance cannot leak in through
`env.time_var`. For sigma2 the tracker reproduces env.time_var exactly (tested).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.data.simulator import MU, SIGMA2, gather_from
from src.envs.cvrptw_stoch_env import _VAR_EPS, CVRPTWStochEnv, _std_normal_cdf
from src.models.cvrptw_gate_controls import gate_signal_tensor, softplus_inverse
from src.models.cvrptw_gated_policy import DEFAULT_GATE_INIT
from src.models.cvrptw_gateonly_policy import CVRPTWGateOnlyPolicy
from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder

RISK_PARAM_NAME = "risk_w"                     # must match src/train/pomo_stoch.py
DEFAULT_RISK_INIT = softplus_inverse(1.0)      # alpha = 1 at init, as ICAM's alpha
RISK_SIGNALS = ("sigma2", "const", "shuffle", "fixed")  # "sigma2" = VGT-R proper
# "fixed": the instance's mean sigma^2 rescaled to this reference variance level, so the
# uncertainty width no longer depends on the day. Midpoint of the training range
# beta_var ~ U[0.15, 0.6].
FIXED_BETA_REF = 0.375


def lateness_probability(
    env: CVRPTWStochEnv,
    risk_bank: Tensor | None = None,     # (B_inst, N, N, K, 2); None = the true bank
    accumulated_var: Tensor | None = None,  # (B*S,); None = env.time_var
) -> Tensor:
    """P(arrival at j > latest_j) if j is visited next, for every candidate. (B*S, N).

    Uses the same Normal approximation, the same mean travel time (current bin) and the
    same variance accumulation as the analytic reward, so for the chosen action it equals
    that step's `late_count` increment. Detached: it is a property of the state, not a
    learned quantity. The two optional arguments exist for the controls only.
    """
    bank = env.bank if risk_bank is None else risk_bank
    acc = env.time_var if accumulated_var is None else accumulated_var
    moments = gather_from(
        bank, env.current_node, env.current_bin(), env.instance_idx
    )                                                           # (B*S, N, 2)
    mu, sigma2 = moments[..., MU], moments[..., SIGMA2]         # (B*S, N) each
    arrival = env.time.unsqueeze(1) + mu                        # (B*S, N)
    latest = env.time_windows[..., 1]                           # (B*S, N)
    var = acc.unsqueeze(1) + sigma2                             # (B*S, N)
    z = (arrival - latest) / (var + _VAR_EPS).sqrt()            # (B*S, N)
    return _std_normal_cdf(z).detach()                          # (B*S, N) in [0, 1]


def track_accumulated_variance(env: CVRPTWStochEnv, risk_bank: Tensor) -> Tensor:
    """Variance accumulated since the last depot visit, from `risk_bank`. (B*S,).

    Mirrors CVRPTWStochEnv's analytic bookkeeping (arc variance read at the DEPARTURE bin,
    summed along the route, reset at the depot) for an arbitrary variance signal. Must be
    called once per decode step (it is, from _gate_penalty); extra calls without a step in
    between change nothing, because a trajectory that has not moved adds no arc.

    State lives ON the env and is keyed by `env._depot_idx`, a tensor created by
    `_reset_state` and never replaced afterwards, so every reset starts a fresh tracker.
    The first arc may already have been taken by a POMO forced start inside reset(). The
    tracker therefore starts from the depot at bin 0, where every route begins, and on
    that FIRST call counts the depot->start arc only where the env itself charged it
    (env.time_var > 0). env.time_var is used there as a yes/no flag only -- its value
    never enters the control, so the true variance cannot leak in.
    """
    state = getattr(env, "_risk_track", None)
    fresh = state is None or state[0] is not env._depot_idx
    if fresh:
        zeros = torch.zeros_like(env.time_var)                       # (B*S,)
        start = torch.zeros_like(env.current_node)                   # (B*S,) depot
        state = (env._depot_idx, zeros, start, torch.zeros_like(start))
    _, acc, last_node, last_bin = state
    node = env.current_node                                          # (B*S,)
    moved = node != last_node                                        # (B*S,)
    if fresh:
        moved = moved & (env.time_var > 0)                           # charged by the env?
    arc = risk_bank[env.instance_idx, last_node, node, last_bin][:, SIGMA2]  # (B*S,)
    acc = torch.where(moved, acc + arc, acc)
    acc = torch.where(node == 0, torch.zeros_like(acc), acc)          # depot: new route
    env._risk_track = (env._depot_idx, acc, node.clone(), env.current_bin().clone())
    return acc


class CVRPTWRiskBiasPolicy(CVRPTWGateOnlyPolicy):
    """VGT (D-prime) plus a learnable, non-negative lateness-risk bias in the decoder."""

    def __init__(
        self,
        encoder: EGATEncoder,
        decoder: PointerDecoder,
        gate_init: float = DEFAULT_GATE_INIT,
        risk_init: float = DEFAULT_RISK_INIT,
        risk_signal: str = "sigma2",
    ) -> None:
        super().__init__(encoder, decoder, gate_init=gate_init)
        if risk_signal not in RISK_SIGNALS:
            raise ValueError(f"risk_signal must be one of {RISK_SIGNALS}, got {risk_signal!r}")
        self.risk_w = nn.Parameter(torch.tensor(float(risk_init)))   # ()
        self.risk_signal = risk_signal
        # Private generator (shuffle control): must not move the global RNG stream.
        self._risk_gen = torch.Generator().manual_seed(
            (torch.initial_seed() + 2) % (2**63)
        )

    def _risk_bank(self, env: CVRPTWStochEnv) -> Tensor:
        """(B_inst, N, N, K, 2) = (true mu, control variance). Built once per env."""
        cache = getattr(env, "_risk_control_cache", None)
        if cache is not None and cache[0] is env.bank and cache[1] == self.risk_signal:
            return cache[2]
        if self.risk_signal == "fixed":
            beta = float(env.sim_cfg.beta_var)
            if beta <= 0.0:
                raise ValueError("the 'fixed' control needs beta_var > 0 to rescale")
            # sigma^2 is proportional to beta_var, so const * (ref / beta) is independent
            # of the day's level: the same uncertainty width on every day.
            signal = gate_signal_tensor(env.bank, "const") * (FIXED_BETA_REF / beta)
        else:
            signal = gate_signal_tensor(env.bank, self.risk_signal, self._risk_gen)
        rbank = torch.stack([env.bank[..., MU], signal], dim=-1)     # (B, N, N, K, 2)
        env._risk_control_cache = (env.bank, self.risk_signal, rbank)
        return rbank

    def risk_probability(self, env: CVRPTWStochEnv) -> Tensor:
        """p_late from the configured signal. (B*S, N).

        In 'sampled' (evaluation) mode past uncertainty has resolved and env.time_var is
        0, so the controls use 0 accumulated variance too -- same semantics as sigma2.
        """
        if self.risk_signal == "sigma2":
            return lateness_probability(env)
        rbank = self._risk_bank(env)
        if env.reward_mode == "analytic":
            acc = track_accumulated_variance(env, rbank)
        else:
            acc = torch.zeros_like(env.time_var)
        return lateness_probability(env, rbank, acc)

    def risk_strength(self) -> Tensor:
        """alpha = softplus(risk_w) >= 0. Scalar tensor."""
        return F.softplus(self.risk_w)

    def _gate_penalty(
        self, env: CVRPTWStochEnv, edge_feat: Tensor | None
    ) -> Tensor:
        """lambda * sigma^2(tau_t) + alpha * p_late, per candidate. (B*S, N).

        The first term is VGT's training-time gate; `forced_gate(policy, 0)` removes it at
        evaluation (it patches gate_strength only). The second term stays.
        """
        gate = super()._gate_penalty(env, edge_feat)                # (B*S, N)
        return gate + self.risk_strength() * self.risk_probability(env)

    def policy_metrics(self) -> dict[str, float]:
        """Logged every step: the training gate lambda and the risk-bias strength alpha."""
        return {**super().policy_metrics(),
                "risk_alpha": float(self.risk_strength().detach())}