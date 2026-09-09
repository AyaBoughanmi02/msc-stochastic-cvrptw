"""Stochastic CVRPTW policy (variants B/C/D): Edge-Bank moments in encoder and pointer.

Subclass of CVRPTWPolicy that overrides only the rollout HOOKS -- the autoregressive loop,
the action selection, the multinomial guard and the completion check are all inherited
unchanged:

  1. `_encoder_inputs`: the encoder's edge feature becomes the horizon-averaged
     (mu_bar, sigma_bar^2) instead of a single distance, so input_edge_dim is 2.
     Still ONE encoder pass per instance -- the amortization story is untouched.
  2. `_decoder_edge_feat`: each decode step, the pointer additionally reads the
     CLOCK-REFRESHED moments (mu, sigma^2)(tau_t) of the edges leaving the current node.
     This is the only genuinely time-dependent quantity the network sees besides the
     clock scalar, and it is why the decoder -- not the encoder -- is where
     time-dependence lives.

Variant B leaves `_gate_penalty` at its inherited None: it PLANS ON THE MEAN and reads
sigma^2 only as a feature. Variant D overrides `_gate_penalty` to return
softplus(w) * sigma^2(tau_t) and changes nothing else, which is what makes B and D differ
by exactly one term.

The Edge Bank is kept at INSTANCE granularity (see CVRPTWStochEnv), so both hooks address
it through `env.instance_idx` rather than a bank replicated per POMO start.
"""

from __future__ import annotations

from torch import Tensor

from src.data.graph_build import EncoderInputs, build_graph_tw
from src.data.simulator import gather_from, horizon_average
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_policy import CVRPTWPolicy


class CVRPTWStochPolicy(CVRPTWPolicy):
    """Encoder + pointer decoder rollout policy for stochastic CVRPTW (variant B)."""

    @staticmethod
    def _require_stoch_env(env: object) -> CVRPTWStochEnv:
        """Crash-not-silent: this policy is meaningless on a deterministic env."""
        if not isinstance(env, CVRPTWStochEnv):
            raise TypeError(
                f"CVRPTWStochPolicy needs a CVRPTWStochEnv (it reads the Edge Bank), got "
                f"{type(env).__name__}"
            )
        return env

    def _encoder_inputs(self, env: CVRPTWStochEnv) -> EncoderInputs:
        """Graph inputs with 2-dim edge features (mu_bar, sigma_bar^2).

        The bank holds one row per INSTANCE; the encoder needs one graph per TRAJECTORY,
        so the moments are expanded through instance_idx. That expansion happens once per
        rollout (not per step): (B*S, N, N, 2), ~2.4 MB at B*S=672, N=21.
        """
        env = self._require_stoch_env(env)
        moments = horizon_average(env.bank)                  # (B_inst, N, N, 2)
        edge_attr = moments[env.instance_idx]                # (B*S,   N, N, 2)
        return build_graph_tw(
            env.coords, env.demand, env.time_windows, env.service_time,
            edge_attr=edge_attr,
        )

    def _decoder_edge_feat(self, env: CVRPTWStochEnv) -> Tensor:
        """Clock-refreshed moments of the edges leaving the current node. (B*S, N, 2).

        Origin- and time-specific, so it cannot be precomputed: the current node and the
        time bin are exactly what change each step.
        """
        return gather_from(
            env.bank, env.current_node, env.current_bin(), env.instance_idx
        )                                                    # (B*S, N, 2)