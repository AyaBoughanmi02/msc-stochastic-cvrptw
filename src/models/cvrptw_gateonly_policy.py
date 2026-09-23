"""Variant D-prime: the monotone gate as the ONLY path from variance to a decision.

Variant D (CVRPTWGatedPolicy) adds the gate `-softplus(w) * sigma^2` but LEAVES sigma^2
in place as an ordinary feature: the decoder's `w_edge` is a 1x2 map whose second column
multiplies sigma^2 with no sign constraint, and the encoder's `fc_edge` embeds both
moments. Variance therefore reaches the logits twice -- once through a term forced to be
a penalty, once through a free weight that can cancel it. Only the SUM is identified by
the loss, so:

  * the monotonicity guarantee does not actually hold (a positive `w_edge[0, 1]` would
    make a riskier edge MORE attractive);
  * the gradient on w is weak, because the loss barely distinguishes "small lambda with
    one w_edge" from "large lambda with another". That is the likeliest explanation for
    training settling at softplus(w) ~ 4.0 while held-out data prefers 6-7.

D-prime closes the second path. The encoder edge feature becomes mu_bar alone
(input_edge_dim=1) and the decoder's per-candidate feature becomes mu alone
(edge_feat_dim=1), while the gate still reads the FULL variance from the Edge Bank. So:

  * D-prime is a STRICT subset of variant B's function class -- the constraint binds;
  * B and D-prime receive the same information (both see mu and sigma^2); they differ
    only in the FORM that information is allowed to take. That is the comparison the
    thesis claim needs.

COST. `_encoder_gate_penalty` and `_gate_penalty` recompute the Edge-Bank moments rather
than reading them off the tensor handed to the network, because that tensor no longer
carries sigma^2. The encoder recompute happens once per rollout; the decoder recompute
once per decode step, which roughly doubles one advanced-index op per step (a small
percentage of step time, measured against a full forward/backward). Correctness over
micro-optimisation: nothing can silently desynchronise, since both the feature and the
penalty are derived from `env.bank` at the same clock.
"""

from __future__ import annotations

from torch import Tensor

from src.data.graph_build import EncoderInputs, build_graph_tw
from src.data.simulator import SIGMA2, gather_from, horizon_average
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_gated_policy import CVRPTWGatedPolicy


class CVRPTWGateOnlyPolicy(CVRPTWGatedPolicy):
    """Variant D-prime: variance reaches the logits through the monotone gate only."""

    def _encoder_inputs(self, env: CVRPTWStochEnv) -> EncoderInputs:
        """Graph inputs whose edge feature is the horizon-averaged mu_bar ALONE.

        `[..., :SIGMA2]` keeps columns [0, SIGMA2) = the mean only. Written against the
        SIGMA2 index rather than a literal 1 so a change to the moment layout cannot
        silently feed variance back in.
        """
        env = self._require_stoch_env(env)
        moments = horizon_average(env.bank)                   # (B_inst, N, N, 2)
        mu_bar = moments[env.instance_idx][..., :SIGMA2]      # (B*S, N, N, 1)
        return build_graph_tw(
            env.coords, env.demand, env.time_windows, env.service_time,
            edge_attr=mu_bar.contiguous(),
        )

    def _encoder_gate_penalty(
        self, env: CVRPTWStochEnv, enc: EncoderInputs
    ) -> Tensor:
        """softplus(w) * sigma_bar^2 per edge, taken from the bank. (num_edges,).

        Variant D reads this off `enc.edge_attr`; here that tensor holds mu_bar only, so
        the variance comes straight from the Edge Bank. The dense (B*S, N, N) variance
        block flattens row-major into exactly `build_graph`'s edge order -- the same
        contract `_flatten_dense_edge_attr` documents -- and the guard below fails loudly
        if that ever stops holding.
        """
        env = self._require_stoch_env(env)
        moments = horizon_average(env.bank)                   # (B_inst, N, N, 2)
        sigma2 = moments[env.instance_idx][..., SIGMA2]       # (B*S, N, N)
        flat = sigma2.reshape(-1)                             # (E,)
        n_edges = enc.edge_index.size(1)
        if flat.numel() != n_edges:
            raise ValueError(
                f"gate penalty has {flat.numel()} entries but the graph has {n_edges} "
                f"edges; the Edge-Bank block and build_graph disagree about layout"
            )
        return self.gate_strength() * flat                    # (E,)

    def _decoder_edge_feat(self, env: CVRPTWStochEnv) -> Tensor:
        """Clock-refreshed mu ALONE for the edges leaving the current node. (B*S, N, 1)."""
        env = self._require_stoch_env(env)
        moments = gather_from(
            env.bank, env.current_node, env.current_bin(), env.instance_idx
        )                                                     # (B*S, N, 2)
        return moments[..., :SIGMA2].contiguous()             # (B*S, N, 1)

    def _gate_penalty(
        self, env: CVRPTWStochEnv, edge_feat: Tensor | None
    ) -> Tensor:
        """softplus(w) * sigma^2(tau_t) per candidate. (B*S, N).

        `edge_feat` is ignored: it now carries mu only. The variance is re-gathered from
        the bank at the SAME current node and time bin, so feature and penalty cannot
        drift apart.
        """
        env = self._require_stoch_env(env)
        moments = gather_from(
            env.bank, env.current_node, env.current_bin(), env.instance_idx
        )                                                     # (B*S, N, 2)
        return self.gate_strength() * moments[..., SIGMA2]    # (B*S, N)