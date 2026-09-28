"""Variant M0: the TRUE mean-only control. sigma^2 influences neither decisions nor training.

Same architecture as variant D-prime -- encoder edge feature mu_bar alone
(input_edge_dim=1), decoder per-candidate feature mu alone (edge_feat_dim=1) -- but with
NO variance gate at all. The gate hooks are inherited from CVRPTWPolicy and return None,
so there is no gate parameter, no optimizer group for one, and no path from sigma^2 to
any logit, at training time or at inference.

WHY A SEPARATE VARIANT, NOT D-prime WITH THE GATE FROZEN. softplus(w) has no finite
pre-image of 0, so "frozen at zero" could only mean "frozen at ~1e-9". Removing the gate
makes the zero exact and keeps the state_dict free of a dead parameter.

SAME INITIALISATION AS D-prime. D-prime's `gate_w` is created with torch.tensor(), which
draws no random numbers, so under the same seed M0 and D-prime start from bit-identical
encoder and decoder weights. The two runs differ ONLY in whether the gate existed during
training -- which is exactly the M-vs-M0 question.

WHAT M0 STILL KNOWS ABOUT RISK. It sees mu everywhere and the coordinates of every node,
and it is trained on the analytic expected-tardiness reward, which prices variance. So it
is risk-aware through the GRADIENT, never through an INPUT. Its actions are provably
invariant to beta_var (tested), because the mask and the clock use mu alone.

The two hooks below duplicate D-prime's on purpose: making D-prime inherit from this class
would change the class hierarchy of existing checkpoints' code for no functional gain.
"""

from __future__ import annotations

from torch import Tensor

from src.data.graph_build import EncoderInputs, build_graph_tw
from src.data.simulator import SIGMA2, gather_from, horizon_average
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_stoch_policy import CVRPTWStochPolicy


class CVRPTWMeanOnlyPolicy(CVRPTWStochPolicy):
    """Variant M0: mu in the encoder and the pointer, sigma^2 nowhere, no gate."""

    def _encoder_inputs(self, env: CVRPTWStochEnv) -> EncoderInputs:
        """Graph inputs whose edge feature is the horizon-averaged mu_bar ALONE.

        `[..., :SIGMA2]` keeps columns [0, SIGMA2), written against the index rather than
        a literal 1 so a change to the moment layout cannot silently feed variance back in.
        """
        env = self._require_stoch_env(env)
        moments = horizon_average(env.bank)                   # (B_inst, N, N, 2)
        mu_bar = moments[env.instance_idx][..., :SIGMA2]      # (B*S, N, N, 1)
        return build_graph_tw(
            env.coords, env.demand, env.time_windows, env.service_time,
            edge_attr=mu_bar.contiguous(),
        )

    def _decoder_edge_feat(self, env: CVRPTWStochEnv) -> Tensor:
        """Clock-refreshed mu ALONE for the edges leaving the current node. (B*S, N, 1)."""
        env = self._require_stoch_env(env)
        moments = gather_from(
            env.bank, env.current_node, env.current_bin(), env.instance_idx
        )                                                     # (B*S, N, 2)
        return moments[..., :SIGMA2].contiguous()             # (B*S, N, 1)