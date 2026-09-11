"""Variant D: the monotone variance gate -- the thesis contribution.

Subclass of CVRPTWStochPolicy (variant B) that adds EXACTLY ONE learnable scalar and
changes nothing else. Both stages of the network get the same gate:

    u_ij = score_ij - softplus(w) * sigma^2_ij          applied PRE-SOFTMAX

  * ENCODER self-attention, on the horizon-averaged sigma_bar^2 (static, computed once,
    handed unchanged to all four layers).
  * DECODER pointer, on the clock-refreshed sigma^2(tau_t) of the edges leaving the
    current node (recomputed each step).

The SCALAR IS THE ONLY THING SHARED -- the variance input differs by design, as locked.
`softplus(w) >= 0` makes the gate monotone by construction: more variance can only ever
mean less attention, never more, so the inductive bias cannot invert during training.

ONE PARAMETER, ONE SOURCE OF TRUTH. w lives here, on the policy. The encoder and decoder
hold no gate parameter of their own; they receive the already-computed penalty as an
argument. Gradients from both paths accumulate into the same tensor, so "shared" is a
property of the computation graph rather than a convention that could drift. That is what
makes the B-vs-D ablation worth exactly one scalar's worth of inductive bias.

WHY D TOUCHES NO OTHER NETWORK FILE. The decoder's `gate_penalty` slot shipped with
variant B precisely so this class would need no edit to `pointer_decoder.py`, and the
encoder's gate argument is inert when None. So B and D differ by one term, provably --
verified bitwise by the gate chain (`assert_gate_chain_inert`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.graph_build import EncoderInputs
from src.data.simulator import SIGMA2
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_stoch_policy import CVRPTWStochPolicy
from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder

# softplus(0) = 0.6931: small against attention scores of order 1-2, so variant D starts
# as essentially variant B, but NON-zero and on the healthy part of the curve
# (d softplus/dw = sigmoid(0) = 0.5, no vanishing gradient). The analogue of w_edge's
# zero-init. Note this only works because gate_w gets its own optimizer group with a
# higher learning rate -- see build_optimizer in src/train/pomo_stoch.py.
DEFAULT_GATE_INIT = 0.0


class CVRPTWGatedPolicy(CVRPTWStochPolicy):
    """Variant B plus a single shared monotone variance gate."""

    def __init__(
        self,
        encoder: EGATEncoder,
        decoder: PointerDecoder,
        gate_init: float = DEFAULT_GATE_INIT,
    ) -> None:
        super().__init__(encoder, decoder)
        # THE contribution: one scalar, shared by encoder and decoder.
        self.gate_w = nn.Parameter(torch.tensor(float(gate_init)))   # ()

    def gate_strength(self) -> Tensor:
        """softplus(w) >= 0 -- the monotone gate coefficient. Scalar tensor."""
        return F.softplus(self.gate_w)

    def _encoder_gate_penalty(
        self, env: CVRPTWStochEnv, enc: EncoderInputs
    ) -> Tensor:
        """softplus(w) * sigma_bar^2 per edge. (num_edges,).

        Read from `enc.edge_attr` -- the very tensor handed to the encoder -- so the
        penalty and the graph cannot disagree about edge ordering.
        """
        if enc.edge_attr.size(-1) <= SIGMA2:
            raise ValueError(
                f"the variance gate needs 2-dim edge features (mu_bar, sigma_bar^2); got "
                f"{enc.edge_attr.size(-1)}-dim. Build the policy with input_edge_dim=2."
            )
        return self.gate_strength() * enc.edge_attr[:, SIGMA2]       # (E,)

    def _gate_penalty(
        self, env: CVRPTWStochEnv, edge_feat: Tensor | None
    ) -> Tensor:
        """softplus(w) * sigma^2(tau_t) per candidate. (B*S, N).

        Clock-refreshed, unlike the encoder's horizon average -- same scalar, different
        variance input, exactly as locked.
        """
        if edge_feat is None:
            raise RuntimeError(
                "the variance gate needs the decoder's per-step edge features; the policy "
                "was built without edge_feat_dim=2"
            )
        return self.gate_strength() * edge_feat[..., SIGMA2]         # (B*S, N)

    def policy_metrics(self) -> dict[str, float]:
        """Logged every training step: is the gate learning to fire, or collapsing?"""
        return {"softplus_w": float(self.gate_strength().detach())}