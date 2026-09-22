"""Variant C: a SED2AM-style learned gate, ported into this framework as the competitor.

Mozhdehi et al. (SED2AM, ACM TKDD 2025) gate attention with the edge embedding. Their
encoder computes (their Eq. 10-11)

    u_ij = q_i . k_j / sqrt(d) + eps_ij          (additive edge bias, PRE-softmax)
    a_ij = softmax_j(u_ij) * sigmoid(eps_ij)     (multiplicative gate, POST-softmax,
                                                  deliberately NOT renormalised)

This class ports that MECHANISM, not their model: their problem (multi-trip,
time-dependent, maximum working hours) has no travel-time uncertainty at all, and their
gate is an unconstrained learned function of the edge embedding. Here the same gate form
is applied to OUR edge features (mu, sigma^2), so B, C and D differ only in how edge
information modulates attention:

    B  edge features enter the score, no gate
    C  learned sigmoid gate, POST-softmax, unconstrained (this file)
    D  monotone penalty softplus(w) * sigma^2, PRE-softmax, one shared scalar

WHERE THE GATE LIVES

  * ENCODER: one gate logit per head per layer, built from that layer's own edge
    projection (see EGATLayer with sigmoid_gate=True). Faithful to their Eq. 11.
  * DECODER pointer: SED2AM has no gate there, but variant D does, so C gets one too --
    otherwise "which gate" would be confounded with "where the gate is applied". A
    post-softmax multiplicative gate on a pointer must be renormalised to stay a
    probability distribution, and multiplying by sigmoid(g) then renormalising is exactly
    adding log sigmoid(g) to the logit. So the pointer gate is the penalty

        gate_penalty_j = -log sigmoid(g_j) = softplus(-g_j) >= 0,
        g_j = w . [mu_j, sigma^2_j] + b

    which fits the decoder's existing gate slot: pointer_decoder.py is untouched.

WHAT MAKES C A FAIR COMPETITOR

  * Same architecture, data, reward, training budget and seeds as B and D.
  * Starts as ~B: the gate weights are zero and the bias is +2, so every edge starts at
    sigmoid(2) = 0.881 and edge dependence is learned from there.
  * Its gate parameters get the SAME optimizer treatment as D's scalar (own group,
    lr 1e-2, no weight decay) -- see build_optimizer in src/train/pomo_stoch.py. Without
    that, C would be handicapped exactly where D was protected.

HONEST NOTE FOR THE WRITE-UP. In the POINTER, variant B's existing edge term
w_edge . [mu, sigma^2] can already express a linear response to sigma^2, so C's pointer
gate differs from B mainly by saturation (log sigmoid is bounded above by 0), and D's
differs mainly by the sign constraint and by sharing one scalar with the encoder. The
mechanisms genuinely differ in the ENCODER, where C's gate is post-softmax and
unnormalised while D's is pre-softmax and renormalising.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.graph_build import EncoderInputs
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_stoch_policy import CVRPTWStochPolicy
from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder

# sigmoid(2.0) = 0.8808: the gate starts near-open (C ~ B) but on a responsive part of the
# curve (sigmoid'(2) = 0.105), the analogue of variant D's softplus(0) = 0.693 init.
DEFAULT_SED_GATE_BIAS_INIT = 2.0


class CVRPTWSED2AMPolicy(CVRPTWStochPolicy):
    """Variant B plus a SED2AM-style learned sigmoid gate in encoder and pointer."""

    def __init__(
        self,
        encoder: EGATEncoder,
        decoder: PointerDecoder,
        gate_bias_init: float = DEFAULT_SED_GATE_BIAS_INIT,
    ) -> None:
        super().__init__(encoder, decoder)
        if not getattr(encoder.layers[0], "sigmoid_gate", False):
            raise ValueError(
                "variant C needs an encoder built with sigmoid_gate=True; build it with "
                "build_cvrptw_sed2am_policy (see src/models/factory.py)"
            )
        # Pointer gate: g_j = w . [mu_j, sigma^2_j] + b, one logit per candidate.
        self.sed_gate_ptr = nn.Linear(2, 1)                      # (2 -> 1), 3 parameters
        nn.init.zeros_(self.sed_gate_ptr.weight)                 # starts edge-independent
        nn.init.constant_(self.sed_gate_ptr.bias, float(gate_bias_init))
        self.sed_gate_active = True      # inference-time switch, not a saved parameter

    def set_sed_gates(self, active: bool) -> None:
        """Turn every SED2AM-style gate on or off (inference-time ablation).

        With active=False the encoder gate is skipped and the pointer penalty is None, so
        the policy reduces exactly to variant B's function with C's shared weights. This is
        C's analogue of forcing softplus(w) = 0 on variant D.
        """
        self.sed_gate_active = bool(active)
        for layer in self.encoder.layers:
            layer.sed_gate_active = bool(active)

    def _gate_penalty(
        self, env: CVRPTWStochEnv, edge_feat: Tensor | None
    ) -> Tensor | None:
        """-log sigmoid(g_j) >= 0 per candidate. (B*S, N), or None when the gate is off."""
        if not self.sed_gate_active:
            return None
        if edge_feat is None:
            raise RuntimeError(
                "variant C's pointer gate needs the per-step edge features; the policy was "
                "built without edge_feat_dim=2"
            )
        logit = self.sed_gate_ptr(edge_feat).squeeze(-1)          # (B*S, N)
        return F.softplus(-logit)   # = -log sigmoid(logit), >= 0, 0 as sigmoid -> 1

    def _encoder_gate_penalty(
        self, env: CVRPTWStochEnv, enc: EncoderInputs
    ) -> Tensor | None:
        """None by design: C's encoder gate lives INSIDE the layers (post-softmax), not in
        the pre-softmax penalty slot variant D uses."""
        return None

    def policy_metrics(self) -> dict[str, float]:
        """Logged every training step: what has the pointer gate learned?

        `ptr_w_sigma2` < 0 means more variance -> smaller sigmoid -> larger penalty, i.e.
        the unconstrained gate has LEARNED the monotone behaviour variant D assumes. A
        positive value means it learned the opposite, which variant D cannot do.
        """
        w = self.sed_gate_ptr.weight.detach().flatten()           # (2,) [mu, sigma^2]
        return {
            "ptr_w_mu": float(w[0]),
            "ptr_w_sigma2": float(w[1]),
            "ptr_b": float(self.sed_gate_ptr.bias.detach().flatten()[0]),
            "enc_gate_mean": float(
                torch.sigmoid(
                    torch.stack([l.sed_gate_b.detach().mean() for l in self.encoder.layers])
                ).mean()
            ),
        }