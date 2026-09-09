"""Autoregressive pointer decoder (Kool et al. 2019 / Lei et al. 2022), one step.

This module scores ONE decode step: given the (fixed) node embeddings, the
current partial-tour state, and a feasibility mask, it returns a probability
distribution over the next node to visit. The rollout loop and the mask/capacity
updates live elsewhere (env + POMO loop); keeping the decoder single-step makes
it testable in isolation and gives the POMO multi-start loop a clean home.

Attention (Attention Model / Lei convention):
  - context query  q = W_q [ graph_emb || last_node_emb || remaining_capacity
                             (|| context_extra) ]
  - multi-head glimpse: q attends over node embeddings (masked), -> refined query
  - single-head pointer: refined query vs node keys -> logits, clipped by
    C * tanh(.), masked, then log-softmax.

context_extra is an optional block of dynamic scalars appended to the context
(CVRPTW passes the normalized clock); context_extra_dim=0 (default) reproduces the
CVRP decoder exactly.

Mask convention (shared with the env): mask is (B, N) boolean, True = infeasible.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DecoderCache(NamedTuple):
    """Per-instance projections of node embeddings, fixed across decode steps."""

    glimpse_k: Tensor  # (B, n_heads, N, head_dim)
    glimpse_v: Tensor  # (B, n_heads, N, head_dim)
    pointer_k: Tensor  # (B, N, hidden)


class PointerDecoder(nn.Module):
    """Single-step glimpse + pointer decoder with tanh-clipped logits."""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        tanh_clip: float = 10.0,
        context_extra_dim: int = 0,   # extra dynamic context scalars (CVRPTW clock = 1)
        edge_feat_dim: int = 0,       # per-candidate edge features; 0 = CVRP/variant-A decoder
        edge_bias_mode: str = "pre_tanh",   # 'pre_tanh' (locked) or 'post_tanh' (diagnostic)
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        if edge_feat_dim < 0:
            raise ValueError(f"edge_feat_dim must be >= 0, got {edge_feat_dim}")
        if edge_bias_mode not in ("pre_tanh", "post_tanh"):
            raise ValueError(
                f"unknown edge_bias_mode {edge_bias_mode!r}; use 'pre_tanh' or 'post_tanh'"
            )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.tanh_clip = tanh_clip
        self.context_extra_dim = context_extra_dim
        self.edge_feat_dim = edge_feat_dim
        self.edge_bias_mode = edge_bias_mode

        # Context query: [graph_emb || last_node_emb || remaining_capacity || extra] -> hidden.
        self.wq_context = nn.Linear(2 * hidden_dim + 1 + context_extra_dim, hidden_dim, bias=False)
        # Glimpse (multi-head) projections of node embeddings.
        self.wk_glimpse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv_glimpse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, hidden_dim, bias=False)  # after glimpse
        # Pointer (single-head) key projection of node embeddings.
        self.wk_pointer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Per-candidate edge term (variants B/C/D). Created ONLY when requested, so with
        # edge_feat_dim=0 the state_dict keys are identical to the certified decoder's and
        # cvrp20_cos3e4.pt / cvrptwA.pt keep loading with strict=True.
        if edge_feat_dim > 0:
            self.w_edge = nn.Linear(edge_feat_dim, 1, bias=False)

        # Diagnostics scratch (same idiom as EGATLayer._alpha): when record_diagnostics is
        # on, the last tanh ARGUMENT is stashed, detached, for the saturation meter.
        # Off by default -> zero training overhead.
        self.record_diagnostics = False
        self._last_u: Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (
            self.wq_context,
            self.wk_glimpse,
            self.wv_glimpse,
            self.w_out,
            self.wk_pointer,
        ):
            nn.init.xavier_uniform_(lin.weight)
        if self.edge_feat_dim > 0:
            # ZERO, deliberately (not xavier): the edge term starts inert, so variant B
            # begins training as exactly variant A's pointer function. This removes an
            # initialisation confound between A/B/D and costs no learnability -- the
            # gradient w.r.t. w_edge is dL/du * edge_feat, which is not zero.
            nn.init.zeros_(self.w_edge.weight)

    def precompute(self, node_emb: Tensor) -> DecoderCache:
        """Project node embeddings once; reused every decode step.

        node_emb: (B, N, hidden)
        """
        b, n, _ = node_emb.shape
        # (B, N, hidden) -> (B, n_heads, N, head_dim)
        k = self.wk_glimpse(node_emb).view(b, n, self.n_heads, self.head_dim)
        v = self.wv_glimpse(node_emb).view(b, n, self.n_heads, self.head_dim)
        glimpse_k = k.permute(0, 2, 1, 3).contiguous()  # (B, H, N, hd)
        glimpse_v = v.permute(0, 2, 1, 3).contiguous()  # (B, H, N, hd)
        pointer_k = self.wk_pointer(node_emb)            # (B, N, hidden)
        return DecoderCache(glimpse_k, glimpse_v, pointer_k)

    def forward(
        self,
        cache: DecoderCache,
        graph_emb: Tensor,           # (B, hidden)
        last_node_emb: Tensor,       # (B, hidden)  embedding of current/last node
        remaining_capacity: Tensor,  # (B, 1)       normalised remaining capacity
        mask: Tensor,                # (B, N) bool, True = infeasible
        context_extra: Tensor | None = None,   # (B, context_extra_dim) dynamic scalars
        edge_feat: Tensor | None = None,       # (B, N, edge_feat_dim) per-candidate edges
        gate_penalty: Tensor | None = None,    # (B, N) >= 0, subtracted from the logit
    ) -> Tensor:
        """Return log-probabilities over the next node. Shape: (B, N).

        edge_feat carries the CLOCK-REFRESHED travel-time moments of the edges leaving the
        current node, (mu, sigma^2)(tau_t), projected to one scalar per candidate:

            u_j = compat_j + w_edge . edge_feat_j - gate_penalty_j        [pre_tanh]
            logits = C * tanh(u_j)

        gate_penalty is the slot variant D subtracts softplus(w) * sigma^2 into. It is
        passed IN rather than computed here because w is shared with the encoder and so
        lives on the policy -- which means variant D needs no change to this file, and B
        and D provably differ by exactly one term.

        In 'post_tanh' mode the two terms are added AFTER the clip instead; that is
        unbounded (a large penalty can effectively mask a node) and is a diagnostic for
        testing whether tanh saturation is muzzling the gate, not the locked default.
        """
        b, n = mask.shape
        if self.edge_feat_dim > 0 and edge_feat is None:
            raise ValueError(
                f"decoder was built with edge_feat_dim={self.edge_feat_dim} but edge_feat "
                f"was not supplied"
            )
        if self.edge_feat_dim == 0 and edge_feat is not None:
            raise ValueError(
                "edge_feat was supplied but the decoder was built with edge_feat_dim=0"
            )
        if edge_feat is not None and edge_feat.shape != (b, n, self.edge_feat_dim):
            raise ValueError(
                f"edge_feat must be {(b, n, self.edge_feat_dim)}, got "
                f"{tuple(edge_feat.shape)}"
            )
        if gate_penalty is not None and gate_penalty.shape != (b, n):
            raise ValueError(
                f"gate_penalty must be {(b, n)}, got {tuple(gate_penalty.shape)}"
            )

        # --- context query ---
        parts = [graph_emb, last_node_emb, remaining_capacity]
        if context_extra is not None:
            parts.append(context_extra)                    # (B, context_extra_dim)
        context = torch.cat(parts, dim=-1)                 # (B, 2*hidden + 1 + extra)
        # Guard the seam: a mismatch crashes loudly instead of training silently.
        if context.shape[-1] != self.wq_context.in_features:
            raise ValueError(
                f"decoder context width {context.shape[-1]} != expected "
                f"{self.wq_context.in_features} (context_extra_dim={self.context_extra_dim})"
            )
        q = self.wq_context(context)                       # (B, hidden)
        q = q.view(b, self.n_heads, 1, self.head_dim)      # (B, H, 1, hd)

        # --- multi-head glimpse: q attends over nodes (masked) ---
        scores = torch.matmul(q, cache.glimpse_k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )                                                  # (B, H, 1, N)
        glimpse_mask = mask.view(b, 1, 1, n)               # broadcast over heads
        scores = scores.masked_fill(glimpse_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)               # (B, H, 1, N)
        glimpse = torch.matmul(attn, cache.glimpse_v)      # (B, H, 1, hd)
        glimpse = glimpse.reshape(b, self.hidden_dim)      # (B, hidden)  concat heads
        glimpse_q = self.w_out(glimpse)                    # (B, hidden)

        # --- single-head pointer: refined query vs node keys ---
        compat = torch.matmul(
            cache.pointer_k, glimpse_q.unsqueeze(-1)
        ).squeeze(-1) / math.sqrt(self.hidden_dim)         # (B, N)

        edge_term = None
        if self.edge_feat_dim > 0:
            edge_term = self.w_edge(edge_feat).squeeze(-1)  # (B, N, 1) -> (B, N)

        if self.edge_bias_mode == "pre_tanh":
            u = compat                                      # (B, N)
            if edge_term is not None:
                u = u + edge_term
            if gate_penalty is not None:
                u = u - gate_penalty
            logits = self.tanh_clip * torch.tanh(u)         # tanh clip, C=10
        else:                                               # post_tanh (diagnostic)
            u = compat
            logits = self.tanh_clip * torch.tanh(u)
            if edge_term is not None:
                logits = logits + edge_term
            if gate_penalty is not None:
                logits = logits - gate_penalty

        # u is the tanh ARGUMENT, i.e. what saturates. Detached so nothing is retained
        # in the autograd graph across decode steps.
        self._last_u = u.detach() if self.record_diagnostics else None

        logits = logits.masked_fill(mask, float("-inf"))   # forbid infeasible nodes
        return F.log_softmax(logits, dim=-1)               # (B, N)