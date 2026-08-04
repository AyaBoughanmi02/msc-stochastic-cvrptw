"""Autoregressive pointer decoder (Kool et al. 2019 / Lei et al. 2022), one step.

This module scores ONE decode step: given the (fixed) node embeddings, the
current partial-tour state, and a feasibility mask, it returns a probability
distribution over the next node to visit. The rollout loop and the mask/capacity
updates live elsewhere (env + POMO loop); keeping the decoder single-step makes
it testable in isolation and gives the POMO multi-start loop a clean home.

Attention (Attention Model / Lei convention):
  - context query  q = W_q [ graph_emb || last_node_emb || remaining_capacity ]
  - multi-head glimpse: q attends over node embeddings (masked), -> refined query
  - single-head pointer: refined query vs node keys -> logits, clipped by
    C * tanh(.), masked, then log-softmax.

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
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.tanh_clip = tanh_clip

        # Context query: [graph_emb || last_node_emb || remaining_capacity] -> hidden.
        self.wq_context = nn.Linear(2 * hidden_dim + 1, hidden_dim, bias=False)
        # Glimpse (multi-head) projections of node embeddings.
        self.wk_glimpse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv_glimpse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, hidden_dim, bias=False)  # after glimpse
        # Pointer (single-head) key projection of node embeddings.
        self.wk_pointer = nn.Linear(hidden_dim, hidden_dim, bias=False)

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
    ) -> Tensor:
        """Return log-probabilities over the next node. Shape: (B, N)."""
        b, n = mask.shape

        # --- context query ---
        context = torch.cat(
            [graph_emb, last_node_emb, remaining_capacity], dim=-1
        )  # (B, 2*hidden + 1)
        q = self.wq_context(context)                       # (B, hidden)
        q = q.view(b, self.n_heads, 1, self.head_dim)      # (B, H, 1, hd)

        # --- multi-head glimpse: q attends over nodes (masked) ---
        # scores: (B, H, 1, N)
        scores = torch.matmul(q, cache.glimpse_k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        glimpse_mask = mask.view(b, 1, 1, n)               # broadcast over heads
        scores = scores.masked_fill(glimpse_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)               # (B, H, 1, N)
        glimpse = torch.matmul(attn, cache.glimpse_v)      # (B, H, 1, hd)
        glimpse = glimpse.reshape(b, self.hidden_dim)      # (B, hidden)  concat heads
        glimpse_q = self.w_out(glimpse)                    # (B, hidden)

        # --- single-head pointer: refined query vs node keys ---
        logits = torch.matmul(
            cache.pointer_k, glimpse_q.unsqueeze(-1)
        ).squeeze(-1) / math.sqrt(self.hidden_dim)         # (B, N)
        logits = self.tanh_clip * torch.tanh(logits)       # tanh clip, C=10
        logits = logits.masked_fill(mask, float("-inf"))   # forbid infeasible nodes
        return F.log_softmax(logits, dim=-1)               # (B, N)