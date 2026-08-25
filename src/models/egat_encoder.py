"""Residual E-GAT encoder (Lei et al. 2022, Eq. 3.5 / 3.6), scalar-per-head form.

Faithful to Lei's E-GAT architecture, with two locked, documented deviations
(see notebooks/lei_code_map.md):

  1. Scalar-per-head attention (Lei's released code uses channel-wise vector
     attention). Reason: the variance gate u_ij = score_ij - softplus(w)*sigma^2
     needs a well-defined scalar pre-softmax logit.
  2. H heads in the encoder (Lei's encoder is headless). Reason: POMO convention
     consistency (Kwon et al. 2020 use H=8) and partial recovery of expressiveness.

Everything else matches Lei:
  - LeakyReLU is applied OUTSIDE the attention projection => GAT-v1 (static
    attention), not GATv2.
  - The edge feature is concatenated into the pre-softmax score.
  - The edge embedding is computed once up front and reused (frozen) by every
    layer (static edges).
  - Layers are residual (Eq. 3.6). Global embedding = per-graph mean of node
    embeddings.

Convention: i = target/receiver node, j = source/neighbor node. PyG's default
message flow is source_to_target, so inside `message` x_i is the target and x_j
the source; attention is normalized over the neighbors j of each target i.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


def _make_norm(kind: str, dim: int) -> nn.Module:
    """Input normalization for the encoder.

    'batch' matches Lei (Eq. 3.3/3.4 / released code: BatchNorm at the input).
    'layer' and 'none' have NO train/eval discrepancy, unlike BatchNorm whose
    running statistics differ between the two modes.
    """
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}; use 'batch', 'layer', or 'none'")


class EGATLayer(MessagePassing):
    """One E-GAT attention layer (no residual; the encoder adds the residual).

    Two attention granularities:
      - 'scalar' (locked model): one LeakyReLU score PER HEAD per edge,
            score_ij^h = LeakyReLU( a_h . [W h_i || W h_j || W_e e_ij] ),
        so the variance gate has a well-defined scalar pre-softmax logit.
      - 'channel' (Lei's released GatConv, diagnostic only): a full
        hidden-dim attention vector per edge,
            alpha_ij = softmax_j( LeakyReLU( Linear([h_i || h_j || e_ij]) ) ),
        with the raw (unprojected) edge embedding, matching Lei exactly. Used to
        measure the expressiveness cost of the scalar-head simplification; it is
        NOT used by variants A-D.

    Both modes share one node projection W (= value = score input), matching
    Lei's `x_j * alpha` with `x = fc(x)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        n_heads: int,
        negative_slope: float = 0.2,
        attn: str = "scalar",     # 'scalar' (locked) or 'channel' (Lei diagnostic)
    ) -> None:
        # node_dim=0: per-node tensors are indexed along dim 0 by edge_index in
        # both modes ((N, H, hd) for scalar, (N, hidden) for channel).
        super().__init__(aggr="add", node_dim=0)
        if attn not in ("scalar", "channel"):
            raise ValueError(f"unknown attn {attn!r}; use 'scalar' or 'channel'")
        self.attn = attn
        self.hidden_dim = hidden_dim
        self.negative_slope = negative_slope
        self.lin_node = nn.Linear(hidden_dim, hidden_dim)  # W (score + value)

        if attn == "scalar":
            if hidden_dim % n_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
                )
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
            self.lin_edge = nn.Linear(edge_dim, hidden_dim)  # W_e (edge -> per-head)
            # Per-head attention vector a_h over [W h_i || W h_j || W_e e_ij].
            self.att = nn.Parameter(torch.empty(n_heads, 3 * self.head_dim))  # (H, 3*hd)
        else:  # channel: Lei's Linear(2*out + edge_channels -> out), raw edge in
            self.attn_lin = nn.Linear(2 * hidden_dim + edge_dim, hidden_dim)

        self._alpha: Tensor | None = None  # scratch: last attention weights
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Orthogonal init on weights, zeros on biases (held fixed across modes)."""
        nn.init.orthogonal_(self.lin_node.weight, gain=1.0)
        nn.init.zeros_(self.lin_node.bias)
        if self.attn == "scalar":
            nn.init.orthogonal_(self.lin_edge.weight, gain=1.0)
            nn.init.zeros_(self.lin_edge.bias)
            nn.init.xavier_uniform_(self.att)
        else:
            nn.init.orthogonal_(self.attn_lin.weight, gain=1.0)
            nn.init.zeros_(self.attn_lin.bias)

    def forward(
        self,
        x: Tensor,            # (num_nodes, hidden_dim)
        edge_index: Tensor,   # (2, num_edges)  row 0 = source j, row 1 = target i
        edge_attr: Tensor,    # (num_edges, edge_dim)  frozen edge embedding
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return updated node features (num_nodes, hidden_dim)."""
        if self.attn == "scalar":
            h = self.lin_node(x).view(-1, self.n_heads, self.head_dim)          # (N,H,hd)
            e = self.lin_edge(edge_attr).view(-1, self.n_heads, self.head_dim)  # (E,H,hd)
            out = self.propagate(edge_index, x=h, edge=e)                       # (N,H,hd)
            out = out.reshape(-1, self.hidden_dim)                              # (N,hidden)
        else:  # channel: flat, raw edge concatenated into the attention Linear
            h = self.lin_node(x)                                               # (N,hidden)
            out = self.propagate(edge_index, x=h, edge=edge_attr)              # (N,hidden)

        if return_attention_weights:
            alpha = self._alpha
            self._alpha = None
            assert alpha is not None
            return out, (edge_index, alpha)
        self._alpha = None
        return out

    def message(
        self,
        x_i: Tensor,          # (E,H,hd) scalar | (E,hidden) channel  target (i)
        x_j: Tensor,          # source (j), same shape as x_i
        edge: Tensor,         # (E,H,hd) scalar | (E,edge_dim) channel
        index: Tensor,        # (E,)  target index i (softmax groups)
        size_i: int | None,   # number of target nodes
    ) -> Tensor:
        cat = torch.cat([x_i, x_j, edge], dim=-1)
        if self.attn == "scalar":
            # scalar score per head; LeakyReLU OUTSIDE => GAT-v1 (static)
            score = F.leaky_relu((cat * self.att).sum(dim=-1), self.negative_slope)  # (E,H)
            alpha = softmax(score, index, num_nodes=size_i)                    # (E,H) over j
            self._alpha = alpha
            return x_j * alpha.unsqueeze(-1)                                   # (E,H,hd)
        # channel: full hidden-dim attention vector per edge (Lei GatConv)
        alpha = F.leaky_relu(self.attn_lin(cat), self.negative_slope)         # (E,hidden)
        alpha = softmax(alpha, index, num_nodes=size_i)                       # (E,hidden) over j
        self._alpha = alpha
        return x_j * alpha                                                    # (E,hidden)


class EGATEncoder(nn.Module):
    """Initial embeddings + L residual E-GAT layers with static (frozen) edges."""

    def __init__(
        self,
        input_node_dim: int,      # CVRP: 3 = [x, y, demand]
        input_edge_dim: int,      # Stage-1: 1 = distance; later 2 = (mu, sigma^2)
        hidden_dim: int = 128,
        hidden_edge_dim: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        negative_slope: float = 0.2,
        norm: str = "batch",      # 'batch' (Lei), 'layer', or 'none'
        attn: str = "scalar",     # 'scalar' (locked) or 'channel' (Lei diagnostic)
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.attn = attn
        self.fc_node = nn.Linear(input_node_dim, hidden_dim)
        self.bn_node = _make_norm(norm, hidden_dim)        # name kept for ckpt compat
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)
        self.bn_edge = _make_norm(norm, hidden_edge_dim)   # name kept for ckpt compat
        self.layers = nn.ModuleList(
            EGATLayer(hidden_dim, hidden_edge_dim, n_heads, negative_slope, attn)
            for _ in range(n_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.fc_node.weight, gain=1.0)
        nn.init.zeros_(self.fc_node.bias)
        nn.init.orthogonal_(self.fc_edge.weight, gain=1.0)
        nn.init.zeros_(self.fc_edge.bias)
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        x: Tensor,            # (num_nodes, input_node_dim)
        edge_index: Tensor,   # (2, num_edges)
        edge_attr: Tensor,    # (num_edges, input_edge_dim)
        batch: Tensor | None = None,  # (num_nodes,) graph id per node; None => 1 graph
    ) -> tuple[Tensor, Tensor]:
        """Return (node_emb, global_emb).

        node_emb:   (num_nodes, hidden_dim)
        global_emb: (num_graphs, hidden_dim)  per-graph mean of node embeddings
        """
        h = self.bn_node(self.fc_node(x))          # (num_nodes, hidden)  project once
        # Edge embedding computed ONCE and reused (frozen) by every layer.
        e = self.bn_edge(self.fc_edge(edge_attr))  # (num_edges, hidden_edge)
        for layer in self.layers:
            h = h + layer(h, edge_index, e)        # residual (Eq. 3.6); same frozen e
        if batch is None:
            batch = h.new_zeros(h.size(0), dtype=torch.long)  # single graph
        global_emb = global_mean_pool(h, batch)    # (num_graphs, hidden)
        return h, global_emb