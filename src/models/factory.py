from __future__ import annotations

from typing import Any

from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder
from src.models.policy import CVRPPolicy
from src.models.cvrptw_policy import CVRPTWPolicy
from src.models.cvrptw_stoch_policy import CVRPTWStochPolicy

# Default hyperparameters for the CVRP reproduction of Lei et al. (2022).
DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "input_node_dim": 3,     # [x, y, demand]
    "input_edge_dim": 1,     # distance (Stage-1); 2 later for (mu, sigma^2)
    "hidden_dim": 128,
    "hidden_edge_dim": 16,
    "n_layers": 4,
    "n_heads": 8,
    "tanh_clip": 10.0,
    "negative_slope": 0.2,
    "norm": "batch",         # encoder input norm: 'batch' (Lei), 'layer', or 'none'
    "attn": "scalar",        # attention granularity: 'scalar' (locked) or 'channel' (diagnostic)
    "context_extra_dim": 0,  # extra dynamic decoder-context scalars (CVRPTW clock = 1)
    "edge_feat_dim": 0,      # decoder per-candidate edge features (0 CVRP/A; 2 for B/C/D)
    "edge_bias_mode": "pre_tanh",  # 'pre_tanh' (locked) or 'post_tanh' (diagnostic)
}


def build_policy(config: dict[str, Any] | None = None) -> CVRPPolicy:
    """Instantiate a CVRPPolicy. Missing keys fall back to DEFAULT_MODEL_CONFIG.

    Old checkpoints whose model_config predates a key (e.g. 'norm') reconstruct
    with the default for that key, so they keep loading.
    """
    cfg = {**DEFAULT_MODEL_CONFIG, **(config or {})}
    encoder = EGATEncoder(
        input_node_dim=cfg["input_node_dim"],
        input_edge_dim=cfg["input_edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        hidden_edge_dim=cfg["hidden_edge_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        negative_slope=cfg["negative_slope"],
        norm=cfg["norm"],
        attn=cfg["attn"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPPolicy(encoder, decoder)


def model_config_from(**overrides: Any) -> dict[str, Any]:
    """Return a full model config dict with the given overrides applied."""
    return {**DEFAULT_MODEL_CONFIG, **overrides}


# CVRPTW config: 6-dim node features [x,y,demand,earliest,latest,service] and a
# decoder clock scalar. Everything else inherits the certified CVRP settings.
DEFAULT_CVRPTW_CONFIG: dict[str, Any] = {
    **DEFAULT_MODEL_CONFIG,
    "input_node_dim": 6,
    "context_extra_dim": 1,
}


# Variants B/C/D: the encoder edge feature becomes the horizon-averaged
# (mu_bar, sigma_bar^2) from the Edge Bank instead of a single distance, so
# input_edge_dim goes 1 -> 2. NOTHING else changes: fc_edge.weight is the only
# parameter in the encoder whose shape moves ((16, 1) -> (16, 2)). Existing defaults
# are untouched, so DEFAULT_MODEL_CONFIG / DEFAULT_CVRPTW_CONFIG checkpoints keep
# reconstructing their original architecture.
DEFAULT_CVRPTW_STOCH_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_CONFIG,
    "input_edge_dim": 2,     # encoder: horizon-averaged (mu_bar, sigma_bar^2)
    "edge_feat_dim": 2,      # decoder: clock-refreshed (mu, sigma^2)(tau_t)
}


def build_cvrptw_policy(config: dict[str, Any] | None = None) -> "CVRPTWPolicy":
    """Instantiate a CVRPTWPolicy (6-dim node features + decoder clock scalar)."""
    cfg = {**DEFAULT_CVRPTW_CONFIG, **(config or {})}
    encoder = EGATEncoder(
        input_node_dim=cfg["input_node_dim"],
        input_edge_dim=cfg["input_edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        hidden_edge_dim=cfg["hidden_edge_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        negative_slope=cfg["negative_slope"],
        norm=cfg["norm"],
        attn=cfg["attn"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWPolicy(encoder, decoder)


def build_cvrptw_stoch_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWStochPolicy":
    """Instantiate a CVRPTWStochPolicy (variant B: 2-dim edge features in both stages).

    Same certified encoder/decoder settings as variant A; only the edge dimensions move.
    Variant D reuses this builder and adds the shared gate scalar on top.
    """
    cfg = {**DEFAULT_CVRPTW_STOCH_CONFIG, **(config or {})}
    encoder = EGATEncoder(
        input_node_dim=cfg["input_node_dim"],
        input_edge_dim=cfg["input_edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        hidden_edge_dim=cfg["hidden_edge_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        negative_slope=cfg["negative_slope"],
        norm=cfg["norm"],
        attn=cfg["attn"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWStochPolicy(encoder, decoder)