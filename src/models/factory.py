from __future__ import annotations

from typing import Any

from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder
from src.models.policy import CVRPPolicy
from src.models.cvrptw_policy import CVRPTWPolicy

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
    )
    return CVRPTWPolicy(encoder, decoder)