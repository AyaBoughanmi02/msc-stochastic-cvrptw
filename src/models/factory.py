from __future__ import annotations

from typing import Any

from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder
from src.models.policy import CVRPPolicy
from src.models.cvrptw_policy import CVRPTWPolicy
from src.models.cvrptw_gated_policy import DEFAULT_GATE_INIT, CVRPTWGatedPolicy
from src.models.cvrptw_gateonly_policy import CVRPTWGateOnlyPolicy
from src.models.cvrptw_meanonly_policy import CVRPTWMeanOnlyPolicy
from src.models.cvrptw_gate_controls import CVRPTWGateControlPolicy
from src.models.cvrptw_riskbias_policy import DEFAULT_RISK_INIT, CVRPTWRiskBiasPolicy
from src.models.cvrptw_sed2am_policy import (
    DEFAULT_SED_GATE_BIAS_INIT,
    CVRPTWSED2AMPolicy,
)
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
    "sigmoid_gate": False,   # variant C only: SED2AM-style post-softmax gate in the encoder
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
        sigmoid_gate=cfg["sigmoid_gate"],
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
        sigmoid_gate=cfg["sigmoid_gate"],
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
        sigmoid_gate=cfg["sigmoid_gate"],
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


# Variant D: variant B plus ONE shared scalar for the monotone variance gate. The
# architecture is otherwise byte-identical to B, which is what makes the ablation worth
# exactly one parameter.
DEFAULT_CVRPTW_GATED_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_STOCH_CONFIG,
    "gate_init": DEFAULT_GATE_INIT,   # softplus(0) = 0.6931; see cvrptw_gated_policy
}


# Variant C: variant B plus a SED2AM-style learned sigmoid gate (encoder + pointer). The
# competitor to variant D's monotone gate; see src/models/cvrptw_sed2am_policy.py.
DEFAULT_CVRPTW_SED2AM_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_STOCH_CONFIG,
    "sigmoid_gate": True,
    "sed_gate_bias_init": DEFAULT_SED_GATE_BIAS_INIT,   # sigmoid(2.0) = 0.881; C starts ~B
}


def build_cvrptw_sed2am_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWSED2AMPolicy":
    """Instantiate a CVRPTWSED2AMPolicy (variant C).

    NOTE: like variant D's scalar, C's gate parameters need their own optimizer group
    (lr 1e-2, no weight decay) or the comparison is unfair -- see build_optimizer in
    src/train/pomo_stoch.py.
    """
    cfg = {**DEFAULT_CVRPTW_SED2AM_CONFIG, **(config or {})}
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
        sigmoid_gate=True,
        sigmoid_gate_bias_init=cfg["sed_gate_bias_init"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWSED2AMPolicy(encoder, decoder, gate_bias_init=cfg["sed_gate_bias_init"])


def build_cvrptw_gated_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWGatedPolicy":
    """Instantiate a CVRPTWGatedPolicy (variant D: variant B + the shared variance gate).

    NOTE: gate_w needs its own optimizer group (higher lr, no weight decay) or the gate
    cannot travel far enough in 800 steps for the experiment to be informative --
    see build_optimizer in src/train/pomo_stoch.py.
    """
    cfg = {**DEFAULT_CVRPTW_GATED_CONFIG, **(config or {})}
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
        sigmoid_gate=cfg["sigmoid_gate"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWGatedPolicy(encoder, decoder, gate_init=cfg["gate_init"])

# Variant D-prime: variant D with sigma^2 removed from every path except the gate, so the
# monotone constraint actually binds. The encoder edge feature is mu_bar alone
# (input_edge_dim 2 -> 1) and the decoder's per-candidate feature is mu alone
# (edge_feat_dim 2 -> 1); the gate still reads the full variance from the Edge Bank.
DEFAULT_CVRPTW_GATEONLY_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_GATED_CONFIG,
    "input_edge_dim": 1,     # encoder: mu_bar only
    "edge_feat_dim": 1,      # decoder: mu only
}


def build_cvrptw_gateonly_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWGateOnlyPolicy":
    """Instantiate a CVRPTWGateOnlyPolicy (variant D-prime).

    Same note as variant D: gate_w needs its own optimizer group (higher lr, no weight
    decay) -- see build_optimizer in src/train/pomo_stoch.py. The gate matters MORE here,
    since it is now the only route from variance to any decision.
    """
    cfg = {**DEFAULT_CVRPTW_GATEONLY_CONFIG, **(config or {})}
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
        sigmoid_gate=cfg["sigmoid_gate"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWGateOnlyPolicy(encoder, decoder, gate_init=cfg["gate_init"])


# Variant M0: the TRUE mean-only control. D-prime's architecture (mu alone in encoder and
# decoder) with NO gate, so sigma^2 influences neither training nor decisions.
DEFAULT_CVRPTW_MEANONLY_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_STOCH_CONFIG,
    "input_edge_dim": 1,     # encoder: mu_bar only
    "edge_feat_dim": 1,      # decoder: mu only
}


def build_cvrptw_meanonly_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWMeanOnlyPolicy":
    """Instantiate a CVRPTWMeanOnlyPolicy (variant M0). No gate, so no gate optimizer group."""
    cfg = {**DEFAULT_CVRPTW_MEANONLY_CONFIG, **(config or {})}
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
        sigmoid_gate=cfg["sigmoid_gate"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWMeanOnlyPolicy(encoder, decoder)


# Gate controls: D-prime whose gate penalises a CONTROL signal (shuffled sigma^2, a
# constant, or rescaled mu^2) instead of the true variance. Same architecture and state
# dict as D-prime; the signal is recorded in model_config["gate_signal"] so the checkpoint
# loader can rebuild the right class. See src/models/cvrptw_gate_controls.py.
DEFAULT_CVRPTW_GATE_CONTROL_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_GATEONLY_CONFIG,
    "gate_signal": "shuffle",
}


def build_cvrptw_gate_control_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWGateControlPolicy":
    """Instantiate a CVRPTWGateControlPolicy (a D-prime mechanism control)."""
    cfg = {**DEFAULT_CVRPTW_GATE_CONTROL_CONFIG, **(config or {})}
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
        sigmoid_gate=cfg["sigmoid_gate"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWGateControlPolicy(
        encoder, decoder, gate_init=cfg["gate_init"], gate_signal=cfg["gate_signal"]
    )

# Variant VGT-R: VGT (D-prime) plus a learnable non-negative lateness-risk bias in the
# decoder, -alpha * P(late at j). Same encoder/decoder shapes as D-prime; the only new
# parameter is the scalar `risk_w` (alpha = softplus(risk_w)), which must sit in the gate
# optimizer group. See src/models/cvrptw_riskbias_policy.py.
DEFAULT_CVRPTW_RISKBIAS_CONFIG: dict[str, Any] = {
    **DEFAULT_CVRPTW_GATEONLY_CONFIG,
    "risk_init": DEFAULT_RISK_INIT,   # softplus^-1(1.0): alpha starts at 1, as in ICAM
    "risk_signal": "sigma2",          # or a control: "const" / "shuffle"
}


def build_cvrptw_riskbias_policy(
    config: dict[str, Any] | None = None,
) -> "CVRPTWRiskBiasPolicy":
    """Instantiate a CVRPTWRiskBiasPolicy (variant VGT-R)."""
    cfg = {**DEFAULT_CVRPTW_RISKBIAS_CONFIG, **(config or {})}
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
        sigmoid_gate=cfg["sigmoid_gate"],
    )
    decoder = PointerDecoder(
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        tanh_clip=cfg["tanh_clip"],
        context_extra_dim=cfg["context_extra_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        edge_bias_mode=cfg["edge_bias_mode"],
    )
    return CVRPTWRiskBiasPolicy(
        encoder, decoder, gate_init=cfg["gate_init"], risk_init=cfg["risk_init"],
        risk_signal=cfg["risk_signal"],
    )