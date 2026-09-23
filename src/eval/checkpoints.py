"""Rebuild a trained stochastic-CVRPTW policy from a checkpoint.

Shared by scripts/check_trained.py and scripts/compare_variants.py so both entry points
reconstruct a policy exactly the same way -- a comparison between two variants is only
meaningful if neither was rebuilt slightly differently from the other.

The variant is inferred from the SAVED WEIGHTS, not from a flag: a variant-D checkpoint
carries the gate scalar `gate_w`, a variant-B one does not. That way an older checkpoint
without a 'variant' key still loads, and a mislabelled one cannot be silently rebuilt as
the wrong architecture (load_state_dict(strict=True) would refuse anyway).

D and D-prime both carry `gate_w`, so they are separated by the WIDTH of the decoder's
per-candidate edge map: D reads (mu, sigma^2) and has a 1x2 `w_edge`; D-prime reads mu
alone and has a 1x1 one, because its only path from variance to a decision is the gate.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from src.data.simulator import SimulatorConfig
from src.models.factory import (
    build_cvrptw_gated_policy,
    build_cvrptw_gateonly_policy,
    build_cvrptw_sed2am_policy,
    build_cvrptw_stoch_policy,
)
from src.train.pomo_stoch import GATE_PARAM_NAME, SED_GATE_TAG

_REQUIRED = ("model_state", "model_config", "sim_config", "tardiness_weight")


class LoadedPolicy(NamedTuple):
    """A trained policy plus everything needed to reproduce its training setting."""

    policy: torch.nn.Module
    sim_cfg: SimulatorConfig      # the simulator it was trained under
    blob: dict                    # the raw checkpoint (instance dials, val stats, ...)
    variant: str                  # 'B', 'C', 'D' or "D'", inferred from the weights


def _decoder_edge_width(state: dict, model_config: dict) -> int:
    """Number of edge features the decoder's per-candidate map consumes.

    Read from the weight itself so the architecture, not a label, decides. Falls back to
    the saved config for a checkpoint whose decoder has no edge map at all.
    """
    for key, tensor in state.items():
        if key.endswith("w_edge.weight"):
            return int(tensor.shape[1])           # (1, edge_feat_dim)
    return int(model_config.get("edge_feat_dim", 0))


def load_stoch_policy(path: str, device: torch.device | str = "cpu") -> LoadedPolicy:
    """Load a checkpoint written by scripts/train_stoch.py. Returns it in eval mode.

    eval mode is set here deliberately: with BatchNorm still in train mode, an OOD
    variance shift would be partly absorbed by re-estimated running statistics, damping
    the very effect being measured.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for key in _REQUIRED:
        if key not in blob:
            raise KeyError(
                f"{path} has no '{key}'; it does not look like a stochastic-CVRPTW "
                f"checkpoint written by scripts/train_stoch.py"
            )
    state = blob["model_state"]
    # Variant is inferred from the WEIGHTS: D and D-prime carry the single scalar `gate_w`
    # (and differ in how wide the decoder's edge map is), C carries the SED2AM-style gate
    # parameters, B carries neither.
    if GATE_PARAM_NAME in state:
        if _decoder_edge_width(state, blob["model_config"]) == 1:
            variant, builder = "D'", build_cvrptw_gateonly_policy
        else:
            variant, builder = "D", build_cvrptw_gated_policy
    elif any(SED_GATE_TAG in k for k in state):
        variant, builder = "C", build_cvrptw_sed2am_policy
    else:
        variant, builder = "B", build_cvrptw_stoch_policy
    policy = builder(blob["model_config"]).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return LoadedPolicy(policy, SimulatorConfig(**blob["sim_config"]), blob, variant)