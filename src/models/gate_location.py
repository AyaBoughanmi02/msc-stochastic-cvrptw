"""WHERE the training-time gate acts: encoder, decoder, or both (variant D-prime only).

Finding being dissected (3 seeds, omega = 1): DpConst -- D-prime whose gate penalises the
instance-mean sigma^2 on every road, i.e. NO risk pattern -- reproduces all of M's gain
over M0. A penalty without a pattern can only act through the architecture, and it acts
differently in the two places D-prime applies it:

  encoder  The control is 0 on the diagonal and c on every real edge, and the encoder
           graph holds all N^2 edges INCLUDING self-loops (the size guard in
           `_encoder_gate_penalty` enforces it). Every node-wise attention softmax
           therefore contains one self-loop and N - 1 real edges, so subtracting
           lambda * c from the real edges is a bonus of lambda * c for a node attending
           to ITSELF, growing with the batch's beta_var.
  decoder  The penalty is the same for every candidate leaving the current node (the
           current node itself is masked). It is subtracted BEFORE 10 * tanh
           (PointerDecoder, edge_bias_mode='pre_tanh'), so it slides every candidate
           along the tanh curve and changes how peaked the policy is: a temperature.
           Added AFTER the tanh it would cancel in the softmax and do exactly nothing.

`restrict_gate` switches ONE of the two hooks off on a built policy, so the gate scalar
w still exists, keeps its optimizer group, and learns from the remaining path only.
Same idiom as the gate schedule and `forced_gate`: an instance attribute shadows the
method, the state dict is untouched, and the checkpoint loader re-applies it from
model_config["gate_location"].

At lambda = 0 the three locations compute bit-identical rollouts (x - 0 == x), so the
lambda = 0 evaluation used for every gate arm does not depend on the location.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.data.graph_build import EncoderInputs
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_gateonly_policy import CVRPTWGateOnlyPolicy
from src.models.cvrptw_riskbias_policy import CVRPTWRiskBiasPolicy

GATE_LOCATIONS: tuple[str, ...] = ("both", "encoder", "decoder")


def _no_encoder_gate(env: CVRPTWStochEnv, enc: EncoderInputs) -> None:
    """Replacement hook: the encoder runs ungated (its gate argument is inert when None)."""
    return None


def _no_decoder_gate(env: CVRPTWStochEnv, edge_feat: Tensor | None) -> None:
    """Replacement hook: the pointer receives no gate penalty."""
    return None


def gate_location_of(policy: nn.Module) -> str:
    """Where `policy`'s gate acts. 'both' for any policy never restricted."""
    return getattr(policy, "gate_location", "both")


def restrict_gate(policy: nn.Module, location: str) -> None:
    """Keep the gate in `location` only ('both' = leave the policy unchanged). In place.

    Only for D-prime and its gate controls. VGT-R is refused: its `_gate_penalty` also
    carries the lateness-risk bias, which switching the decoder hook off would remove too.
    """
    if location not in GATE_LOCATIONS:
        raise ValueError(f"unknown gate location {location!r}; use one of {GATE_LOCATIONS}")
    if not isinstance(policy, CVRPTWGateOnlyPolicy) or isinstance(
        policy, CVRPTWRiskBiasPolicy
    ):
        raise TypeError(
            f"gate location applies to variant D-prime and its gate controls only, got "
            f"{type(policy).__name__}"
        )
    current = gate_location_of(policy)
    if current != "both":
        raise RuntimeError(
            f"policy is already restricted to the {current}; build a fresh one instead of "
            f"stacking restrictions (that would switch the gate off everywhere)"
        )
    if location == "encoder":
        policy._gate_penalty = _no_decoder_gate              # decoder path off
    elif location == "decoder":
        policy._encoder_gate_penalty = _no_encoder_gate      # encoder path off
    policy.gate_location = location