"""Controls for the TRAINING-TIME gate: what is it about sigma^2 that teaches the policy?

Finding being explained (3 seeds, omega = 1): D-prime trained WITH the monotone gate and
evaluated with the gate OFF (lambda = 0, "M") beats the identical network trained with no
gate at all ("M0"), at every variance level. The gate helps as a TRAINING signal and is
useless (lambda = 0 always selected) as a DECISION rule.

Each control below is D-prime with ONE change: the quantity the gate penalises. Everything
else -- architecture, initialisation, optimizer group, the reward -- is identical, and the
gate is matched in MAGNITUDE per instance, so only the PATTERN of the penalty differs:

    signal    what the gate penalises                          what it keeps
    ------    -----------------------------------------------  -------------------------
    sigma2    the true variance (= variant D-prime, "M")        magnitude, contrast, place
    shuffle   the instance's own sigma^2 values, randomly      magnitude, contrast;
              permuted among its roads (symmetric)              WRONG place
    const     the instance's mean sigma^2 on every road         magnitude only; no contrast
    dist      mu^2, rescaled to the instance's mean sigma^2     contrast on the LENGTH
                                                                pattern

Reading (pre-registered): shuffle ~ const ~ M0 and M better => the correct risk LOCATION
is what teaches (information). const ~ shuffle ~ M => any logit perturbation regularises.
dist ~ M => the gate is a disguised length penalty.

Invariants kept by every signal (asserted in tests/test_gate_controls.py):
  * the diagonal (self-loops) is exactly 0, as in the real Edge Bank;
  * the per-instance mean over real edges and time bins equals that of sigma^2;
  * the result is symmetric in (i, j), like the bank;
  * the environment is NEVER touched: travel times, masking and the reward still use the
    true Edge Bank. Only what the gate reads changes.

Shuffling draws from a PRIVATE CPU generator, so it does not consume the global RNG: the
training-instance stream is the same as D-prime's for the same --seed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.data.graph_build import EncoderInputs
from src.data.simulator import MU, SIGMA2, gather_from, horizon_average
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.cvrptw_gated_policy import DEFAULT_GATE_INIT
from src.models.cvrptw_gateonly_policy import CVRPTWGateOnlyPolicy
from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder

GATE_SIGNALS = ("sigma2", "shuffle", "const", "dist")   # "sigma2" = plain D-prime
CONTROL_SIGNALS = GATE_SIGNALS[1:]
_EPS = 1e-12


def softplus_inverse(lam: float) -> float:
    """w such that softplus(w) == lam. lam must be > 0 (softplus never reaches 0)."""
    if not (math.isfinite(lam) and lam > 0.0):
        raise ValueError(f"lambda must be finite and > 0, got {lam}")
    return math.log(math.expm1(lam))


def _offdiag(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """1 on real edges, 0 on self-loops. (1, N, N, 1)."""
    return (1.0 - torch.eye(n, device=device, dtype=dtype)).view(1, n, n, 1)


def _offdiag_mean(x: Tensor) -> Tensor:
    """Per-instance mean over real edges and time bins. x: (B, N, N, K) -> (B,)."""
    _, n, _, k = x.shape
    off = _offdiag(n, x.device, x.dtype)                               # (1, N, N, 1)
    return (x * off).sum(dim=(1, 2, 3)) / (off.sum() * k)


def gate_signal_tensor(
    bank: Tensor,                              # (B, N, N, K, 2) the true Edge Bank
    signal: str,
    generator: torch.Generator | None = None,  # CPU generator for "shuffle"
) -> Tensor:
    """The quantity the gate penalises instead of sigma^2. -> (B, N, N, K)."""
    if bank.dim() != 5 or bank.size(-1) != 2:
        raise ValueError(f"bank must be (B, N, N, K, 2), got {tuple(bank.shape)}")
    sigma2 = bank[..., SIGMA2]                                         # (B, N, N, K)
    b, n, _, k = sigma2.shape
    off = _offdiag(n, sigma2.device, sigma2.dtype)                     # (1, N, N, 1)

    if signal == "sigma2":
        return sigma2
    if signal == "const":
        c = _offdiag_mean(sigma2)                                      # (B,)
        return (c.view(b, 1, 1, 1) * off).expand(b, n, n, k).contiguous()
    if signal == "dist":
        mu2 = bank[..., MU].square()                                   # (B, N, N, K)
        scale = _offdiag_mean(sigma2) / _offdiag_mean(mu2).clamp_min(_EPS)  # (B,)
        return mu2 * scale.view(b, 1, 1, 1) * off
    if signal == "shuffle":
        iu = torch.triu_indices(n, n, offset=1, device=sigma2.device)  # (2, P)
        p = iu.size(1)
        vals = sigma2[:, iu[0], iu[1], :]                              # (B, P, K)
        keys = torch.rand(b, p, generator=generator)                   # (B, P) on CPU
        perm = keys.argsort(dim=1).to(sigma2.device)                   # (B, P)
        shuffled = vals.gather(1, perm.unsqueeze(-1).expand(b, p, k))  # (B, P, K)
        out = torch.zeros_like(sigma2)
        out[:, iu[0], iu[1], :] = shuffled
        out[:, iu[1], iu[0], :] = shuffled                             # keep symmetry
        return out
    raise ValueError(f"unknown gate signal {signal!r}; use one of {GATE_SIGNALS}")


class CVRPTWGateControlPolicy(CVRPTWGateOnlyPolicy):
    """D-prime whose gate penalises a CONTROL signal instead of the true sigma^2."""

    def __init__(
        self,
        encoder: EGATEncoder,
        decoder: PointerDecoder,
        gate_init: float = DEFAULT_GATE_INIT,
        gate_signal: str = "shuffle",
    ) -> None:
        super().__init__(encoder, decoder, gate_init=gate_init)
        if gate_signal not in CONTROL_SIGNALS:
            raise ValueError(
                f"gate_signal must be one of {CONTROL_SIGNALS} (use variant D-prime for "
                f"the true sigma^2), got {gate_signal!r}"
            )
        self.gate_signal = gate_signal
        # Private generator: shuffling must not move the global RNG stream. Seeded from
        # the training seed WITHOUT drawing from it (initial_seed() consumes nothing).
        self._perm_gen = torch.Generator().manual_seed(
            (torch.initial_seed() + 1) % (2**63)
        )

    def _gate_bank(self, env: CVRPTWStochEnv) -> Tensor:
        """(B_inst, N, N, K, 2) = (true mu, control signal). Built once per env.

        Cached ON the env, keyed by the bank object, so the encoder and every decode step
        of one rollout see the SAME shuffle. A new rollout builds a new env -> new draw.
        """
        cache = getattr(env, "_gate_control_cache", None)
        if cache is not None and cache[0] is env.bank and cache[1] == self.gate_signal:
            return cache[2]
        gate = gate_signal_tensor(env.bank, self.gate_signal, self._perm_gen)
        gbank = torch.stack([env.bank[..., MU], gate], dim=-1)          # (B, N, N, K, 2)
        env._gate_control_cache = (env.bank, self.gate_signal, gbank)
        return gbank

    def _encoder_gate_penalty(
        self, env: CVRPTWStochEnv, enc: EncoderInputs
    ) -> Tensor:
        """softplus(w) * horizon-averaged control signal per edge. (num_edges,)."""
        env = self._require_stoch_env(env)
        moments = horizon_average(self._gate_bank(env))                # (B_inst, N, N, 2)
        signal = moments[env.instance_idx][..., SIGMA2]                # (B*S, N, N)
        flat = signal.reshape(-1)                                      # (E,)
        if flat.numel() != enc.edge_index.size(1):
            raise ValueError(
                f"gate penalty has {flat.numel()} entries but the graph has "
                f"{enc.edge_index.size(1)} edges"
            )
        return self.gate_strength() * flat

    def _gate_penalty(
        self, env: CVRPTWStochEnv, edge_feat: Tensor | None
    ) -> Tensor:
        """softplus(w) * control signal at the current node and bin. (B*S, N)."""
        env = self._require_stoch_env(env)
        moments = gather_from(
            self._gate_bank(env), env.current_node, env.current_bin(), env.instance_idx
        )                                                              # (B*S, N, 2)
        return self.gate_strength() * moments[..., SIGMA2]