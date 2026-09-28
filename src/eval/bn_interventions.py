"""Inference-time interventions for the Stage-5 mechanism study. No retraining.

WHY THIS FILE EXISTS. scripts/train_stoch.py resamples beta_var ONCE PER MINI-BATCH
(`train_sim_cfg` is a callable that train_step_stoch calls once per mini-batch), and the
policy trains in train mode. So every training batch shared ONE beta, and `bn_edge`
normalised that batch with the batch's OWN statistics -- which depend on that beta. At
evaluation, `bn_edge` uses its running statistics instead: an exponential moving average
(momentum 0.1, i.e. dominated by the last ~20 training batches) over a MIXTURE of betas.

Consequence: the encoder was trained on edge embeddings from which part of the day's
variance LEVEL had been normalised away, and is evaluated on embeddings where it has not.
That is a train/eval discrepancy that exists even IN distribution, and grows out of it.
Variant M's edge feature (mu) does not depend on beta, so M is immune by construction.

Two counterfactuals, each removing ONE candidate cause and nothing else:

  batch_stat_edge_norm(policy, stats)
      Evaluate with `bn_edge` normalising by the statistics of a batch drawn AT THE
      EVALUATED beta_var -- exactly what BatchNorm did in training. Implemented by
      swapping the running buffers (the module stays in eval mode), so the result does
      NOT depend on --eval-batch-size, and nothing else in the network changes. This is
      AdaBN (Li et al., "Revisiting Batch Normalization for Practical Domain
      Adaptation", 2016/2018; Schneider et al., NeurIPS 2020). It uses an unlabelled
      batch at the test variance level, so it is a DIAGNOSTIC, not a deployable method.

  scaled_decoder_variance(policy, factor)
      Multiply the sigma^2 column the POINTER reads by a constant. The env's Edge Bank is
      untouched, so the simulated day -- and therefore the cost -- is unchanged; only what
      the policy is SHOWN changes. With factor = beta_ref / beta_eval the pointer sees the
      variance scale of the training midpoint. The decoder path has no normalisation
      layer, so this isolates plain linear extrapolation through `w_edge`.

Both are context managers that restore the model bit-exactly on exit (including on an
exception) and never write to a checkpoint.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SIGMA2, SimulatorConfig
from src.train.pomo_stoch import build_stoch_env


class EdgeNormStats(NamedTuple):
    """Per-channel statistics of the pre-normalisation edge embedding fc_edge(edge_attr)."""

    mean: Tensor  # (C,)  C = hidden_edge_dim
    var: Tensor   # (C,)  BIASED variance: what BatchNorm divides by in train mode


def _bn_edge(policy: nn.Module) -> nn.BatchNorm1d:
    """The encoder's edge BatchNorm, with a crash-not-silent type and state check."""
    bn = policy.encoder.bn_edge
    if not isinstance(bn, nn.BatchNorm1d):
        raise TypeError(
            f"encoder.bn_edge is {type(bn).__name__}, not BatchNorm1d; the train/eval "
            f"discrepancy under study only exists for norm='batch'"
        )
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("bn_edge has track_running_stats=False; nothing to intervene on")
    return bn


@torch.no_grad()
def pre_norm_edge_features(
    policy: nn.Module, data: CVRPTWData, sim_cfg: SimulatorConfig
) -> Tensor:
    """fc_edge output for every edge of `data` under `sim_cfg`. -> (B*N*N, C).

    Built through the policy's OWN `_encoder_inputs` hook, so B gets (mu_bar, sigma_bar^2)
    and D'/M gets mu_bar alone -- exactly the tensor each network normalises. One start
    per instance is enough: POMO replicas duplicate edges and leave mean/variance unchanged.
    """
    env = build_stoch_env(
        data, sim_cfg, n_starts=1, tardiness_weight=0.0, reward_mode="analytic"
    )
    enc = policy._encoder_inputs(env)                         # edge_attr: (B*N*N, F)
    return policy.encoder.fc_edge(enc.edge_attr)              # (B*N*N, C)


@torch.no_grad()
def edge_norm_stats(
    policy: nn.Module, data: CVRPTWData, sim_cfg: SimulatorConfig
) -> EdgeNormStats:
    """Batch statistics train-mode BatchNorm would compute on `data` at `sim_cfg`."""
    x = pre_norm_edge_features(policy, data, sim_cfg)         # (E, C)
    return EdgeNormStats(mean=x.mean(dim=0), var=x.var(dim=0, correction=0))


@contextmanager
def batch_stat_edge_norm(policy: nn.Module, stats: EdgeNormStats) -> Iterator[None]:
    """Temporarily make eval-mode bn_edge normalise with `stats`. Restores on exit.

    The module must be in EVAL mode: in train mode BatchNorm ignores its running buffers
    entirely, so swapping them would silently do nothing (and would be overwritten).
    """
    bn = _bn_edge(policy)
    if bn.training:
        raise RuntimeError(
            "bn_edge is in train mode; call policy.eval() first -- in train mode the "
            "running buffers are ignored and this intervention would be a silent no-op"
        )
    if stats.mean.shape != bn.running_mean.shape or stats.var.shape != bn.running_var.shape:
        raise ValueError(
            f"stats have shapes {tuple(stats.mean.shape)}/{tuple(stats.var.shape)}, "
            f"bn_edge expects {tuple(bn.running_mean.shape)}"
        )
    saved_mean = bn.running_mean.clone()                      # (C,)
    saved_var = bn.running_var.clone()                        # (C,)
    try:
        with torch.no_grad():
            bn.running_mean.copy_(stats.mean.to(bn.running_mean))
            bn.running_var.copy_(stats.var.to(bn.running_var))
        yield
    finally:
        with torch.no_grad():
            bn.running_mean.copy_(saved_mean)
            bn.running_var.copy_(saved_var)


@contextmanager
def scaled_decoder_variance(policy: nn.Module, factor: float) -> Iterator[None]:
    """Temporarily multiply the sigma^2 column of the decoder's edge feature by `factor`.

    Only what the POINTER is shown changes; the env still simulates the real day. The
    mu column is multiplied by exactly 1.0, so it is bit-identical. Patched on the
    instance, the same idiom as `forced_gate` in scripts/compare_variants.py.

    NOTE for variant D (not used here): D's gate reads the same tensor, so this would
    rescale the gate's input too.
    """
    if not (math.isfinite(factor) and factor > 0.0):
        raise ValueError(f"factor must be finite and > 0, got {factor}")
    if policy.decoder.edge_feat_dim <= SIGMA2:
        raise ValueError(
            f"the decoder reads {policy.decoder.edge_feat_dim} edge feature(s), i.e. no "
            f"sigma^2 column; this intervention only applies to variant B"
        )
    original = policy._decoder_edge_feat

    def scaled(env) -> Tensor:
        feat = original(env)                                   # (B*S, N, F)
        scale = torch.ones(feat.size(-1), device=feat.device, dtype=feat.dtype)  # (F,)
        scale[SIGMA2] = factor
        return feat * scale                                    # (B*S, N, F)

    policy._decoder_edge_feat = scaled
    try:
        yield
    finally:
        policy._decoder_edge_feat = original


@torch.no_grad()
def normalisation_drift(
    policy: nn.Module, data: CVRPTWData, sim_cfg: SimulatorConfig
) -> dict[str, float]:
    """How far eval-mode bn_edge output is from what train-mode BN gives on the same batch.

    Per channel c, with x = fc_edge(edge_attr):
        z_train = (x - batch_mean) / sqrt(batch_var + eps)     what training produced
        z_eval  = (x - running_mean) / sqrt(running_var + eps) what evaluation produces
    Channels are weighted by gamma_c^2 (bn_edge.weight), because a channel the network
    scaled to ~0 cannot matter downstream.

    Returns
        drift        gamma-weighted RMS of (z_eval - z_train), in normalised units.
                     0 means eval sees exactly what training saw.
        loc_shift    gamma-weighted RMS of (batch_mean - running_mean) / running_sd.
        scale_ratio  gamma-weighted mean of batch_sd / running_sd (1 = no scale change).
    """
    bn = _bn_edge(policy)
    x = pre_norm_edge_features(policy, data, sim_cfg)          # (E, C)
    b_mean = x.mean(dim=0)                                     # (C,)
    b_var = x.var(dim=0, correction=0)                         # (C,)
    r_sd = torch.sqrt(bn.running_var + bn.eps)                 # (C,)
    b_sd = torch.sqrt(b_var + bn.eps)                          # (C,)

    z_train = (x - b_mean) / b_sd                              # (E, C)
    z_eval = (x - bn.running_mean) / r_sd                      # (E, C)
    w = (bn.weight.detach().square() if bn.affine
         else torch.ones_like(b_mean))                         # (C,)
    w_sum = w.sum()

    per_channel = (z_eval - z_train).square().mean(dim=0)      # (C,)
    loc = ((b_mean - bn.running_mean) / r_sd).square()         # (C,)
    return {
        "drift": float(torch.sqrt((w * per_channel).sum() / w_sum)),
        "loc_shift": float(torch.sqrt((w * loc).sum() / w_sum)),
        "scale_ratio": float((w * (b_sd / r_sd)).sum() / w_sum),
    }