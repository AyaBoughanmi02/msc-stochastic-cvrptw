"""Stochastic, time-dependent travel-time simulator (the "Edge Bank") for variants B/C/D.

This is the module that CREATES the uncertainty the variance gate manages. It is kept
strictly separate from the geometric instance (coords / demand / windows): an instance
plus a `SimulatorConfig` deterministically produces an Edge Bank

    bank[b, i, j, k] = (mu, sigma^2) of the travel time on edge (i, j) in time bin k,

from which the env reads planned travel times (`mu`, used for masking) and samples
executed travel times (used to advance the clock). Variant A is the `alpha = 0,
beta_var = 0` corner: `mu == pairwise_distances(coords)` and `sigma^2 == 0` BIT-EXACTLY,
which is what lets the stochastic machinery reduce to the certified deterministic case.

Moment model (the "ripple"), for bin centres tau_k = (k + 0.5) * shift_end / K:

    d_ij            = ||coords_i - coords_j||                     (the Stage-2 seam)
    center          = mean of the instance's node coordinates
    radius_ij       = || midpoint(i, j) - center ||               spatial position of the edge
    phase_ij        = phase_k * radius_ij                         congestion lag (rush hour
                                                                  peaks later away from the
                                                                  centre)
    mu_ij(tau)      = d_ij * (1 + alpha * sin(2*pi*(tau - phase_ij) / T))
    sigma^2_ij(tau) = beta_var * r_ij * mu_ij(tau)^2              heteroscedastic

`alpha < 1` keeps `mu > 0`. With `T == shift_end` the K bin centres are equally spaced
over one full cycle, so `mean_k sin(.) == 0` exactly: the ripple is MEAN-PRESERVING
(`mean_k mu_ij == d_ij` for every edge). That is deliberate -- it keeps the Stage-2
feasibility-by-construction calibration valid.

Reliability field `r_ij` (`eta`, `field_scale`)
-----------------------------------------------
With `eta = 0` (the default) `r_ij == 1` and the model is exactly the locked
`sigma^2 = beta_var * mu^2`. That locked form makes sigma^2 a pointwise deterministic
function of mu, and mean-preservation makes it worse in the horizon average:

    mean_k mu     = d                       (mean_k sin  = 0)
    mean_k mu^2   = d^2 * (1 + alpha^2 / 2) (mean_k sin^2 = 1/2)

so the encoder's (mu_bar, sigma_bar^2) edge feature would lie on a 1-D curve and the
variance gate would degenerate into a learned squared-distance penalty. `eta > 0` adds a
smooth spatial reliability field

    g_ij = exp(-radius_ij^2 / (2 * field_scale^2))     in (0, 1]
    r_ij = 1 + eta * (2 * g_ij - 1)                    in [1 - eta, 1 + eta]

("central corridors are less reliable at equal length"), which decouples sigma^2 from mu
WITHOUT touching mu at all -- so the instance calibration is untouched and `beta_var`
remains the pure magnitude knob for the out-of-distribution experiment.

Sampling
--------
Executed travel time is lognormal with the MOMENTS ABOVE, obtained by moment matching:

    s^2 = log(1 + sigma^2 / mu^2);  m = log(mu) - s^2 / 2;  travel = exp(m + s * Z)

which gives E[travel] = mu and Var[travel] = sigma^2 exactly. Passing (mu, sigma^2)
straight into a Lognormal(loc, scale) would instead treat them as LOG-SPACE parameters --
a silent bug that distorts precisely the high-variance edges the gate targets.

The sampler is split into `draw_noise` (the standard normal Z) and `apply_noise` (the
deterministic transform) so the caller chooses the noise granularity. Sharing one Z field
across an instance's POMO start replicas ("common random numbers") makes every trajectory
face the same realised day, which keeps POMO's shared baseline a comparison of DECISIONS
rather than of luck.

Out of scope (recorded, not hidden): spatially correlated noise, temporal autocorrelation,
asymmetric congestion (the bank is symmetric in (i, j)), real GPS calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.cvrptw_generator import _MAX_RADIUS, pairwise_distances

# Index of each moment along the Edge Bank's last axis.
MU = 0
SIGMA2 = 1
# Floor for mu inside log(): the bank's diagonal is exactly 0 (self-loops, never traversed).
_MU_EPS = 1e-12


@dataclass(frozen=True)
class SimulatorConfig:
    """Parameters of the ripple travel-time model. Frozen so it can be stored in a checkpoint.

    n_bins:      K, number of time bins the shift is discretised into.
    period:      T of the congestion sine. T == shift_end => one mean-preserving cycle/day.
    alpha:       ripple amplitude, 0 <= alpha < 1 (alpha < 1 keeps mu > 0).
    beta_var:    variance scale; sigma^2 = beta_var * r * mu^2. THE OOD MAGNITUDE KNOB.
    phase_k:     spatial congestion-lag coefficient (phase = phase_k * radius).
    shift_end:   per-route shift horizon; must match the generator/env value.
    eta:         reliability-field strength, 0 <= eta < 1. 0.0 == the locked homogeneous
                 model (r == 1) bit-exactly. THE OOD STRUCTURE KNOB.
    field_scale: gaussian width of the reliability field, in coordinate units.
    """

    n_bins: int = 24
    period: float = 4.0
    alpha: float = 0.3
    beta_var: float = 0.1
    phase_k: float = 1.0
    shift_end: float = 4.0
    eta: float = 0.0
    field_scale: float = 0.3

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        if self.period <= 0.0:
            raise ValueError(f"period must be > 0, got {self.period}")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1) to keep mu > 0, got {self.alpha}")
        if self.beta_var < 0.0:
            raise ValueError(f"beta_var must be >= 0, got {self.beta_var}")
        if self.shift_end <= 0.0:
            raise ValueError(f"shift_end must be > 0, got {self.shift_end}")
        if not 0.0 <= self.eta < 1.0:
            raise ValueError(f"eta must be in [0, 1) to keep r > 0, got {self.eta}")
        if self.field_scale <= 0.0:
            raise ValueError(f"field_scale must be > 0, got {self.field_scale}")

    @property
    def bin_width(self) -> float:
        """Duration of one time bin."""
        return self.shift_end / self.n_bins

    @property
    def is_mean_preserving(self) -> bool:
        """True iff the K bin centres span whole cycles, so `mean_k mu_ij == d_ij` exactly.

        Only then does the ripple leave the Stage-2 instance calibration untouched. Changing
        `shift_end` without `period` would break this silently, so callers that rely on it
        (the env, the calibration diagnostics) should assert on this flag.
        """
        cycles = self.shift_end / self.period
        if not math.isclose(cycles, round(cycles), abs_tol=1e-9):
            return False                       # bin centres do not span whole cycles
        c = round(cycles)
        # Aliasing: the K sampled phases must not collapse onto a single point (mean sin),
        # nor onto a half-cycle pair (mean sin^2 = 1/2).
        return c % self.n_bins != 0 and (2 * c) % self.n_bins != 0


def ripple_reachability_floor(cfg: SimulatorConfig, service_time: float) -> float:
    """Smallest `shift_end` for which every customer stays servable under peak congestion.

    Stage 2's floor is `2*sqrt(2) + service` (travel == distance). Peak travel here is
    `(1 + alpha) * d`, so the floor scales by `(1 + alpha)`.
    """
    return (1.0 + cfg.alpha) * 2.0 * _MAX_RADIUS + service_time


def assert_ripple_reachable(cfg: SimulatorConfig, service_time: float) -> None:
    """Crash-not-silent guard: raise if the ripple can make a far-corner customer unservable.

    Stage 2's feasibility-by-construction (and therefore the "plans are always feasible;
    only EXECUTION creates tardiness" story) assumes every customer can be reached and
    returned from within one shift. Raising `alpha` breaks that silently -- rollouts would
    simply fail to serve everyone -- so it is checked explicitly here.
    """
    floor = ripple_reachability_floor(cfg, service_time)
    if cfg.shift_end < floor:
        raise ValueError(
            f"shift_end={cfg.shift_end} is below the ripple reachability floor {floor:.4f} "
            f"(= (1+alpha)*2*sqrt(2)+service_time, alpha={cfg.alpha}); a far-corner customer "
            f"would be unservable at peak congestion"
        )


def build_edge_bank(coords: Tensor, cfg: SimulatorConfig) -> Tensor:
    """Build the Edge Bank of travel-time moments for a batch of instances.

    coords: (B, N, 2) -> bank: (B, N, N, K, 2), last axis (MU, SIGMA2).

    Deterministic given (coords, cfg) -- no RNG. Device and dtype follow `coords`.
    With `alpha == 0` and `beta_var == 0` this returns `mu == pairwise_distances(coords)`
    and `sigma^2 == 0` bit-exactly (variant A).
    """
    if coords.dim() != 3 or coords.size(-1) != 2:
        raise ValueError(f"coords must have shape (B, N, 2), got {tuple(coords.shape)}")
    dev, dtype = coords.device, coords.dtype
    b, n, _ = coords.shape
    k = cfg.n_bins

    d = pairwise_distances(coords)                                   # (B, N, N)

    # Where each edge sits relative to the instance's centre of mass.
    center = coords.mean(dim=1)                                      # (B, 2)
    midpoint = 0.5 * (coords.unsqueeze(2) + coords.unsqueeze(1))     # (B, N, N, 2)
    radius = (midpoint - center.view(b, 1, 1, 2)).norm(dim=-1)       # (B, N, N)

    # Bin centres tau_k = (k + 0.5) * bin_width.
    bin_centres = (
        torch.arange(k, device=dev, dtype=dtype) + 0.5
    ) * cfg.bin_width                                                # (K,)

    # mu_ij(tau_k): sinusoidal congestion ripple, phase-lagged by spatial position.
    phase = cfg.phase_k * radius                                     # (B, N, N)
    angle = (2.0 * math.pi / cfg.period) * (
        bin_centres.view(1, 1, 1, k) - phase.unsqueeze(-1)
    )                                                                # (B, N, N, K)
    mu = d.unsqueeze(-1) * (1.0 + cfg.alpha * torch.sin(angle))      # (B, N, N, K)

    # sigma^2_ij(tau_k) = beta_var * r_ij * mu^2, r == 1 exactly when eta == 0.
    if cfg.eta == 0.0:
        reliability = torch.ones(1, 1, 1, device=dev, dtype=dtype)   # (1, 1, 1) broadcast
    else:
        g = torch.exp(-radius.square() / (2.0 * cfg.field_scale**2))  # (B, N, N) in (0, 1]
        reliability = 1.0 + cfg.eta * (2.0 * g - 1.0)                 # (B, N, N)
    sigma2 = cfg.beta_var * reliability.unsqueeze(-1) * mu.square()   # (B, N, N, K)

    return torch.stack([mu, sigma2], dim=-1)                          # (B, N, N, K, 2)


def horizon_average(bank: Tensor) -> Tensor:
    """Average both moments over the K time bins -> the ENCODER's static edge feature.

    bank: (B, N, N, K, 2) -> (B, N, N, 2), i.e. (mu_bar, sigma_bar^2).

    Note this is `mean_k sigma^2_k` (the average within-bin variance), NOT the variance of
    a bin-mixture (which would add `Var_k(mu)`). The encoder is told "how variable is this
    edge at a typical moment", which is the quantity the gate penalises.
    """
    if bank.dim() != 5 or bank.size(-1) != 2:
        raise ValueError(f"bank must have shape (B, N, N, K, 2), got {tuple(bank.shape)}")
    return bank.mean(dim=3)                                           # (B, N, N, 2)


def bin_of(clock: Tensor, cfg: SimulatorConfig) -> Tensor:
    """Time bin index of a per-trajectory clock. clock: (B,) -> (B,) long, in [0, K-1].

    Clamped at both ends: a sampled travel time can push the clock past `shift_end`
    (that is exactly what produces tardiness), and it must not index out of the bank.
    """
    idx = torch.floor(clock / cfg.bin_width).long()                   # (B,)
    return idx.clamp_(0, cfg.n_bins - 1)                              # (B,)


def lookup_bin(bank: Tensor, bin_idx: Tensor) -> Tensor:
    """Full moment matrix at a per-instance time bin -> the DECODER's edge input.

    bank: (B, N, N, K, 2), bin_idx: (B,) long -> (B, N, N, 2).
    """
    if bin_idx.dim() != 1 or bin_idx.size(0) != bank.size(0):
        raise ValueError(
            f"bin_idx must have shape ({bank.size(0)},), got {tuple(bin_idx.shape)}"
        )
    b, n, _, _, _ = bank.shape
    index = bin_idx.view(b, 1, 1, 1, 1).expand(b, n, n, 1, 2)         # (B, N, N, 1, 2)
    return bank.gather(3, index).squeeze(3)                           # (B, N, N, 2)


def _resolve_instance_idx(
    bank: Tensor,
    node_idx: Tensor,
    bin_idx: Tensor,
    instance_idx: Tensor | None,
) -> Tensor:
    """Validate a gather's index tensors and default `instance_idx` to the identity.

    `instance_idx` maps each ROW of the query (one per trajectory) to a row of the
    Edge Bank. It exists so a POMO batch of B*S trajectories can share a bank of only
    B instances instead of replicating it S times. Values are NOT range-checked here
    (that would force a device sync every step); callers check once at construction.
    """
    if node_idx.dim() != 1:
        raise ValueError(f"node_idx must be 1-D, got shape {tuple(node_idx.shape)}")
    if node_idx.dtype != torch.long:
        raise ValueError(f"node_idx must be int64, got {node_idx.dtype}")
    if bin_idx.shape != node_idx.shape:
        raise ValueError(
            f"bin_idx {tuple(bin_idx.shape)} must match node_idx {tuple(node_idx.shape)}"
        )
    if bin_idx.dtype != torch.long:
        raise ValueError(f"bin_idx must be int64, got {bin_idx.dtype}")
    if instance_idx is None:
        if node_idx.size(0) != bank.size(0):
            raise ValueError(
                f"without instance_idx the query must be instance-aligned: got "
                f"{node_idx.size(0)} rows for a bank of {bank.size(0)} instances"
            )
        return torch.arange(node_idx.size(0), device=bank.device)
    if instance_idx.shape != node_idx.shape:
        raise ValueError(
            f"instance_idx {tuple(instance_idx.shape)} must match node_idx "
            f"{tuple(node_idx.shape)}"
        )
    if instance_idx.dtype != torch.long:
        raise ValueError(f"instance_idx must be int64, got {instance_idx.dtype}")
    return instance_idx


def gather_from(
    bank: Tensor,                        # (B, N, N, K, 2)
    node_idx: Tensor,                    # (M,) long, origin node per query row
    bin_idx: Tensor,                     # (M,) long, time bin per query row
    instance_idx: Tensor | None = None,  # (M,) long into the bank's batch; None = identity
) -> Tensor:
    """Moments FROM one origin node TO every node, at one time bin. -> (M, N, 2).

    This is the env's per-step lookup: only O(M*N*K) is materialised, never the full
    (M, N, N) matrix.
    """
    inst = _resolve_instance_idx(bank, node_idx, bin_idx, instance_idx)
    m, n = node_idx.size(0), bank.size(1)
    rows = bank[inst, node_idx]                                       # (M, N, K, 2)
    index = bin_idx.view(m, 1, 1, 1).expand(m, n, 1, 2)               # (M, N, 1, 2)
    return rows.gather(2, index).squeeze(2)                           # (M, N, 2)


def gather_to(
    bank: Tensor,                        # (B, N, N, K, 2)
    node_idx: Tensor,                    # (M,) long, destination node per query row
    bin_idx: Tensor,                     # (M,) long
    instance_idx: Tensor | None = None,  # (M,) long into the bank's batch
) -> Tensor:
    """Moments FROM every node TO one destination node, at one time bin. -> (M, N, 2).

    Written as an explicit transpose rather than reusing `gather_from` and leaning on
    the bank's symmetry, so it stays correct if directional congestion is ever added.
    """
    inst = _resolve_instance_idx(bank, node_idx, bin_idx, instance_idx)
    m, n = node_idx.size(0), bank.size(1)
    cols = bank.transpose(1, 2)[inst, node_idx]                       # (M, N, K, 2) view
    index = bin_idx.view(m, 1, 1, 1).expand(m, n, 1, 2)               # (M, N, 1, 2)
    return cols.gather(2, index).squeeze(2)                           # (M, N, 2)


def gather_edge(
    bank: Tensor,                        # (B, N, N, K, 2)
    from_idx: Tensor,                    # (M,) long
    to_idx: Tensor,                      # (M,) long
    bin_idx: Tensor,                     # (M,) long
    instance_idx: Tensor | None = None,  # (M,) long into the bank's batch
) -> Tensor:
    """Moments of one traversed edge at one time bin. -> (M, 2)."""
    inst = _resolve_instance_idx(bank, from_idx, bin_idx, instance_idx)
    if to_idx.shape != from_idx.shape:
        raise ValueError(
            f"to_idx {tuple(to_idx.shape)} must match from_idx {tuple(from_idx.shape)}"
        )
    m = from_idx.size(0)
    moments = bank[inst, from_idx, to_idx]                            # (M, K, 2)
    index = bin_idx.view(m, 1, 1).expand(m, 1, 2)                     # (M, 1, 2)
    return moments.gather(1, index).squeeze(1)                        # (M, 2)


def lookup_row(bank: Tensor, from_idx: Tensor, bin_idx: Tensor) -> Tensor:
    """Instance-aligned special case of `gather_from`.

    bank: (B, N, N, K, 2), from_idx: (B,) long, bin_idx: (B,) long -> (B, N, 2).
    """
    return gather_from(bank, from_idx, bin_idx, instance_idx=None)


def draw_noise(
    shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Standard normal noise Z of the given shape, on `device`.

    Kept separate from `apply_noise` so the caller controls noise granularity (fresh per
    traversal, or one shared field per instance = common random numbers across POMO starts).
    `torch.distributions.LogNormal` is deliberately NOT used: its `.sample()` takes no
    `generator`, so reproducibility would depend on global RNG state.
    """
    dev = torch.device(device)
    if generator is not None and generator.device.type != dev.type:
        raise ValueError(
            f"generator is on device type '{generator.device.type}' but noise was requested "
            f"on '{dev.type}'; torch requires them to match"
        )
    return torch.randn(shape, device=dev, dtype=dtype, generator=generator)


def apply_noise(mu: Tensor, sigma2: Tensor, z: Tensor) -> Tensor:
    """Lognormal travel time with mean `mu` and variance `sigma2`, driven by noise `z`.

    All three broadcast against each other; the result has the broadcast shape.

    Moment matching (the correctness crux):
        s^2 = log1p(sigma2 / mu^2)      # log-space variance
        m   = log(mu) - s^2 / 2         # log-space mean
        out = exp(m + s * z)            # E[out] = mu, Var[out] = sigma2
    Degenerate entries (sigma2 == 0, i.e. variant A; or mu == 0, i.e. the self-loop
    diagonal) return `mu` EXACTLY rather than `exp(log(mu))`, so the deterministic case is
    bit-identical rather than merely close.
    """
    mu_safe = mu.clamp_min(_MU_EPS)                                   # keeps log() finite
    s2 = torch.log1p(sigma2.clamp_min(0.0) / mu_safe.square())        # broadcast
    log_mean = mu_safe.log() - 0.5 * s2
    sampled = torch.exp(log_mean + s2.sqrt() * z)
    degenerate = (sigma2 <= 0.0) | (mu <= _MU_EPS)
    # torch.where evaluates both branches, but the clamp above keeps the unused one finite.
    return torch.where(degenerate, mu.expand_as(sampled), sampled)


def sample_travel(
    mu: Tensor,
    sigma2: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Convenience sampler: fresh noise per element. mu, sigma2 -> same (broadcast) shape.

    Use `draw_noise` + `apply_noise` directly when the noise must be shared (e.g. one
    realised travel-time field per instance across POMO starts).
    """
    shape = torch.broadcast_shapes(mu.shape, sigma2.shape)
    z = draw_noise(shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return apply_noise(mu, sigma2, z)