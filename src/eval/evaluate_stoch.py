"""Stochastic CVRPTW evaluation: best-of-N greedy, 8x augmentation, scenario averaging.

Kept separate from src/eval/evaluate.py so the certified CVRP evaluation path is not
touched; `augment_coords_8` is imported from there rather than duplicated.

Three things differ from the deterministic case, and each one is a correctness question
rather than a convenience:

1. THE BANK IS REBUILT ON THE AUGMENTED COORDINATES, not transformed. The Edge Bank is a
   pure function of coords, so rebuilding is correct by construction. It also happens to
   be invariant: the 8 dihedral maps are affine isometries of the unit square, so pairwise
   distances, midpoints and the centroid all map consistently and both `phase_ij` and the
   reliability field come out unchanged (to float error). That invariance is asserted in
   the tests -- it means augmentation changes what the POLICY sees without changing the
   travel-time model it is being judged on.

2. ALL 8 AUGMENTATIONS SHARE ONE NOISE FIELD. The dihedral maps move coordinates but do
   not permute node indices, so edge (i, j) is the same edge in every augmentation and can
   legitimately be given the same realised travel time. Without this, "best over 8
   augmentations" would partly be selecting the luckiest weather rather than the best
   route -- the same argument that makes common random numbers necessary across POMO
   starts.

3. ONE DAY IS NOT A MEASUREMENT. Selecting the minimum over starts x augmentations on a
   single noise draw is optimistic, because the selection sees the noise. `scenario_seeds`
   repeats the whole evaluation over independent days and averages, which is what a
   headline number (in particular the OOD variance-shift result) should report.

Selection is on the PENALISED cost; the reported distance/tardiness are those of the
selected trajectory, i.e. of the route the policy would actually run.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SimulatorConfig, build_edge_bank, draw_noise
from src.eval.evaluate import augment_coords_8
from src.train.pomo import build_starts
from src.train.pomo_stoch import assert_ood_eval_mode, build_stoch_env

_COMPONENTS = ("distance", "tardiness", "late_count")


def _augment(data: CVRPTWData) -> CVRPTWData:
    """Stack the 8 dihedral symmetries along the batch axis. (B, ...) -> (8B, ...).

    Only the coordinates move; demand, windows, service and the node INDEXING are
    untouched, which is what lets the noise field be shared across augmentations.
    """
    b, n, _ = data.coords.shape
    coords = augment_coords_8(data.coords).reshape(8 * b, n, 2)      # (8B, N, 2)
    rep = lambda t: t.unsqueeze(0).expand(8, *t.shape).reshape(8 * b, *t.shape[1:])  # noqa: E731
    return CVRPTWData(
        coords=coords,
        demand=rep(data.demand),                                     # (8B, N)
        capacity=data.capacity,
        time_windows=rep(data.time_windows),                         # (8B, N, 2)
        service_time=rep(data.service_time),                         # (8B, N)
        shift_end=data.shift_end,
    )


@torch.no_grad()
def _one_scenario(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float,
    n_starts: int,
    augment: bool,
    noise: Tensor,                 # (B, N, N, K) the realised day, drawn by the caller
) -> dict[str, Tensor]:
    """Best trajectory per instance on ONE realised day. Returns (B,) tensors."""
    b, n, _ = data.coords.shape
    dev = data.coords.device
    n_aug = 8 if augment else 1

    if augment:
        # Tile the SAME day across the 8 augmentations: the maps move coordinates but do
        # not permute node indices, so edge (i, j) is the same edge in each of them.
        data = _augment(data)
        noise = noise.unsqueeze(0).expand(8, *noise.shape).reshape(8 * b, n, n, -1)

    env = build_stoch_env(
        data, sim_cfg, n_starts, tardiness_weight, noise=noise,
    )
    starts = build_starts(n_aug * b, n_starts, dev)                  # (A*B*S,)
    actions, _, cost = policy(env, decode_type="greedy", force_first=starts)
    parts = env.cost_components(actions)

    # (A*B*S,) -> (B, A*S): trajectories are aug-major, then instance, then start.
    def fold(flat: Tensor) -> Tensor:
        return flat.reshape(n_aug, b, n_starts).permute(1, 0, 2).reshape(b, n_aug * n_starts)

    folded_cost = fold(cost)                                         # (B, A*S)
    best = folded_cost.argmin(dim=1, keepdim=True)                   # (B, 1) select on cost
    out = {"cost": folded_cost.gather(1, best).squeeze(1)}           # (B,)
    for key in _COMPONENTS:
        out[key] = fold(parts[key]).gather(1, best).squeeze(1)       # (B,) of the SAME route
    return out


@torch.no_grad()
def evaluate_stoch(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    tardiness_weight: float = 1.0,
    n_starts: int | None = None,
    augment: bool = True,
    eval_batch_size: int | None = None,   # chunk instances to bound peak memory
    scenario_seeds: Sequence[int] = (0,),
) -> dict[str, Tensor]:
    """Per-instance evaluation averaged over independent days. Returns (B,) tensors.

    Keys: 'cost' (penalised, the selection criterion), 'distance', 'tardiness',
    'late_count' -- the last three belonging to the selected trajectory.

    The policy must already be in eval mode; this raises otherwise (see below).

    eval_batch_size chunks the instance axis exactly as the CVRP evaluator does; the set
    expands by augment (x8) and n_starts (xN), so on 4 GB the single-pass forward can OOM.
    Chunking is EXACTLY result-preserving: the noise field for a scenario is drawn once for
    the whole set and then sliced, so an instance sees the same day whatever the chunk
    boundaries are. (Drawing per chunk would have silently made the chunk size an
    experimental variable.)
    """
    if len(scenario_seeds) == 0:
        raise ValueError("scenario_seeds must be non-empty")
    # Deliberately does NOT call policy.eval() for you. Flipping the mode here would make
    # the assertion below unfireable, and the mistake it guards against -- evaluating an
    # OOD shift while BatchNorm is still updating its running statistics, which damps the
    # very shift being measured -- happens at the CALL SITE, in a loop that was training a
    # moment earlier. Raise there rather than paper over it here.
    assert_ood_eval_mode(policy)

    b, n, _ = data.coords.shape
    starts = (n - 1) if n_starts is None else n_starts
    if not 1 <= starts <= n - 1:
        raise ValueError(f"n_starts must be in [1, {n - 1}], got {starts}")
    chunk = b if eval_batch_size is None else min(eval_batch_size, b)

    dev = data.coords.device
    totals: dict[str, Tensor] = {}
    for seed in scenario_seeds:
        # Drawn ONCE for the whole set, then sliced -> chunking cannot change the days.
        noise = draw_noise(
            (b, n, n, sim_cfg.n_bins), device=dev, dtype=data.coords.dtype,
            generator=torch.Generator(device=dev).manual_seed(seed),
        )                                                            # (B, N, N, K)
        pieces: list[dict[str, Tensor]] = []
        for i in range(0, b, chunk):
            sl = slice(i, i + chunk)
            piece = CVRPTWData(
                coords=data.coords[sl], demand=data.demand[sl], capacity=data.capacity,
                time_windows=data.time_windows[sl], service_time=data.service_time[sl],
                shift_end=data.shift_end,
            )
            pieces.append(
                _one_scenario(policy, piece, sim_cfg, tardiness_weight, starts,
                              augment, noise[sl])
            )
        for key in pieces[0]:
            joined = torch.cat([p[key] for p in pieces], dim=0)      # (B,)
            totals[key] = joined if key not in totals else totals[key] + joined

    return {k: v / len(scenario_seeds) for k, v in totals.items()}


def bank_is_augmentation_invariant(
    coords: Tensor,                # (B, N, 2)
    sim_cfg: SimulatorConfig,
    atol: float = 1e-5,
) -> bool:
    """Does the Edge Bank survive the 8 dihedral maps unchanged?

    True means augmentation changes what the policy SEES without changing the
    travel-time model it is judged on -- the precondition for taking the best cost
    across augmentations. Exposed (and tested) rather than assumed, because it would
    stop holding if a non-isometric augmentation were ever added.
    """
    reference = build_edge_bank(coords, sim_cfg)                     # (B, N, N, K, 2)
    for variant in augment_coords_8(coords):                         # (B, N, 2) each
        if not torch.allclose(build_edge_bank(variant, sim_cfg), reference, atol=atol):
            return False
    return True