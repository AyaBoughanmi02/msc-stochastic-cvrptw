"""Equal-compute day averaging for the sampled reward (experiment E1 of the reward paper).

A POMO mini-batch normally holds B distinct instances, each driven on ONE realised day.
`replicate_days` instead takes B/K distinct instances and repeats each one K times along
the batch axis. Nothing else changes: `build_stoch_env` draws one noise field over all B
rows, so the K copies of an instance face K independent days, and each (instance, day)
row keeps its own POMO group and shared baseline. The number of rollouts per optimizer
step is unchanged, so the comparison with the hybrid reward is at equal compute: the
sampled reward spends its rollouts on more days instead of more instances.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData


def replicate_days(data: CVRPTWData, k_days: int) -> CVRPTWData:
    """Repeat every instance `k_days` times, contiguously. (B, ...) -> (B * k_days, ...).

    Rows [i * k_days, (i + 1) * k_days) are copies of instance i. Scalars (capacity,
    shift_end) are shared and left untouched. k_days = 1 returns `data` itself, so the
    default training path is bit-identical to before.
    """
    if k_days < 1:
        raise ValueError(f"k_days must be >= 1, got {k_days}")
    if k_days == 1:
        return data

    def rep(t: Tensor) -> Tensor:
        return t.repeat_interleave(k_days, dim=0)

    return data._replace(
        coords=rep(data.coords),              # (B*K, N, 2)
        demand=rep(data.demand),              # (B*K, N)
        time_windows=rep(data.time_windows),  # (B*K, N, 2)
        service_time=rep(data.service_time),  # (B*K, N)
    )
