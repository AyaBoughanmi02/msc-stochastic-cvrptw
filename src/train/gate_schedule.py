"""WHEN the gate is active during training (Stage 5, mechanism tests).

    constant  gate on for the whole run (variant D-prime as trained so far)
    early     gate on for the first `off_frac` of the steps, then exactly off
    anneal    gate strength multiplied by a factor falling linearly from 1 to 0 at
              `off_frac` of the steps, then exactly off

Why: if the gate helps by shaping WHICH routes are sampled while the policy is still
forming (the exploration hypothesis), "early" should recover most of M's advantage over
M0. "anneal" is also the cleanest candidate METHOD: training ends in exactly the
condition the policy is deployed in (no gate).
"""

from __future__ import annotations

GATE_SCHEDULES = ("constant", "early", "anneal")


def gate_scale(step: int, total_steps: int, schedule: str, off_frac: float | None) -> float:
    """Multiplier on softplus(w) at training step `step` (1-based)."""
    if schedule == "constant":
        return 1.0
    if schedule not in GATE_SCHEDULES:
        raise ValueError(f"unknown gate schedule {schedule!r}; use one of {GATE_SCHEDULES}")
    if off_frac is None or not 0.0 < off_frac <= 1.0:
        raise ValueError(f"schedule {schedule!r} needs off_frac in (0, 1], got {off_frac}")
    if total_steps < 1 or step < 1:
        raise ValueError(f"steps must be >= 1, got step={step}, total={total_steps}")
    t_off = max(1, round(off_frac * total_steps))
    if step >= t_off:
        return 0.0
    if schedule == "early":
        return 1.0
    return 1.0 - step / t_off                                          # anneal