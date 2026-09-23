"""Guard against comparing checkpoints trained for different numbers of steps.

`train_stoch` rewrites `<out>.pt` whenever validation cost improves, so that file
freezes at whatever step happened to win -- 7,000 for one run, 14,000 for another,
even across two runs of the same length. Comparing a baseline saved at step 7,000
against a candidate saved at step 14,000 confounds the VARIANT with the training
BUDGET, and the confound is invisible in the output. The `<out>_last.pt`
checkpoints are always written at the full budget and are what comparisons should
use.

Checkpoints written before the step-recording patch carry no step count at all.
Those are treated as failures too: an unverifiable budget is not a matched one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def checkpoint_steps(blob: Mapping[str, Any]) -> int | None:
    """Return the training step recorded in a loaded checkpoint, or None if absent.

    Args:
        blob: the dict returned by `torch.load` on a checkpoint (`loaded.blob`).
    """
    for key in ("steps", "step", "global_step"):
        if key in blob and blob[key] is not None:
            return int(blob[key])
    return None


def assert_matched_budget(
    items: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    allow: bool = False,
) -> None:
    """Print each checkpoint's training budget; abort unless every budget matches.

    Args:
        items: (path, blob) for every checkpoint about to be compared -- baselines
            and candidates together, since the budget must match ACROSS variants,
            not just within one.
        allow: downgrade the abort to a warning. Diagnostics only.

    Raises:
        SystemExit: budgets differ or are unrecorded, and `allow` is False.
    """
    steps = [checkpoint_steps(blob) for _, blob in items]  # list[int | None]

    print("TRAINING BUDGET of the checkpoints being compared")
    for (path, _), step in zip(items, steps):
        shown = step if step is not None else "UNKNOWN"
        print(f"  {Path(path).name:32s} steps = {shown}")

    unknown = [Path(p).name for (p, _), s in zip(items, steps) if s is None]
    distinct = sorted({s for s in steps if s is not None})

    if unknown:
        problem = f"no step count recorded in: {', '.join(unknown)}"
    elif len(distinct) > 1:
        problem = f"checkpoints trained for different numbers of steps: {distinct}"
    else:
        print(f"  all at {distinct[0]} steps -- budget matched\n")
        return

    if allow:
        print(f"  WARNING: {problem}")
        print("  continuing anyway (--allow-budget-mismatch): results are CONFOUNDED\n")
        return

    raise SystemExit(
        f"\nERROR: {problem}.\n"
        "  Compare the '_last' checkpoints -- those are written at the full budget.\n"
        "  A best-validation checkpoint freezes at a different step per run, which\n"
        "  makes the variant and the training budget impossible to tell apart.\n"
        "  Pass --allow-budget-mismatch to override deliberately."
    )


def add_budget_argument(parser) -> None:
    """Register --allow-budget-mismatch on an argparse parser."""
    parser.add_argument(
        "--allow-budget-mismatch",
        action="store_true",
        help="compare checkpoints with differing or unrecorded step counts. The "
             "comparison is then confounded by the training budget; diagnostics only",
    )