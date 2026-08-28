"""Hand-computed tests for the time-window feasibility mask.

All inputs are constructed by hand so the expected mask is known by arithmetic,
not by trusting the function. Geometry: depot (0,0), c1 (1,0), c2 (0,1), so
depot<->c1 = depot<->c2 = 1.0 and c1<->c2 = sqrt(2). service = 0.2, shift = 4.0.
"""

from __future__ import annotations

import math

import torch

from src.utils.masking import tw_feasibility_mask

_SQRT2 = math.sqrt(2.0)
_SHIFT = 4.0


def test_at_depot_fresh_shift_masks_nothing_extra() -> None:
    # At depot, clock 0: both customers reachable and returnable -> no TW mask.
    base = torch.tensor([[True, False, False]])            # at depot: depot masked
    time = torch.tensor([0.0])
    travel_from_current = torch.tensor([[0.0, 1.0, 1.0]])  # depot -> {depot,c1,c2}
    travel_to_depot = torch.tensor([[0.0, 1.0, 1.0]])
    tw = torch.tensor([[[0.0, _SHIFT], [1.0, 1.5], [1.0, 2.0]]])  # (1,3,2)
    service = torch.tensor([[0.0, 0.2, 0.2]])
    mask = tw_feasibility_mask(base, time, travel_from_current, travel_to_depot,
                               tw, service, _SHIFT)
    assert mask.tolist() == [[True, False, False]]         # unchanged


def test_window_closed_forces_depot_return() -> None:
    # reading_notes example: after serving c1 (clock 1.2 at c1), c2's window has
    # closed -> only the depot stays feasible, forcing a return + clock reset.
    base = torch.tensor([[False, True, False]])            # at c1: depot free, c1 visited
    time = torch.tensor([1.2])
    travel_from_current = torch.tensor([[1.0, 0.0, _SQRT2]])  # c1 -> {depot,c1,c2}
    travel_to_depot = torch.tensor([[0.0, 1.0, 1.0]])
    tw = torch.tensor([[[0.0, _SHIFT], [1.0, 1.5], [1.0, 2.0]]])
    service = torch.tensor([[0.0, 0.2, 0.2]])
    # arrival at c2 = 1.2 + sqrt(2) = 2.614 > latest 2.0 -> (a) closed
    mask = tw_feasibility_mask(base, time, travel_from_current, travel_to_depot,
                               tw, service, _SHIFT)
    assert mask.tolist() == [[False, True, True]]          # only depot feasible


def test_cant_return_within_shift_masks_customer() -> None:
    # In-window but serving would blow the shift on the return leg -> (b).
    base = torch.tensor([[False, False]])                  # depot free, D feasible by capacity
    time = torch.tensor([3.5])
    travel_from_current = torch.tensor([[0.7, 0.5]])       # current -> {depot, D}
    travel_to_depot = torch.tensor([[0.0, 1.0]])           # D -> depot = 1.0
    tw = torch.tensor([[[0.0, _SHIFT], [0.0, _SHIFT]]])    # D window wide open
    service = torch.tensor([[0.0, 0.2]])
    # arrival 4.0 (== latest, in time); depart 4.2; +return 1.0 = 5.2 > 4.0 -> (b)
    mask = tw_feasibility_mask(base, time, travel_from_current, travel_to_depot,
                               tw, service, _SHIFT)
    assert mask.tolist() == [[False, True]]


def test_early_arrival_waits_not_masked() -> None:
    # Arriving before `earliest` is fine (wait), so it must NOT be masked.
    base = torch.tensor([[True, False]])                   # at depot
    time = torch.tensor([0.0])
    travel_from_current = torch.tensor([[0.0, 0.5]])       # depot -> D = 0.5
    travel_to_depot = torch.tensor([[0.0, 0.5]])
    tw = torch.tensor([[[0.0, _SHIFT], [2.0, 3.0]]])       # D opens late at 2.0
    service = torch.tensor([[0.0, 0.2]])
    # arrival 0.5 < earliest 2.0 -> wait to 2.0; 0.5 <= latest 3.0 ok;
    # depart 2.2 + return 0.5 = 2.7 <= 4.0 ok -> feasible
    mask = tw_feasibility_mask(base, time, travel_from_current, travel_to_depot,
                               tw, service, _SHIFT)
    assert mask.tolist() == [[True, False]]


def test_depot_never_time_masked() -> None:
    # Even at an absurd clock, the depot column must stay whatever the base said.
    base = torch.tensor([[False, True]])
    time = torch.tensor([100.0])
    travel_from_current = torch.tensor([[5.0, 5.0]])
    travel_to_depot = torch.tensor([[0.0, 5.0]])
    tw = torch.tensor([[[0.0, _SHIFT], [0.0, 1.0]]])
    service = torch.tensor([[0.0, 0.2]])
    mask = tw_feasibility_mask(base, time, travel_from_current, travel_to_depot,
                               tw, service, _SHIFT)
    assert mask[0, 0].item() is False or mask[0, 0].item() == False  # depot feasible
    assert mask.tolist() == [[False, True]]