"""Tests for the variant-B pointer: edge term, gate slot, bias mode, diagnostics scratch.

Load-bearing tests:
  - `test_edge_feat_dim_zero_is_byte_identical_to_the_certified_decoder`: the certified
    checkpoints must keep loading, which requires the parameter set to be unchanged when
    the new machinery is switched off.
  - `test_zero_w_edge_reproduces_the_variant_a_pointer`: the zero-init promise -- variant
    B starts as exactly variant A's pointer function.
  - `test_zero_gate_penalty_is_identical_to_none`: the slot variant D will subtract into
    must be provably inert when empty, or B-vs-D stops being a one-term difference.
"""

from __future__ import annotations

import pytest
import torch

from src.models.pointer_decoder import PointerDecoder

_HIDDEN, _HEADS = 64, 8


def _inputs(b: int, n: int, extra: int = 0, edge_dim: int = 0, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    node_emb = torch.rand(b, n, _HIDDEN, generator=gen)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, 0] = True
    return {
        "node_emb": node_emb,
        "graph_emb": node_emb.mean(dim=1),
        "last_node_emb": node_emb[:, 0, :],
        "remaining": torch.rand(b, 1, generator=gen),
        "mask": mask,
        "context_extra": torch.rand(b, extra, generator=gen) if extra else None,
        "edge_feat": torch.rand(b, n, edge_dim, generator=gen) if edge_dim else None,
    }


def _run(dec: PointerDecoder, inp: dict, **over) -> torch.Tensor:
    kwargs = {**inp, **over}
    node_emb = kwargs.pop("node_emb")
    return dec(dec.precompute(node_emb), kwargs.pop("graph_emb"),
               kwargs.pop("last_node_emb"), kwargs.pop("remaining"), kwargs.pop("mask"),
               **kwargs)


# ------------------------------------------------------------------ the parameter set


def test_edge_feat_dim_zero_is_byte_identical_to_the_certified_decoder() -> None:
    torch.manual_seed(0)
    certified = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, context_extra_dim=1)
    assert not hasattr(certified, "w_edge")
    assert set(certified.state_dict()) == {
        "wq_context.weight", "wk_glimpse.weight", "wv_glimpse.weight",
        "w_out.weight", "wk_pointer.weight",
    }
    with_edges = PointerDecoder(
        hidden_dim=_HIDDEN, n_heads=_HEADS, context_extra_dim=1, edge_feat_dim=2
    )
    assert set(with_edges.state_dict()) - set(certified.state_dict()) == {"w_edge.weight"}
    assert with_edges.w_edge.weight.shape == (1, 2)
    # 2 extra parameters, nothing else moves.
    assert sum(p.numel() for p in with_edges.parameters()) == (
        sum(p.numel() for p in certified.parameters()) + 2
    )


def test_w_edge_is_zero_initialised() -> None:
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    assert torch.equal(dec.w_edge.weight, torch.zeros(1, 2))
    dec.reset_parameters()
    assert torch.equal(dec.w_edge.weight, torch.zeros(1, 2))   # survives a re-init


# --------------------------------------------------------------- the inertness chain


def test_zero_w_edge_reproduces_the_variant_a_pointer() -> None:
    """With w_edge zeroed the output must be BITWISE equal to having no edge term."""
    torch.manual_seed(1)
    with_edges = PointerDecoder(
        hidden_dim=_HIDDEN, n_heads=_HEADS, context_extra_dim=1, edge_feat_dim=2
    )
    without = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, context_extra_dim=1)
    without.load_state_dict(
        {k: v for k, v in with_edges.state_dict().items() if k != "w_edge.weight"},
        strict=True,
    )
    inp = _inputs(4, 7, extra=1, edge_dim=2, seed=1)
    a = _run(with_edges, inp)
    b = _run(without, {**inp, "edge_feat": None})
    assert torch.equal(a, b)


def test_a_trained_w_edge_actually_changes_the_output() -> None:
    """Sanity on the other side: the term is inert only because it is zero."""
    torch.manual_seed(2)
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    inp = _inputs(4, 7, edge_dim=2, seed=2)
    before = _run(dec, inp)
    with torch.no_grad():
        dec.w_edge.weight.fill_(0.7)
    assert not torch.equal(before, _run(dec, inp))


@pytest.mark.parametrize("mode", ["pre_tanh", "post_tanh"])
def test_zero_gate_penalty_is_identical_to_none(mode: str) -> None:
    """The slot variant D subtracts into must be inert when empty, in BOTH modes."""
    torch.manual_seed(3)
    dec = PointerDecoder(
        hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2, edge_bias_mode=mode
    )
    with torch.no_grad():
        dec.w_edge.weight.normal_()          # a non-trivial edge term, so this isn't vacuous
    inp = _inputs(3, 6, edge_dim=2, seed=3)
    zeros = torch.zeros(3, 6)
    assert torch.equal(_run(dec, inp, gate_penalty=None),
                       _run(dec, inp, gate_penalty=zeros))


def test_gate_penalty_pushes_probability_away_monotonically() -> None:
    """More penalty on a candidate => strictly less probability on it (pre-tanh)."""
    torch.manual_seed(4)
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    inp = _inputs(2, 6, edge_dim=2, seed=4)
    probs = []
    for scale in (0.0, 0.5, 2.0):
        penalty = torch.zeros(2, 6)
        penalty[:, 3] = scale
        probs.append(_run(dec, inp, gate_penalty=penalty).exp()[:, 3])
    assert bool((probs[0] > probs[1]).all()) and bool((probs[1] > probs[2]).all())


# -------------------------------------------------------------------- the mode flag


def test_post_tanh_mode_differs_and_escapes_the_clip() -> None:
    torch.manual_seed(5)
    inp = _inputs(3, 6, edge_dim=2, seed=5)
    pre = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    post = PointerDecoder(
        hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2, edge_bias_mode="post_tanh"
    )
    post.load_state_dict(pre.state_dict(), strict=True)
    with torch.no_grad():
        pre.w_edge.weight.fill_(3.0)
        post.w_edge.weight.fill_(3.0)
    assert not torch.equal(_run(pre, inp), _run(post, inp))


def test_unknown_bias_mode_raises() -> None:
    with pytest.raises(ValueError):
        PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_bias_mode="sigmoid")


# ----------------------------------------------------------------------- the guards


@pytest.mark.parametrize(
    "edge_dim, over, exc",
    [
        (2, {"edge_feat": None}, ValueError),                       # missing when required
        (0, {"edge_feat": torch.rand(4, 7, 2)}, ValueError),        # supplied when unused
        (2, {"edge_feat": torch.rand(4, 7, 3)}, ValueError),        # wrong width
        (2, {"gate_penalty": torch.rand(4, 8)}, ValueError),        # wrong gate shape
    ],
)
def test_forward_guards_raise(edge_dim: int, over: dict, exc) -> None:
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=edge_dim)
    inp = _inputs(4, 7, edge_dim=edge_dim, seed=6)
    with pytest.raises(exc):
        _run(dec, inp, **over)


def test_negative_edge_feat_dim_raises() -> None:
    with pytest.raises(ValueError):
        PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=-1)


# ------------------------------------------------------------------- diagnostics


def test_last_u_is_recorded_only_when_asked_and_is_detached() -> None:
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    inp = _inputs(3, 6, edge_dim=2, seed=7)

    _run(dec, inp)
    assert dec._last_u is None                     # off by default: zero training overhead

    dec.record_diagnostics = True
    _run(dec, inp)
    assert dec._last_u is not None
    assert dec._last_u.shape == (3, 6)
    assert not dec._last_u.requires_grad          # detached: nothing retained across steps


def test_recorded_u_is_the_tanh_argument() -> None:
    """The saturation meter thresholds |u|, so u must be what tanh actually sees."""
    torch.manual_seed(8)
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    dec.record_diagnostics = True
    with torch.no_grad():
        dec.w_edge.weight.fill_(1.5)
    inp = _inputs(2, 5, edge_dim=2, seed=8)
    log_p = _run(dec, inp)
    u = dec._last_u
    expected = torch.log_softmax(
        (dec.tanh_clip * torch.tanh(u)).masked_fill(inp["mask"], float("-inf")), dim=-1
    )
    assert torch.equal(log_p, expected)


def test_gradients_flow_into_w_edge() -> None:
    dec = PointerDecoder(hidden_dim=_HIDDEN, n_heads=_HEADS, edge_feat_dim=2)
    inp = _inputs(2, 5, edge_dim=2, seed=9)
    _run(dec, inp)[:, 1].sum().backward()
    # Zero-init must not be a dead path: dL/dw_edge = dL/du * edge_feat != 0.
    assert dec.w_edge.weight.grad is not None
    assert float(dec.w_edge.weight.grad.abs().sum()) > 0.0