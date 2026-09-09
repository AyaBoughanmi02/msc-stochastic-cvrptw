"""Tests for the variant-B encoder edge features: dense (mu_bar, sigma_bar^2) -> edge_attr.

Load-bearing tests:

1. `test_dense_edge_attr_lands_in_the_right_edge_slot` -- the flatten relies on a dense
   (B, N, N, F) tensor being row-major in the SAME order as build_graph's edge_index. A
   transposed or mis-ordered input would flatten without error into a wrong-but-plausible
   graph, and every downstream number would look reasonable. The test tags each entry with
   the unique value b*100 + i*10 + j and re-derives the expected value from edge_index
   itself, so it checks the contract rather than restating the implementation.
2. `test_variant_a_graph_is_unchanged` -- with edge_attr=None nothing about the CVRPTW
   graph moves, and build_graph (the CVRP path) is not reachable from the new argument.
3. `test_certified_configs_still_load_strict` -- gate B: the checkpointed configs must
   still reconstruct their original architecture and load with strict=True.
"""

from __future__ import annotations

import pytest
import torch

from src.data.cvrptw_generator import generate_cvrptw
from src.data.graph_build import build_graph, build_graph_tw
from src.data.simulator import MU, SIGMA2, SimulatorConfig, build_edge_bank, horizon_average
from src.models.egat_encoder import EGATEncoder
from src.models.factory import (
    DEFAULT_CVRPTW_CONFIG,
    DEFAULT_CVRPTW_STOCH_CONFIG,
    DEFAULT_MODEL_CONFIG,
    build_cvrptw_policy,
    build_policy,
)

_SHIFT = 4.0
_SERVICE = 0.2
_CAPACITY = 20


def _data(batch: int = 3, n_customers: int = 5, seed: int = 0):
    return generate_cvrptw(
        batch, n_customers, capacity=_CAPACITY, seed=seed,
        shift_end=_SHIFT, service_time=_SERVICE,
    )


# ------------------------------------------------------- the flatten order is the risk


def test_dense_edge_attr_lands_in_the_right_edge_slot() -> None:
    """dense[b, i, j] must end up on the edge whose source is i and target is j."""
    data = _data(batch=3, n_customers=5, seed=1)
    b, n, _ = data.coords.shape

    # Unique tag per (instance, source, target); F=1 so the value IS the tag.
    tag = (
        torch.arange(b).view(b, 1, 1) * 100
        + torch.arange(n).view(1, n, 1) * 10
        + torch.arange(n).view(1, 1, n)
    ).to(torch.float32).unsqueeze(-1)                                   # (B, N, N, 1)

    enc = build_graph_tw(
        data.coords, data.demand, data.time_windows, data.service_time, edge_attr=tag
    )
    src, tgt = enc.edge_index                                            # (B*N*N,) each
    # Re-derive the expectation from edge_index rather than from the flatten's own logic.
    instance = torch.div(src, n, rounding_mode="floor")                  # (E,)
    expected = (instance * 100 + (src % n) * 10 + (tgt % n)).to(torch.float32)
    assert torch.equal(enc.edge_attr.squeeze(-1), expected)


def test_transposing_the_dense_input_changes_the_graph() -> None:
    """The [source, target] convention is real: a transpose must not be a no-op.

    Our Edge Bank is symmetric today, so a transpose bug would be invisible now and would
    surface only if directional congestion is ever added. Pin it while it is cheap.
    """
    data = _data(batch=2, n_customers=4, seed=2)
    b, n, _ = data.coords.shape
    asymmetric = torch.rand(b, n, n, 2)
    args = (data.coords, data.demand, data.time_windows, data.service_time)
    straight = build_graph_tw(*args, edge_attr=asymmetric).edge_attr
    flipped = build_graph_tw(*args, edge_attr=asymmetric.transpose(1, 2)).edge_attr
    assert not torch.equal(straight, flipped)


@pytest.mark.parametrize(
    "bad",
    [
        lambda b, n: torch.rand(b, n, n),            # missing the feature axis
        lambda b, n: torch.rand(b, n, n + 1, 2),     # wrong node count
        lambda b, n: torch.rand(b + 1, n, n, 2),     # wrong batch
        lambda b, n: torch.rand(n, n, 2),            # missing the batch axis
    ],
)
def test_edge_attr_shape_guards_raise(bad) -> None:
    data = _data(batch=2, n_customers=4, seed=3)
    b, n, _ = data.coords.shape
    with pytest.raises(ValueError):
        build_graph_tw(
            data.coords, data.demand, data.time_windows, data.service_time,
            edge_attr=bad(b, n),
        )


# ------------------------------------------------------------- variant A is untouched


def test_variant_a_graph_is_unchanged() -> None:
    """edge_attr=None must reproduce the distance edge feature exactly."""
    data = _data(batch=4, n_customers=6, seed=4)
    args = (data.coords, data.demand, data.time_windows, data.service_time)
    tw = build_graph_tw(*args)
    cvrp = build_graph(data.coords, data.demand)

    assert tw.edge_attr.shape == (4 * 7 * 7, 1)
    assert torch.equal(tw.edge_attr, cvrp.edge_attr)     # still the CVRP distance column
    assert torch.equal(tw.edge_index, cvrp.edge_index)
    assert torch.equal(tw.batch, cvrp.batch)
    assert tw.x.shape == (4 * 7, 6)


def test_bank_off_gives_back_the_distance_edge_feature() -> None:
    """alpha = 0, beta_var = 0: the 2-dim feature is [distance, 0]."""
    data = _data(batch=4, n_customers=6, seed=5)
    cfg = SimulatorConfig(alpha=0.0, beta_var=0.0, shift_end=_SHIFT)
    moments = horizon_average(build_edge_bank(data.coords, cfg))          # (B, N, N, 2)
    args = (data.coords, data.demand, data.time_windows, data.service_time)

    stoch = build_graph_tw(*args, edge_attr=moments).edge_attr            # (E, 2)
    baseline = build_graph_tw(*args).edge_attr                            # (E, 1) distance
    # mu_bar is a K-way mean of identical values, so equal to float tolerance, not bitwise.
    assert torch.allclose(stoch[:, MU : MU + 1], baseline, rtol=1e-6, atol=0.0)
    assert torch.equal(stoch[:, SIGMA2], torch.zeros_like(stoch[:, SIGMA2]))


# --------------------------------------------------------------- the encoder consumes it


def test_encoder_accepts_two_dim_edges_and_only_fc_edge_changes_shape() -> None:
    """fc_edge.weight is the ONLY parameter in the encoder whose shape moves."""
    torch.manual_seed(0)
    one = EGATEncoder(input_node_dim=6, input_edge_dim=1)
    two = EGATEncoder(input_node_dim=6, input_edge_dim=2)

    shapes_one = {k: tuple(v.shape) for k, v in one.state_dict().items()}
    shapes_two = {k: tuple(v.shape) for k, v in two.state_dict().items()}
    assert shapes_one.keys() == shapes_two.keys()
    differing = {k for k in shapes_one if shapes_one[k] != shapes_two[k]}
    assert differing == {"fc_edge.weight"}
    assert shapes_one["fc_edge.weight"] == (16, 1)
    assert shapes_two["fc_edge.weight"] == (16, 2)


def test_encoder_forward_with_bank_edge_features() -> None:
    data = _data(batch=3, n_customers=9, seed=6)
    b, n, _ = data.coords.shape
    cfg = SimulatorConfig(alpha=0.3, beta_var=0.3, eta=0.6, shift_end=_SHIFT)
    moments = horizon_average(build_edge_bank(data.coords, cfg))          # (B, N, N, 2)
    enc_in = build_graph_tw(
        data.coords, data.demand, data.time_windows, data.service_time, edge_attr=moments
    )
    assert enc_in.edge_attr.shape == (b * n * n, 2)

    encoder = EGATEncoder(**{
        "input_node_dim": 6, "input_edge_dim": 2, "hidden_dim": 128,
        "hidden_edge_dim": 16, "n_layers": 4, "n_heads": 8,
    })
    node_emb, global_emb = encoder(
        enc_in.x, enc_in.edge_index, enc_in.edge_attr, enc_in.batch
    )
    assert node_emb.shape == (b * n, 128)
    assert global_emb.shape == (b, 128)
    assert torch.isfinite(node_emb).all() and torch.isfinite(global_emb).all()


def test_variance_shift_moves_the_embedding_in_eval_mode() -> None:
    """The OOD signal must actually reach the node embeddings through bn_edge."""
    data = _data(batch=3, n_customers=9, seed=7)
    args = (data.coords, data.demand, data.time_windows, data.service_time)
    encoder = EGATEncoder(input_node_dim=6, input_edge_dim=2)
    encoder.eval()                       # frozen BN statistics: the shift propagates

    def embed(beta_var: float) -> torch.Tensor:
        cfg = SimulatorConfig(alpha=0.3, beta_var=beta_var, eta=0.6, shift_end=_SHIFT)
        moments = horizon_average(build_edge_bank(data.coords, cfg))
        e = build_graph_tw(*args, edge_attr=moments)
        with torch.no_grad():
            return encoder(e.x, e.edge_index, e.edge_attr, e.batch)[0]

    assert not torch.allclose(embed(0.3), embed(0.9), atol=1e-5)


# ------------------------------------------------------------------------- gate B


@pytest.mark.parametrize(
    "config, builder",
    [
        (DEFAULT_MODEL_CONFIG, build_policy),
        (DEFAULT_CVRPTW_CONFIG, build_cvrptw_policy),
        (DEFAULT_CVRPTW_STOCH_CONFIG, build_cvrptw_policy),
    ],
)
def test_certified_configs_still_load_strict(config, builder, tmp_path) -> None:
    """A checkpoint's stored config must reconstruct the same architecture, strictly."""
    torch.manual_seed(0)
    original = builder(config)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_config": config, "state_dict": original.state_dict()}, path)

    blob = torch.load(path, weights_only=False)
    restored = builder(blob["model_config"])
    missing, unexpected = restored.load_state_dict(blob["state_dict"], strict=True)
    assert not missing and not unexpected
    for (ka, va), (kb, vb) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert ka == kb
        assert torch.equal(va, vb)


def test_stoch_config_only_moves_the_edge_dim() -> None:
    """The variant-B config is the CVRPTW config plus exactly one changed key."""
    differing = {
        k for k in DEFAULT_CVRPTW_CONFIG
        if DEFAULT_CVRPTW_CONFIG[k] != DEFAULT_CVRPTW_STOCH_CONFIG[k]
    }
    assert differing == {"input_edge_dim"}
    assert DEFAULT_CVRPTW_STOCH_CONFIG["input_edge_dim"] == 2
    # The certified defaults must not have been edited in the process.
    assert DEFAULT_MODEL_CONFIG["input_edge_dim"] == 1
    assert DEFAULT_MODEL_CONFIG["input_node_dim"] == 3
    assert DEFAULT_CVRPTW_CONFIG["input_edge_dim"] == 1