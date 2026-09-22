"""Gate-chain assertions: every term added on top of variant A must be inert when disabled.

Variant B's NETWORK cannot be bit-for-bit equal to variant A's, and no assertion should
pretend otherwise: B's encoder edge projection is (16, 2) against A's (16, 1) -- a
different function with different parameters -- and with the bank off B is fed
[mu_bar, 0] where mu_bar is a K-way float mean that differs from the raw distance by
~1e-7. An `allclose` with a tolerance loose enough to pass that would be loose enough to
hide real drift, which is worse than no assertion.

What IS exactly provable is a CHAIN, each link bitwise:

  1. ENV     -- with the bank off (alpha = 0, beta_var = 0) the stochastic env behaves
                identically to the certified CVRPTWEnv, and incurs zero tardiness.
  2. DECODER -- with w_edge zeroed, the variant-B pointer equals the variant-A pointer
                (edge_feat_dim = 0) for ANY edge features.
  3. GATE    -- with gate_penalty zeroed (or None), variant D's pointer equals variant B's.
  4. ENCODER -- with the encoder gate zeroed (or None), variant D's encoder equals
                variant B's, for ANY graph. This guards the final certified-file edit.
  5. SED GATE -- with variant C's SED2AM-style gates switched off, its encoder equals a
                gateless encoder holding the same shared weights, and its pointer gate is
                absent. C then reduces to B exactly, as D does with the gate off.

Stacking B then D can therefore only change behaviour through terms that are switched on,
never through drift. Cheap enough to call periodically from a training loop; links 2 and 3
touch no data and mutate nothing (they compare against freshly built reference decoders
rather than zeroing the live parameters, so they are safe to call at any point).
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from src.data.cvrptw_generator import CVRPTWData
from src.data.simulator import SimulatorConfig, build_edge_bank, draw_noise
from src.envs.cvrptw_env import CVRPTWEnv
from src.envs.cvrptw_stoch_env import CVRPTWStochEnv
from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder


def _reference_decoder(decoder: PointerDecoder, edge_feat_dim: int) -> PointerDecoder:
    """A copy of `decoder` with the given edge width and a ZEROED edge term.

    Built fresh rather than mutating the live module, so these checks never touch the
    parameters being trained.
    """
    ref = PointerDecoder(
        hidden_dim=decoder.hidden_dim,
        n_heads=decoder.n_heads,
        tanh_clip=decoder.tanh_clip,
        context_extra_dim=decoder.context_extra_dim,
        edge_feat_dim=edge_feat_dim,
        edge_bias_mode=decoder.edge_bias_mode,
    )
    state = {
        k: v.detach().clone()
        for k, v in decoder.state_dict().items()
        if not k.startswith("w_edge")
    }
    if edge_feat_dim > 0:
        state["w_edge.weight"] = torch.zeros(
            1, edge_feat_dim, dtype=decoder.wk_pointer.weight.dtype
        )
    ref.load_state_dict(state, strict=True)
    ref.to(decoder.wk_pointer.weight.device)
    return ref.eval()


def _fake_decode_inputs(
    decoder: PointerDecoder, batch: int, n_nodes: int, seed: int
) -> dict[str, Tensor | None]:
    """Random but seeded decode-step inputs. The identities must hold for ANY input."""
    dev = decoder.wk_pointer.weight.device
    dtype = decoder.wk_pointer.weight.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=gen, device=dev, dtype=dtype)

    node_emb = rand(batch, n_nodes, decoder.hidden_dim)           # (B, N, H)
    mask = torch.zeros(batch, n_nodes, dtype=torch.bool, device=dev)
    mask[:, 0] = True                                             # depot forbidden
    return {
        "node_emb": node_emb,
        "graph_emb": node_emb.mean(dim=1),                        # (B, H)
        "last_node_emb": node_emb[:, 0, :],                       # (B, H)
        "remaining_capacity": rand(batch, 1),                     # (B, 1)
        "mask": mask,                                             # (B, N)
        "context_extra": (
            rand(batch, decoder.context_extra_dim)
            if decoder.context_extra_dim > 0 else None
        ),
        "edge_feat": (
            rand(batch, n_nodes, decoder.edge_feat_dim)
            if decoder.edge_feat_dim > 0 else None
        ),
    }


@torch.no_grad()
def assert_decoder_terms_inert(
    decoder: PointerDecoder,
    batch: int = 3,
    n_nodes: int = 7,
    seed: int = 0,
) -> None:
    """Links 2 and 3 of the chain. Raises RuntimeError on any drift.

    Touches no data and mutates nothing, so it is safe to call at any point in training.
    """
    inputs = _fake_decode_inputs(decoder, batch, n_nodes, seed)
    node_emb = inputs.pop("node_emb")
    edge_feat = inputs.pop("edge_feat")
    cache = decoder.precompute(node_emb)

    # --- link 3: gate_penalty of zeros must be identical to gate_penalty=None ---
    zeros = torch.zeros_like(inputs["mask"], dtype=node_emb.dtype)   # (B, N)
    out_none = decoder(cache, **inputs, edge_feat=edge_feat, gate_penalty=None)
    out_zero = decoder(cache, **inputs, edge_feat=edge_feat, gate_penalty=zeros)
    if not torch.equal(out_none, out_zero):
        raise RuntimeError(
            "gate chain link 3 broken: a zero gate_penalty changed the pointer output, "
            "so variant D would not reduce to variant B when the gate is off"
        )

    # --- link 2: a zeroed edge term must be identical to having no edge term at all ---
    if decoder.edge_feat_dim > 0:
        zeroed = _reference_decoder(decoder, decoder.edge_feat_dim)
        without = _reference_decoder(decoder, 0)
        out_zeroed = zeroed(cache, **inputs, edge_feat=edge_feat)
        out_without = without(cache, **inputs, edge_feat=None)
        if not torch.equal(out_zeroed, out_without):
            raise RuntimeError(
                "gate chain link 2 broken: a zeroed w_edge changed the pointer output, "
                "so variant B would not reduce to variant A's pointer function"
            )


@torch.no_grad()
def assert_encoder_gate_inert(
    encoder: EGATEncoder,
    n_nodes: int = 6,
    seed: int = 0,
    jitter_tolerance: float = 8.0,
) -> None:
    """Link 4: an all-zero encoder gate must change nothing about the node embeddings.

    Arithmetically the guarantee is exact -- `score - 0.0` IS `score` -- so on a
    deterministic backend this is a bitwise assertion.

    On CUDA it cannot be. PyG's `softmax` and `global_mean_pool` are scatter reductions
    that sum with atomics, so the order of floating-point additions varies between calls
    and running the SAME computation twice already differs in the last bits. Demanding
    bitwise equality there would fail on hardware rather than on a bug.

    So the check first measures that backend jitter (two identical ungated calls), then
    requires the zero-gated call to deviate by no more than `jitter_tolerance` times it.
    Bitwise when the backend is exact; otherwise indistinguishable-from-noise. A gate that
    genuinely leaked would move the embeddings by orders of magnitude more than atomics do.

    Touches nothing and allocates nothing persistent, so it is safe to call during training.
    """
    dev = encoder.fc_node.weight.device
    dtype = encoder.fc_node.weight.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.arange(n_nodes, device=dev)
    edge_index = torch.stack(
        [idx.repeat_interleave(n_nodes), idx.repeat(n_nodes)], dim=0
    )                                                              # (2, N*N)
    n_edges = edge_index.size(1)
    x = torch.rand(n_nodes, encoder.fc_node.in_features,
                   generator=gen, device=dev, dtype=dtype)
    edge_attr = torch.rand(n_edges, encoder.fc_edge.in_features,
                           generator=gen, device=dev, dtype=dtype)

    was_training = encoder.training
    encoder.eval()                     # BatchNorm would otherwise update between calls
    try:
        ungated, _ = encoder(x, edge_index, edge_attr)
        repeat, _ = encoder(x, edge_index, edge_attr)        # same call: jitter baseline
        zeroed, _ = encoder(
            x, edge_index, edge_attr,
            gate_penalty=torch.zeros(n_edges, device=dev, dtype=dtype),
        )
    finally:
        encoder.train(was_training)

    if torch.equal(ungated, repeat):                         # deterministic backend
        if not torch.equal(ungated, zeroed):
            raise RuntimeError(
                "gate chain link 4 broken: a zero encoder gate changed the node "
                "embeddings bitwise, on a backend where repeating the same call does "
                "not. Variant D would not reduce to variant B with the gate off."
            )
        return

    baseline = float((ungated - repeat).abs().max())         # backend's own jitter
    delta = float((ungated - zeroed).abs().max())
    if delta > jitter_tolerance * baseline:
        raise RuntimeError(
            f"gate chain link 4 broken: a zero encoder gate moved the node embeddings by "
            f"{delta:.3e}, more than {jitter_tolerance}x this backend's own run-to-run "
            f"jitter of {baseline:.3e}. Variant D would not reduce to variant B with the "
            f"gate off."
        )


@torch.no_grad()
def assert_sed_gate_inert(policy, n_nodes: int = 6, seed: int = 0) -> None:
    """Link 5: variant C with its gates OFF must be variant B, bitwise (CPU).

    Same jitter-aware comparison as link 4: on CUDA, PyG's scatter reductions sum with
    atomics, so two identical calls already differ in the last bits.

    No-op for any policy without SED2AM-style gates (A/B/D), so the chain can be called
    uniformly from the training loop.
    """
    if not hasattr(policy, "set_sed_gates"):
        return
    encoder = policy.encoder
    dev = encoder.fc_node.weight.device
    dtype = encoder.fc_node.weight.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.arange(n_nodes, device=dev)
    edge_index = torch.stack(
        [idx.repeat_interleave(n_nodes), idx.repeat(n_nodes)], dim=0
    )                                                              # (2, N*N)
    x = torch.rand(n_nodes, encoder.fc_node.in_features,
                   generator=gen, device=dev, dtype=dtype)
    edge_attr = torch.rand(edge_index.size(1), encoder.fc_edge.in_features,
                           generator=gen, device=dev, dtype=dtype)

    # A gateless twin holding C's shared weights: what variant B WOULD compute here.
    reference = EGATEncoder(
        input_node_dim=encoder.fc_node.in_features,
        input_edge_dim=encoder.fc_edge.in_features,
        hidden_dim=encoder.hidden_dim,
        hidden_edge_dim=encoder.fc_edge.out_features,
        n_layers=len(encoder.layers),
        n_heads=encoder.layers[0].n_heads,
        negative_slope=encoder.layers[0].negative_slope,
        norm=encoder.norm,
        attn=encoder.attn,
    ).to(dev)
    shared = {k: v.detach().clone() for k, v in encoder.state_dict().items()
              if "sed_gate" not in k}
    reference.load_state_dict(shared, strict=True)
    reference.eval()

    was_training = encoder.training
    was_active = policy.sed_gate_active
    encoder.eval()
    try:
        policy.set_sed_gates(False)
        ungated, _ = reference(x, edge_index, edge_attr)
        repeat, _ = reference(x, edge_index, edge_attr)     # backend jitter baseline
        off, _ = encoder(x, edge_index, edge_attr)
    finally:
        policy.set_sed_gates(was_active)
        encoder.train(was_training)

    if policy._gate_penalty is not None and policy.sed_gate_active is not was_active:
        raise RuntimeError("gate chain link 5 broken: the gate switch was not restored")

    if torch.equal(ungated, repeat):                        # deterministic backend
        if not torch.equal(ungated, off):
            raise RuntimeError(
                "gate chain link 5 broken: variant C with its gates OFF does not "
                "reproduce a gateless encoder with the same weights, bitwise"
            )
        return
    baseline = float((ungated - repeat).abs().max())
    delta = float((ungated - off).abs().max())
    if delta > 8.0 * baseline:
        raise RuntimeError(
            f"gate chain link 5 broken: C with its gates off moved the node embeddings by "
            f"{delta:.3e}, more than 8x this backend's own jitter of {baseline:.3e}"
        )


def assert_env_reduces_to_variant_a(
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    seed: int = 0,
) -> None:
    """Link 1 of the chain: bank off => the stochastic env IS the certified CVRPTW env.

    Runs the same greedy-feasible action sequence through both and compares bitwise.
    """
    off = replace(sim_cfg, alpha=0.0, beta_var=0.0)
    dev = data.coords.device
    bank = build_edge_bank(data.coords, off)                       # (B, N, N, K, 2)
    noise = draw_noise(                                            # deliberately non-zero
        bank.shape[:-1], device=dev, dtype=data.coords.dtype,
        generator=torch.Generator(device=dev).manual_seed(seed),
    )
    args = (
        data.coords, data.demand, data.capacity,
        data.time_windows, data.service_time, data.shift_end,
    )
    deterministic = CVRPTWEnv(*args)
    stochastic = CVRPTWStochEnv(*args, bank=bank, sim_cfg=off, noise=noise,
                                tardiness_weight=1.0)

    mask = deterministic.reset()
    actions: list[Tensor] = []
    for _ in range(3 * deterministic.num_nodes):
        step = deterministic._travel_from_current().masked_fill(mask, float("inf"))
        action = step.argmin(dim=1)                                # (B,)
        actions.append(action)
        mask, done = deterministic.step(action)
        if bool(done.all()):
            break
    actions_t = torch.stack(actions, dim=1)                        # (B, T)

    stoch_mask = stochastic.reset()
    for t in range(actions_t.size(1)):
        stoch_mask, _ = stochastic.step(actions_t[:, t])
    if not torch.equal(deterministic.time, stochastic.time):
        raise RuntimeError("gate chain link 1 broken: clocks diverged with the bank off")
    if float(stochastic.tardiness.abs().sum()) != 0.0:
        raise RuntimeError("gate chain link 1 broken: tardiness with zero variance")
    if not torch.equal(
        stochastic.cost_components(actions_t)["distance"],
        deterministic.cost(actions_t),
    ):
        raise RuntimeError("gate chain link 1 broken: tour distance diverged")


def assert_gate_chain_inert(
    policy,
    data: CVRPTWData,
    sim_cfg: SimulatorConfig,
    seed: int = 0,
) -> None:
    """All links. Call periodically from the training loop and in the test suite.

    Link 5 is a no-op unless `policy` is variant C.
    """
    assert_env_reduces_to_variant_a(data, sim_cfg, seed=seed)
    assert_decoder_terms_inert(policy.decoder, seed=seed)
    assert_encoder_gate_inert(policy.encoder, seed=seed)
    assert_sed_gate_inert(policy, seed=seed)