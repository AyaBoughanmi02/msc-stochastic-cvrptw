"""CVRP policy: encoder + pointer decoder + env, wired into one rollout.

Encodes the instance once, then decodes autoregressively: at each step it builds
the decoder context from the current node and remaining capacity, scores the
next node under the env's feasibility mask, picks an action (greedy or sampled),
and advances the env until every trajectory is done. Returns the action
sequence, the summed log-probability of the sampled actions (for the policy
gradient), and the tour-length cost.

This is the object POMO trains. POMO multi-start is handled by the caller:
replicate each instance across the batch and pass `force_first` (customer k for
trajectory k).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.data.graph_build import build_graph
from src.envs.cvrp_env import CVRPEnv
from src.models.egat_encoder import EGATEncoder
from src.models.pointer_decoder import PointerDecoder


class CVRPPolicy(nn.Module):
    """Encoder + decoder rollout policy for CVRP."""

    def __init__(self, encoder: EGATEncoder, decoder: PointerDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        env: CVRPEnv,
        decode_type: str = "sampling",     # "sampling" (train) or "greedy" (eval)
        force_first: Tensor | None = None,  # (B,) POMO forced first customers
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Roll out one full episode for every instance in the batch.

        Returns:
            actions:  (B, T) long   sequence of visited nodes (depot may recur)
            sum_logp: (B,)   float  summed log-prob of sampled actions (has grad)
            cost:     (B,)   float  tour length (no grad)
        """
        b, n = env.batch_size, env.num_nodes
        dev = env.coords.device

        # --- encode once ---
        enc = build_graph(env.coords, env.demand)
        node_flat, global_emb = self.encoder(
            enc.x, enc.edge_index, enc.edge_attr, enc.batch
        )                                   # (B*N, H), (B, H)
        node_emb = node_flat.view(b, n, -1)  # (B, N, H)  contiguous per instance
        cache = self.decoder.precompute(node_emb)

        # --- autoregressive rollout ---
        mask = env.reset(force_first)                       # (B, N)
        finished = torch.zeros(b, dtype=torch.bool, device=dev)
        sum_logp = torch.zeros(b, device=dev)
        actions: list[Tensor] = []
        batch_idx = torch.arange(b, device=dev)

        for _ in range(2 * n):  # Lei's decode budget; enough for CVRP worst case
            last_node_emb = node_emb[batch_idx, env.current_node]  # (B, H)
            remaining = env.normalized_remaining()                 # (B, 1)
            log_p = self.decoder(cache, global_emb, last_node_emb, remaining, mask)  # (B, N)

            action = self._select(log_p, decode_type)              # (B,)
            step_logp = log_p.gather(1, action.view(-1, 1)).squeeze(1)  # (B,)
            # Do not accumulate log-prob for trajectories already finished.
            sum_logp = sum_logp + step_logp * (~finished)
            actions.append(action)

            mask, done = env.step(action)
            finished = finished | done
            if bool(finished.all()):
                break

        actions_t = torch.stack(actions, dim=1)  # (B, T)
        cost = env.cost(actions_t)               # (B,)
        return actions_t, sum_logp, cost

    @staticmethod
    def _select(log_p: Tensor, decode_type: str) -> Tensor:
        """Pick an action per instance. log_p: (B, N) -> (B,).

        Sampling uses a resample guard: on CUDA, torch.multinomial can rarely
        select a zero-probability (masked) element due to a float-underflow bug,
        which would yield an infeasible action and a -inf log-prob. We resample
        until every chosen node has nonzero probability (Kwon et al. reference).
        Sampling is done under no_grad; the gradient flows through the gathered
        log-prob of the chosen action, not through the sampling op.
        """
        if decode_type == "greedy":
            return log_p.argmax(dim=-1)
        if decode_type == "sampling":
            probs = log_p.exp()  # (B, N); masked entries are exactly 0
            with torch.no_grad():
                while True:
                    action = torch.multinomial(probs, 1).squeeze(1)  # (B,)
                    chosen = probs.gather(1, action.view(-1, 1)).squeeze(1)  # (B,)
                    if bool((chosen > 0).all()):
                        return action
        raise ValueError(f"unknown decode_type: {decode_type!r}")