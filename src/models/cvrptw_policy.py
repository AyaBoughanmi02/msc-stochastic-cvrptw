"""CVRPTW policy: CVRP policy + windows in the encoder + clock in the decoder.

Subclass of CVRPPolicy that changes exactly two things in the rollout and inherits
everything else (action selection, the multinomial sampling guard, the loop):

  1. Encode once with 6-dim node features [x, y, demand, earliest, latest, service]
     via build_graph_tw (static windows enter the ONE-TIME encoding).
  2. At each decode step, pass the normalized clock env.normalized_time() to the
     decoder as `context_extra` (the only time-varying quantity; keeps the encoder
     run-once, preserving the amortization story).

Trains under POMO exactly like the CVRP policy: cost is tour distance (unchanged),
so the shared baseline and loss are identical; only the environment enforces the
time windows and shift via masking.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.graph_build import build_graph_tw
from src.envs.cvrptw_env import CVRPTWEnv
from src.models.policy import CVRPPolicy


class CVRPTWPolicy(CVRPPolicy):
    """Encoder + pointer decoder rollout policy for CVRPTW (variant A)."""

    def forward(
        self,
        env: CVRPTWEnv,
        decode_type: str = "sampling",       # "sampling" (train) or "greedy" (eval)
        force_first: Tensor | None = None,   # (B,) POMO forced first customers
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Roll out one full episode per instance. Returns (actions, sum_logp, cost).

        actions:  (B, T) long   visited nodes (depot may recur)
        sum_logp: (B,)   float  summed log-prob of sampled actions (has grad)
        cost:     (B,)   float  tour length (no grad)
        """
        b, n = env.batch_size, env.num_nodes
        dev = env.coords.device

        # --- encode once (6-dim node features: windows + service are static) ---
        enc = build_graph_tw(env.coords, env.demand, env.time_windows, env.service_time)
        node_flat, global_emb = self.encoder(
            enc.x, enc.edge_index, enc.edge_attr, enc.batch
        )                                       # (B*N, H), (B, H)
        node_emb = node_flat.view(b, n, -1)     # (B, N, H)
        cache = self.decoder.precompute(node_emb)

        # --- autoregressive rollout ---
        mask = env.reset(force_first)                       # (B, N)
        finished = torch.zeros(b, dtype=torch.bool, device=dev)
        sum_logp = torch.zeros(b, device=dev)
        actions: list[Tensor] = []
        batch_idx = torch.arange(b, device=dev)

        for _ in range(2 * n):  # >= max actions (<= 2*(n-1)); assert completion below
            last_node_emb = node_emb[batch_idx, env.current_node]  # (B, H)
            remaining = env.normalized_remaining()                 # (B, 1)
            clock = env.normalized_time()                          # (B, 1) dynamic clock
            log_p = self.decoder(
                cache, global_emb, last_node_emb, remaining, mask, context_extra=clock
            )                                                      # (B, N)

            action = self._select(log_p, decode_type)              # (B,)  inherited
            step_logp = log_p.gather(1, action.view(-1, 1)).squeeze(1)
            sum_logp = sum_logp + step_logp * (~finished)
            actions.append(action)

            mask, done = env.step(action)
            finished = finished | done
            if bool(finished.all()):
                break

        if not bool(finished.all()):  # crash-not-silent: truncated rollout would be a bug
            raise RuntimeError(
                "CVRPTW rollout did not complete within the step budget; "
                "some customers were left unserved"
            )

        actions_t = torch.stack(actions, dim=1)  # (B, T)
        cost = env.cost(actions_t)               # (B,) tour distance (no grad)
        return actions_t, sum_logp, cost