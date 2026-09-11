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

from src.data.graph_build import EncoderInputs, build_graph_tw
from src.envs.cvrptw_env import CVRPTWEnv
from src.models.policy import CVRPPolicy


class CVRPTWPolicy(CVRPPolicy):
    """Encoder + pointer decoder rollout policy for CVRPTW (variant A)."""

    def _encoder_inputs(self, env: CVRPTWEnv) -> EncoderInputs:
        """Graph inputs for the ONE-TIME encoding. Hook: variants B/C/D swap the edge
        feature for the horizon-averaged (mu_bar, sigma_bar^2) moments."""
        return build_graph_tw(env.coords, env.demand, env.time_windows, env.service_time)

    def _decoder_edge_feat(self, env: CVRPTWEnv) -> Tensor | None:
        """Per-step, per-candidate edge features for the pointer. (B, N, F) or None.

        Hook: variant A has none; B/C/D return the clock-refreshed (mu, sigma^2)(tau_t)
        of the edges leaving the current node.
        """
        return None

    def _gate_penalty(
        self, env: CVRPTWEnv, edge_feat: Tensor | None
    ) -> Tensor | None:
        """Non-negative per-candidate penalty subtracted from the pointer logit. (B, N).

        Hook: None for A/B; variant D returns softplus(w) * sigma^2(tau_t). It lives on
        the POLICY because w is shared with the encoder.
        """
        return None

    def _encoder_gate_penalty(
        self, env: CVRPTWEnv, enc: EncoderInputs
    ) -> Tensor | None:
        """Per-edge variance-gate penalty for the ONE-TIME encoding. (num_edges,) or None.

        Hook: None for A/B/C; variant D returns softplus(w) * sigma_bar^2, built from the
        single shared scalar the POLICY owns. Taken from `enc.edge_attr` itself so the
        penalty and the graph cannot disagree about edge ordering.
        """
        return None

    def policy_metrics(self) -> dict[str, float]:
        """Scalars to log alongside training stats. Hook: variant D reports softplus(w)."""
        return {}

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
        enc = self._encoder_inputs(env)                 # hook: edge features (A: distance)
        gate = self._encoder_gate_penalty(env, enc)     # (B*N*N,) or None
        node_flat, global_emb = self.encoder(
            enc.x, enc.edge_index, enc.edge_attr, enc.batch, gate_penalty=gate
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
            edge_feat = self._decoder_edge_feat(env)               # (B, N, F) or None
            gate_penalty = self._gate_penalty(env, edge_feat)      # (B, N) or None
            log_p = self.decoder(
                cache, global_emb, last_node_emb, remaining, mask, context_extra=clock,
                edge_feat=edge_feat, gate_penalty=gate_penalty,
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