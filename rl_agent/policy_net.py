"""
rl_agent/policy_net.py
───────────────────────
Actor-Critic neural network for the PPO agent.

Architecture
────────────
  Shared MLP backbone:
    obs_dim → 256 → 128  (ReLU activations, LayerNorm for stability)

  Actor head (policy π):
    128 → n_actions  (returns raw logits → Categorical distribution)

  Critic head (value V):
    128 → 1          (returns scalar state-value estimate)

Public API
──────────
  ActorCritic(obs_dim, n_actions)
    .act(obs)           → action, log_prob, value, entropy
    .evaluate(obs, act) → log_probs, values, entropy
    .get_value(obs)     → value  (critic only, no grad needed)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp_layer(in_dim: int, out_dim: int) -> nn.Sequential:
    """Linear → LayerNorm → ReLU block."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic for discrete action spaces.

    Parameters
    ----------
    obs_dim   : dimension of the flat observation vector.
    n_actions : number of discrete actions.
    hidden1   : width of the first hidden layer (default 256).
    hidden2   : width of the second hidden layer (default 128).
    """

    def __init__(
        self,
        obs_dim:   int,
        n_actions: int,
        hidden1:   int = 256,
        hidden2:   int = 128,
    ) -> None:
        super().__init__()

        # ── shared backbone ───────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            _mlp_layer(obs_dim, hidden1),
            _mlp_layer(hidden1, hidden2),
        )

        # ── actor head ────────────────────────────────────────────────────────
        self.actor_head = nn.Linear(hidden2, n_actions)

        # ── critic head ───────────────────────────────────────────────────────
        self.critic_head = nn.Linear(hidden2, 1)

        # Weight initialisation — orthogonal init improves PPO stability
        self._init_weights()

    # ── weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        # Actor head uses smaller gain to keep initial policy close to uniform
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    # ── forward helpers ───────────────────────────────────────────────────────

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute shared backbone features."""
        return self.backbone(obs)

    def _distribution(self, features: torch.Tensor) -> Categorical:
        logits = self.actor_head(features)
        return Categorical(logits=logits)

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the current policy (no gradient).

        Parameters
        ----------
        obs : torch.Tensor, shape (..., obs_dim)

        Returns
        -------
        action   : torch.Tensor, int64, shape (...)
        log_prob : torch.Tensor, float32, shape (...)
        value    : torch.Tensor, float32, shape (...)
        entropy  : torch.Tensor, float32, shape (...)
        """
        features = self._features(obs)
        dist     = self._distribution(features)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.critic_head(features).squeeze(-1)
        entropy  = dist.entropy()
        return action, log_prob, value, entropy

    def evaluate(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored (obs, action) pairs — used during PPO update.

        Parameters
        ----------
        obs     : torch.Tensor, shape (B, obs_dim)
        actions : torch.Tensor, int64, shape (B,)

        Returns
        -------
        log_probs : torch.Tensor, shape (B,)
        values    : torch.Tensor, shape (B,)
        entropy   : torch.Tensor, shape (B,)
        """
        features  = self._features(obs)
        dist      = self._distribution(features)
        log_probs = dist.log_prob(actions)
        values    = self.critic_head(features).squeeze(-1)
        entropy   = dist.entropy()
        return log_probs, values, entropy

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic-only forward pass (bootstrapping terminal value).

        Parameters
        ----------
        obs : torch.Tensor, shape (..., obs_dim)

        Returns
        -------
        value : torch.Tensor, shape (...)
        """
        features = self._features(obs)
        return self.critic_head(features).squeeze(-1)
