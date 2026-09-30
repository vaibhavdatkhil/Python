"""
rl_env/wrappers.py
───────────────────
Gymnasium wrappers for K8sAutoscalingEnv.

RecordEpisodeStats
──────────────────
  A thin wrapper that accumulates per-episode statistics without altering
  the environment's observations, actions, or rewards.

  Collected after each episode (on `truncated` or `terminated`):
    - total_reward      : sum of step rewards
    - episode_length    : number of steps
    - slo_compliance    : fraction of steps where SLO was met
    - avg_replicas      : mean replica count over the episode
    - min_replicas      : min replica count
    - max_replicas      : max replica count

  Access via:
    env.episode_stats        → list of dicts, one per completed episode
    env.last_episode_stats   → dict for the most recently completed episode

Usage
─────
  from rl_env import K8sAutoscalingEnv, RecordEpisodeStats

  base_env = K8sAutoscalingEnv(cfg, model=model, scaler=scaler, trace_df=df)
  env      = RecordEpisodeStats(base_env)

  obs, info = env.reset()
  while True:
      obs, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
          print(env.last_episode_stats)
          break
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym


class RecordEpisodeStats(gym.Wrapper):
    """Accumulates per-episode statistics from the info dict.

    The wrapper expects each ``info`` dict produced by the wrapped env to
    contain the keys: ``slo_met`` (bool) and ``replicas`` (int).

    Parameters
    ----------
    env : a K8sAutoscalingEnv (or any compatible env with the above info keys).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_stats: list[dict] = []
        self.last_episode_stats: dict = {}

        # Accumulators reset at each episode
        self._ep_reward:   float = 0.0
        self._ep_length:   int   = 0
        self._slo_met_sum: int   = 0
        self._replicas_buf: list[int] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_accumulators()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._ep_reward   += reward
        self._ep_length   += 1
        self._slo_met_sum += int(info.get("slo_met", False))
        self._replicas_buf.append(int(info.get("replicas", 1)))

        if terminated or truncated:
            stats = {
                "total_reward":   self._ep_reward,
                "episode_length": self._ep_length,
                "slo_compliance": (
                    self._slo_met_sum / max(self._ep_length, 1)
                ),
                "avg_replicas":   float(np.mean(self._replicas_buf)),
                "min_replicas":   int(np.min(self._replicas_buf)),
                "max_replicas":   int(np.max(self._replicas_buf)),
            }
            self.episode_stats.append(stats)
            self.last_episode_stats = stats
            info["episode"] = stats   # standard SB3 / cleanRL convention

        return obs, reward, terminated, truncated, info

    # ── internal ──────────────────────────────────────────────────────────────

    def _reset_accumulators(self) -> None:
        self._ep_reward    = 0.0
        self._ep_length    = 0
        self._slo_met_sum  = 0
        self._replicas_buf = []
