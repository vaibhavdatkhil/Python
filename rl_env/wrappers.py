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

import csv
import io
from pathlib import Path
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


# ─────────────────────────────────────────────────────────────────────────────
# TransitionLogger
# ─────────────────────────────────────────────────────────────────────────────

class TransitionLogger(gym.Wrapper):
    """Gymnasium wrapper that logs every (s, a, r, s', done) transition to a CSV.

    Each row written to ``output_path`` contains:
      - ``obs_0 … obs_{N-1}``  — flattened current-state observation (N floats)
      - ``action``             — integer action taken
      - ``reward``             — scalar reward received
      - ``next_obs_0 … next_obs_{N-1}`` — next-state observation (N floats)
      - ``done``               — 1 if episode ended (terminated or truncated), else 0

    The file is created (with header) on the first call to ``reset()``.
    Rows are flushed after every write so partial runs are always readable.

    Parameters
    ----------
    env         : wrapped environment.
    output_path : destination CSV file.  Parent directory is created if needed.

    Usage
    -----
    .. code-block:: python

        env = K8sAutoscalingEnv(cfg, ...)
        env = TransitionLogger(env, output_path="data/sim_transitions.csv")
        obs, _ = env.reset()
        obs, r, term, trunc, info = env.step(action)
        # → data/sim_transitions.csv is created/appended with one row per step
    """

    def __init__(
        self,
        env: gym.Env,
        output_path: str | Path = "data/sim_transitions.csv",
    ) -> None:
        super().__init__(env)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        obs_dim = int(env.observation_space.shape[0])
        self._obs_dim = obs_dim

        # Build CSV header once
        obs_cols      = [f"obs_{i}"      for i in range(obs_dim)]
        next_obs_cols = [f"next_obs_{i}" for i in range(obs_dim)]
        self._fieldnames = obs_cols + ["action", "reward"] + next_obs_cols + ["done"]

        # Open the file in append mode so multiple runs accumulate data.
        # Write header only when the file is new / empty.
        file_exists_and_nonempty = (
            self.output_path.exists() and self.output_path.stat().st_size > 0
        )
        self._fh = open(self.output_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if not file_exists_and_nonempty:
            self._writer.writeheader()
            self._fh.flush()

        self._current_obs: Optional[np.ndarray] = None

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = np.asarray(obs, dtype=np.float32)

        done = terminated or truncated
        cur  = self._current_obs if self._current_obs is not None else np.zeros(self._obs_dim)

        row: dict[str, Any] = {}
        for i, v in enumerate(cur):
            row[f"obs_{i}"] = float(v)
        row["action"] = int(action)
        row["reward"] = float(reward)
        for i, v in enumerate(next_obs):
            row[f"next_obs_{i}"] = float(v)
        row["done"] = int(done)

        self._writer.writerow(row)
        self._fh.flush()  # persist immediately so partial runs are readable

        self._current_obs = next_obs
        return obs, reward, terminated, truncated, info

    # ── cleanup ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if hasattr(self, "_fh") and self._fh and not self._fh.closed:
            self._fh.close()
        super().close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
