"""
rl_agent/ppo.py
────────────────
Proximal Policy Optimization (PPO-Clip) — from scratch, pure PyTorch.

Algorithm
─────────
  For each iteration:
    1. collect_rollout  — run policy for rollout_steps, store transitions
    2. compute_gae      — Generalized Advantage Estimation (γ, λ)
    3. ppo_update       — K epochs of mini-batch gradient updates

  Loss = -L_clip + c1 * L_vf - c2 * S[π]
    L_clip : clipped surrogate objective (ε)
    L_vf   : value function MSE (clipped variant for stability)
    S[π]   : entropy bonus (encourages exploration)

Reference
─────────
  Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
  https://arxiv.org/abs/1707.06347

  GAE: Schulman et al. (2016)
  https://arxiv.org/abs/1506.02438
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_agent.policy_net import ActorCritic


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """Fixed-length circular buffer for PPO on-policy rollouts.

    Stores T transitions, then is consumed (cleared) after each update.
    All tensors live on CPU during collection; moved to device in ppo_update.
    """
    obs_dim:       int
    n_actions:     int
    rollout_steps: int

    # filled by collect_rollout()
    obs:      torch.Tensor = field(init=False)
    actions:  torch.Tensor = field(init=False)
    rewards:  torch.Tensor = field(init=False)
    dones:    torch.Tensor = field(init=False)
    log_probs: torch.Tensor = field(init=False)
    values:   torch.Tensor = field(init=False)

    # filled by compute_gae()
    advantages: torch.Tensor = field(init=False)
    returns:    torch.Tensor = field(init=False)

    _ptr: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        T = self.rollout_steps
        self.obs       = torch.zeros(T, self.obs_dim)
        self.actions   = torch.zeros(T, dtype=torch.long)
        self.rewards   = torch.zeros(T)
        self.dones     = torch.zeros(T)
        self.log_probs = torch.zeros(T)
        self.values    = torch.zeros(T)
        self.advantages = torch.zeros(T)
        self.returns    = torch.zeros(T)

    def reset(self) -> None:
        self._ptr = 0

    def add(
        self,
        obs:      torch.Tensor,
        action:   torch.Tensor,
        reward:   float,
        done:     bool,
        log_prob: torch.Tensor,
        value:    torch.Tensor,
    ) -> None:
        i = self._ptr
        self.obs[i]       = obs.cpu()
        self.actions[i]   = action.cpu()
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.log_probs[i] = log_prob.cpu()
        self.values[i]    = value.cpu()
        self._ptr += 1

    def is_full(self) -> bool:
        return self._ptr >= self.rollout_steps

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma:      float,
        gae_lambda: float,
    ) -> None:
        """Compute advantages and returns via GAE in-place.

        Parameters
        ----------
        last_value : V(s_{T+1}) — value of the state after the last step.
        gamma      : discount factor.
        gae_lambda : GAE λ parameter.
        """
        T      = self.rollout_steps
        gae    = 0.0
        last_v = last_value.item()

        for t in reversed(range(T)):
            if t == T - 1:
                next_value     = last_v
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_value        = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_non_terminal
                - self.values[t].item()
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_mini_batches(
        self, mini_batch_size: int, device: torch.device
    ):
        """Yield shuffled mini-batches of the stored rollout.

        Yields
        ------
        (obs_b, actions_b, log_probs_old_b, advantages_b, returns_b)
        """
        T      = self.rollout_steps
        indices = torch.randperm(T)

        for start in range(0, T, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield (
                self.obs[idx].to(device),
                self.actions[idx].to(device),
                self.log_probs[idx].to(device),
                self.advantages[idx].to(device),
                self.returns[idx].to(device),
                self.values[idx].to(device),
            )


# ── PPO Update ────────────────────────────────────────────────────────────────

def ppo_update(
    policy:          ActorCritic,
    optimizer:       optim.Optimizer,
    buffer:          RolloutBuffer,
    ppo_cfg:         dict,
    device:          torch.device,
) -> dict[str, float]:
    """Run K epochs of PPO mini-batch gradient updates.

    Parameters
    ----------
    policy    : ActorCritic network (weights updated in-place).
    optimizer : AdamW (or Adam) optimizer.
    buffer    : RolloutBuffer with computed advantages + returns.
    ppo_cfg   : the `ppo:` sub-dict from config.yaml.
    device    : torch device.

    Returns
    -------
    metrics : dict with keys 'policy_loss', 'value_loss', 'entropy', 'kl_approx'.
    """
    clip_eps       = float(ppo_cfg["clip_eps"])
    n_epochs       = int(ppo_cfg["n_epochs"])
    mini_batch_size = int(ppo_cfg["mini_batch_size"])
    value_loss_coef = float(ppo_cfg["value_loss_coef"])
    entropy_coef    = float(ppo_cfg["entropy_coef"])
    max_grad_norm   = float(ppo_cfg["max_grad_norm"])

    # Normalise advantages (across the full rollout, improves stability)
    adv = buffer.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    buffer.advantages = adv

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    total_kl          = 0.0
    n_updates         = 0

    for _ in range(n_epochs):
        for (
            obs_b, acts_b, old_logp_b,
            adv_b, ret_b, old_val_b,
        ) in buffer.get_mini_batches(mini_batch_size, device):

            # ── evaluate current policy ───────────────────────────────────────
            log_probs, values, entropy = policy.evaluate(obs_b, acts_b)

            # ── policy loss (clipped surrogate) ───────────────────────────────
            ratio       = torch.exp(log_probs - old_logp_b)
            surr1       = ratio * adv_b
            surr2       = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── value loss (clipped) ──────────────────────────────────────────
            # Clip value update to prevent large value function jumps
            values_clipped = old_val_b + torch.clamp(
                values - old_val_b, -clip_eps, clip_eps
            )
            vf_loss1   = (values      - ret_b).pow(2)
            vf_loss2   = (values_clipped - ret_b).pow(2)
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            # ── total loss ────────────────────────────────────────────────────
            entropy_loss = -entropy.mean()
            loss = (
                policy_loss
                + value_loss_coef * value_loss
                + entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            # ── metrics ───────────────────────────────────────────────────────
            with torch.no_grad():
                kl = (old_logp_b - log_probs).mean().abs().item()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.mean().item()
            total_kl          += kl
            n_updates         += 1

    denom = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss":  total_value_loss  / denom,
        "entropy":     total_entropy     / denom,
        "kl_approx":   total_kl          / denom,
    }


# ── Full Training Loop ────────────────────────────────────────────────────────

def train_ppo(
    cfg:         dict,
    env_factory,
    device:      torch.device,
    verbose:     bool = True,
    total_timesteps_override: Optional[int] = None,
    log_interval_override:    Optional[int] = None,
) -> dict:
    """Run the PPO training loop.

    Parameters
    ----------
    cfg              : full parsed config dict (from config.yaml).
    env_factory      : callable () -> gym.Env  (creates a fresh env).
    device           : torch device.
    verbose          : if True, print per-rollout progress.
    total_timesteps_override : override `ppo.total_timesteps` from config.
    log_interval_override    : override `ppo.log_interval` from config.

    Returns
    -------
    history : dict with 'episode_rewards', 'slo_pcts', 'avg_replicas'.
    """
    import csv
    from pathlib import Path
    
    ppo_cfg       = cfg["ppo"]
    rollout_steps = int(ppo_cfg["rollout_steps"])
    total_ts      = int(total_timesteps_override or ppo_cfg["total_timesteps"])
    log_interval  = int(log_interval_override    or ppo_cfg["log_interval"])
    lr            = float(ppo_cfg.get("learning_rate", 3e-4))
    gamma         = float(ppo_cfg["gamma"])
    gae_lambda    = float(ppo_cfg["gae_lambda"])
    ckpt_dir      = Path(ppo_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path     = ckpt_dir / "ppo_agent.pt"

    # Optional MLflow tracking
    mlflow_enabled = ppo_cfg.get("mlflow_tracking", False)
    if mlflow_enabled:
        try:
            import mlflow
            mlflow.set_experiment("k8s-rl-autoscaling")
            mlflow.start_run()
            mlflow.log_params({
                "rollout_steps": rollout_steps,
                "total_timesteps": total_ts,
                "learning_rate": lr,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_eps": float(ppo_cfg["clip_eps"]),
                "n_epochs": int(ppo_cfg["n_epochs"]),
                "mini_batch_size": int(ppo_cfg["mini_batch_size"]),
            })
            print(f"  MLflow tracking enabled (experiment: k8s-rl-autoscaling)")
        except ImportError:
            print("  [WARNING] mlflow not installed, skipping tracking")
            mlflow_enabled = False
        except Exception as e:
            print(f"  [WARNING] MLflow setup failed: {e}")
            mlflow_enabled = False
    else:
        mlflow_enabled = False

    # CSV training log
    training_log_path = ckpt_dir / "training_log.csv"
    write_csv_header = not training_log_path.exists() or training_log_path.stat().st_size == 0
    csv_fh = open(training_log_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=[
        "rollout", "timestep", "mean_reward", "slo_pct", "avg_replicas",
        "policy_loss", "value_loss", "entropy",
    ])
    if write_csv_header:
        csv_writer.writeheader()
        csv_fh.flush()

    # ── build env + policy ────────────────────────────────────────────────────
    env       = env_factory()
    obs_dim   = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    policy    = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, eps=1e-5)

    buffer = RolloutBuffer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        rollout_steps=rollout_steps,
    )

    if verbose:
        print(f"  Device         : {device}")
        print(f"  Obs dim        : {obs_dim}")
        print(f"  Actions        : {n_actions}")
        print(f"  Total timesteps: {total_ts:,}")
        print(f"  Rollout steps  : {rollout_steps}")
        print(f"  Policy params  : {sum(p.numel() for p in policy.parameters()):,}")
        print()

    # ── training state ────────────────────────────────────────────────────────
    history: dict[str, list] = {
        "episode_rewards": [],
        "slo_pcts":        [],
        "avg_replicas":    [],
        "policy_loss":     [],
        "value_loss":      [],
        "entropy":         [],
    }

    obs_np, _ = env.reset(seed=0)
    obs        = torch.from_numpy(obs_np).float().to(device)

    timestep      = 0
    rollout_num   = 0
    ep_reward     = 0.0
    slo_count     = 0
    replica_sum   = 0
    ep_steps      = 0
    best_reward   = -float("inf")
    start_time    = time.time()

    # ── main loop ─────────────────────────────────────────────────────────────
    while timestep < total_ts:
        buffer.reset()
        policy.eval()   # eval mode during collection (no dropout in policy net)

        # ── collect rollout ───────────────────────────────────────────────────
        for _ in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, value, _ = policy.act(obs)

            obs_np_next, reward, terminated, truncated, info = env.step(
                action.item()
            )

            buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                done=terminated or truncated,
                log_prob=log_prob,
                value=value,
            )

            ep_reward   += reward
            ep_steps    += 1
            slo_count   += int(info.get("slo_met", False))
            replica_sum += int(info.get("replicas", 1))
            timestep    += 1

            if terminated or truncated:
                history["episode_rewards"].append(ep_reward)
                history["slo_pcts"].append(slo_count / max(ep_steps, 1) * 100)
                history["avg_replicas"].append(replica_sum / max(ep_steps, 1))

                # reset episode counters
                ep_reward   = 0.0
                slo_count   = 0
                replica_sum = 0
                ep_steps    = 0

                obs_np, _ = env.reset()
                obs = torch.from_numpy(obs_np).float().to(device)
            else:
                obs = torch.from_numpy(obs_np_next).float().to(device)

        # ── bootstrap value of last state ─────────────────────────────────────
        with torch.no_grad():
            last_value = policy.get_value(obs)

        buffer.compute_gae(last_value, gamma=gamma, gae_lambda=gae_lambda)

        # ── PPO update ────────────────────────────────────────────────────────
        policy.train()
        metrics = ppo_update(policy, optimizer, buffer, ppo_cfg, device)

        history["policy_loss"].append(metrics["policy_loss"])
        history["value_loss"].append(metrics["value_loss"])
        history["entropy"].append(metrics["entropy"])

        rollout_num += 1

        # ── write training log CSV ────────────────────────────────────────────
        n_eps = len(history["episode_rewards"])
        if n_eps > 0:
            mean_rew = float(np.mean(history["episode_rewards"][-10:]))
            mean_slo = float(np.mean(history["slo_pcts"][-10:]))
            mean_rep = float(np.mean(history["avg_replicas"][-10:]))
        else:
            mean_rew = mean_slo = mean_rep = 0.0
        
        csv_writer.writerow({
            "rollout": rollout_num,
            "timestep": timestep,
            "mean_reward": round(mean_rew, 2),
            "slo_pct": round(mean_slo, 2),
            "avg_replicas": round(mean_rep, 2),
            "policy_loss": round(metrics["policy_loss"], 4),
            "value_loss": round(metrics["value_loss"], 4),
            "entropy": round(metrics["entropy"], 4),
        })
        csv_fh.flush()
        
        # ── MLflow metric logging ─────────────────────────────────────────────
        if mlflow_enabled and n_eps > 0:
            try:
                import mlflow
                mlflow.log_metrics({
                    "mean_reward": mean_rew,
                    "slo_pct": mean_slo,
                    "avg_replicas": mean_rep,
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                }, step=rollout_num)
            except Exception:
                pass  # silently skip if MLflow call fails

        # ── checkpoint: save best ─────────────────────────────────────────────
        if history["episode_rewards"]:
            recent_reward = np.mean(history["episode_rewards"][-5:])
            if recent_reward > best_reward:
                best_reward = recent_reward
                torch.save(
                    {
                        "policy_state": policy.state_dict(),
                        "obs_dim":      obs_dim,
                        "n_actions":    n_actions,
                        "cfg":          cfg,
                        "timestep":     timestep,
                        "best_reward":  best_reward,
                    },
                    ckpt_path,
                )

        # ── logging ───────────────────────────────────────────────────────────
        if verbose and rollout_num % log_interval == 0:
            elapsed = time.time() - start_time
            n_eps   = len(history["episode_rewards"])
            if n_eps > 0:
                mean_rew = np.mean(history["episode_rewards"][-10:])
                mean_slo = np.mean(history["slo_pcts"][-10:])
                mean_rep = np.mean(history["avg_replicas"][-10:])
            else:
                mean_rew = mean_slo = mean_rep = float("nan")

            fps = timestep / max(elapsed, 1e-6)
            if n_eps > 0:
                print(
                    f"  rollout={rollout_num:5d}  "
                    f"ts={timestep:8,}  "
                    f"reward={mean_rew:+8.2f}  "
                    f"SLO={mean_slo:5.1f}%  "
                    f"replicas={mean_rep:4.1f}  "
                    f"pi_loss={metrics['policy_loss']:6.4f}  "
                    f"v_loss={metrics['value_loss']:6.4f}  "
                    f"ent={metrics['entropy']:5.3f}  "
                    f"fps={fps:6.0f}  "
                    f"elapsed={elapsed:6.1f}s"
                )
            else:
                print(
                    f"  rollout={rollout_num:5d}  "
                    f"ts={timestep:8,}  "
                    f"reward=    n/a  "
                    f"SLO=  n/a  "
                    f"replicas= n/a  "
                    f"pi_loss={metrics['policy_loss']:6.4f}  "
                    f"v_loss={metrics['value_loss']:6.4f}  "
                    f"ent={metrics['entropy']:5.3f}  "
                    f"fps={fps:6.0f}  "
                    f"elapsed={elapsed:6.1f}s"
                )

    # ── final save (always) ───────────────────────────────────────────────────
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "obs_dim":      obs_dim,
            "n_actions":    n_actions,
            "cfg":          cfg,
            "timestep":     timestep,
            "best_reward":  best_reward,
            "history":      history,
        },
        ckpt_dir / "ppo_agent_final.pt",
    )

    env.close()
    csv_fh.close()
    
    if mlflow_enabled:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n[OK] Training complete in {elapsed:.1f}s")
        print(f"  Total timesteps  : {timestep:,}")
        if history["episode_rewards"]:
            print(f"  Best mean reward : {best_reward:+.2f}")
            print(f"  Final SLO%       : {np.mean(history['slo_pcts'][-10:]):.1f}%")
        print(f"  Checkpoint saved : {ckpt_path.resolve()}")
        print(f"  Training log saved: {training_log_path.resolve()}")
        if mlflow_enabled:
            print(f"  MLflow tracking  : enabled")

    return history


# ── Checkpoint Loader ─────────────────────────────────────────────────────────

def load_ppo_checkpoint(
    ckpt_path: str | Path,
    device:    torch.device | str = "cpu",
) -> tuple[ActorCritic, dict]:
    """Load a saved PPO checkpoint.

    Parameters
    ----------
    ckpt_path : path to `ppo_agent.pt` (or `ppo_agent_final.pt`).
    device    : torch device.

    Returns
    -------
    policy : ActorCritic in eval mode.
    ckpt   : raw checkpoint dict (contains cfg, timestep, best_reward, etc.).
    """
    device = torch.device(device) if isinstance(device, str) else device
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    policy = ActorCritic(
        obs_dim=ckpt["obs_dim"],
        n_actions=ckpt["n_actions"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    return policy, ckpt
