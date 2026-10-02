"""
rl_agent/train_rl.py
─────────────────────
CLI entry-point for Phase 7: PPO RL agent training.

Workflow
────────
  1. Load config.yaml
  2. Load pre-trained LSTM checkpoint (or run without forecast)
  3. Build K8sAutoscalingEnv
  4. Run PPO training loop (see rl_agent/ppo.py)
  5. Save agent checkpoint to checkpoints/ppo_agent.pt

Usage
─────
  python -m rl_agent.train_rl
  python -m rl_agent.train_rl --config config.yaml
  python -m rl_agent.train_rl --total-timesteps 50000
  python -m rl_agent.train_rl --total-timesteps 2048 --log-interval 1   # quick test
  python -m rl_agent.train_rl --no-model                                 # skip LSTM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def main() -> None:
    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for K8s autoscaling."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: project-root config.yaml).",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override ppo.total_timesteps from config.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=None,
        help="Override ppo.log_interval from config.",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip loading LSTM model (forecast obs will be zeros).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected by default.",
    )
    args = parser.parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / args.config

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  K8s RL Autoscaling — Phase 7: PPO Agent Training")
    print("=" * 65)

    # ── optionally load LSTM model ────────────────────────────────────────────
    lstm_model, scaler = None, None
    ckpt_path = project_root / cfg["training"]["checkpoint_dir"] / "best_model.pt"

    if not args.no_model and ckpt_path.exists():
        from model.inference import load_checkpoint
        lstm_model, scaler, _ = load_checkpoint(ckpt_path, device=device)
        print(f"  LSTM model loaded  : {ckpt_path.name}")
    elif args.no_model:
        print("  LSTM model         : skipped (--no-model flag)")
    else:
        print(f"  LSTM model         : NOT FOUND at {ckpt_path}")
        print("  Tip: run `python -m model.train` first for better observations.")

    # ── load trace ────────────────────────────────────────────────────────────
    from data.loader import load_trace
    df = load_trace(cfg=cfg)
    print(f"  Trace loaded       : {len(df):,} timesteps")

    # ── env factory ───────────────────────────────────────────────────────────
    from rl_env.k8s_env import K8sAutoscalingEnv
    from rl_env.wrappers import TransitionLogger

    def env_factory():
        env = K8sAutoscalingEnv(
            cfg=cfg,
            model=lstm_model,
            scaler=scaler,
            trace_df=df,
            device=device,
        )
        # Optionally wrap with TransitionLogger to persist (s,a,r,s',done) data
        if cfg["ppo"].get("log_transitions", False):
            transitions_path = project_root / "data" / "sim_transitions.csv"
            env = TransitionLogger(env, output_path=transitions_path)
            print(f"  Transition logging enabled → {transitions_path}")
        return env

    # ── run PPO ───────────────────────────────────────────────────────────────
    from rl_agent.ppo import train_ppo

    print()
    history = train_ppo(
        cfg=cfg,
        env_factory=env_factory,
        device=device,
        verbose=True,
        total_timesteps_override=args.total_timesteps,
        log_interval_override=args.log_interval,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    import numpy as np
    print()
    print("=" * 65)
    print("  Training Summary")
    print("=" * 65)
    if history["episode_rewards"]:
        rewards = history["episode_rewards"]
        slos    = history["slo_pcts"]
        reps    = history["avg_replicas"]
        print(f"  Episodes completed : {len(rewards)}")
        print(f"  Mean reward (last 10): {np.mean(rewards[-10:]):+.2f}")
        print(f"  Best episode reward  : {max(rewards):+.2f}")
        print(f"  Mean SLO% (last 10)  : {np.mean(slos[-10:]):.1f}%")
        print(f"  Mean replicas        : {np.mean(reps[-10:]):.2f}")
    print()
    print("  Next step: python -m rl_agent.evaluate_rl")
    print("=" * 65)


if __name__ == "__main__":
    main()
