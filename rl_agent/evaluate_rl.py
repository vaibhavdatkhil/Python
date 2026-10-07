"""
rl_agent/evaluate_rl.py
────────────────────────
Phase 7: Evaluate the trained PPO agent and compare against baselines.

Metrics
───────
  • Episode reward (PPO vs random vs HPA)
  • SLO compliance %  (cpu_per_pod ≤ threshold each step)
  • Average replica count (proxy for infrastructure cost)
  • Reward curve over training (from checkpoint history)

Baselines
─────────
  • Random: uniformly random actions (historical baseline)
  • HPA: reactive Horizontal Pod Autoscaler using standard K8s formula
    (desired_replicas = ceil(current * current_util / target_util))

Outputs
───────
  outputs/rl_evaluation.png  — 4-panel figure:
    Panel 1: Training reward curve (episode rewards + rolling mean)
    Panel 2: Replica trace — PPO vs random vs HPA over one episode
    Panel 3: CPU-per-pod trace — PPO vs random vs HPA (SLO threshold line)
    Panel 4: Action distribution — PPO vs random (bar chart)

Usage
─────
  python -m rl_agent.evaluate_rl
  python -m rl_agent.evaluate_rl --checkpoint checkpoints/ppo_agent_final.pt
  python -m rl_agent.evaluate_rl --no-model
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import yaml


# ── Rollout helper ────────────────────────────────────────────────────────────

def run_episode(env, policy=None, seed: int = 42) -> dict:
    """Run one full episode.

    If `policy` is None, actions are sampled uniformly at random (baseline).

    Returns
    -------
    dict with keys:
      rewards, replicas, cpu_per_pod, actions, slo_met
      total_reward, slo_pct, avg_replicas
    """
    obs_np, _ = env.reset(seed=seed)
    obs        = torch.from_numpy(obs_np).float()

    rewards    = []
    replicas   = []
    cpu_per_pod_list = []
    actions    = []
    slo_met    = []

    done = False
    while not done:
        if policy is not None:
            with torch.no_grad():
                action_t, _, _, _ = policy.act(obs.unsqueeze(0))
                action = int(action_t.item())
        else:
            action = env.action_space.sample()

        obs_np, reward, terminated, truncated, info = env.step(action)
        obs  = torch.from_numpy(obs_np).float()
        done = terminated or truncated

        rewards.append(float(reward))
        replicas.append(int(info["replicas"]))
        cpu_per_pod_list.append(float(info["cpu_per_pod"]))
        actions.append(int(action))
        slo_met.append(bool(info["slo_met"]))

    n = max(len(rewards), 1)
    return {
        "rewards":      rewards,
        "replicas":     replicas,
        "cpu_per_pod":  cpu_per_pod_list,
        "actions":      actions,
        "slo_met":      slo_met,
        "total_reward": sum(rewards),
        "slo_pct":      sum(slo_met) / n * 100,
        "avg_replicas": np.mean(replicas),
    }


# ── HPA Baseline Policy ───────────────────────────────────────────────────────

def run_episode_hpa(env, cfg: dict, seed: int = 42) -> dict:
    """Run one full episode with a reactive HPA policy.

    Uses the standard Kubernetes HPA formula (reused from dashboard/app.py):
      desired_replicas = ceil(current_replicas * (current_cpu / target_cpu))
    with actuation lag matching the env's scale_lag_steps.

    Returns
    -------
    dict with keys:
      rewards, replicas, cpu_per_pod, actions, slo_met
      total_reward, slo_pct, avg_replicas
    """
    obs_np, _ = env.reset(seed=seed)

    rewards    = []
    replicas   = []
    cpu_per_pod_list = []
    actions    = []
    slo_met    = []

    # HPA state
    current_replicas = cfg["rl_env"]["initial_replicas"]
    min_r = cfg["rl_env"]["min_replicas"]
    max_r = cfg["rl_env"]["max_replicas"]
    target_slo = cfg["rl_env"]["slo_threshold"]
    lag = cfg["rl_env"]["scale_lag_steps"]

    # Action history for applying lag
    action_queue = []

    done = False
    step = 0
    while not done:
        # HPA decision: compute desired replicas based on observed CPU
        # Extract cpu_per_pod from info after taking an action
        # For the first step, use a neutral action (action=2 → delta=0)
        if step == 0:
            action = 2  # hold
        else:
            # Use lagged observation for HPA (mimics metrics scraping delay)
            lagged_idx = max(0, step - lag)
            lagged_cpu = cpu_per_pod_list[lagged_idx] if lagged_idx < len(cpu_per_pod_list) else target_slo
            lagged_replicas = replicas[lagged_idx] if lagged_idx < len(replicas) else current_replicas

            # Standard HPA formula
            desired = int(np.ceil(lagged_replicas * (lagged_cpu / max(target_slo, 1e-3))))
            desired = int(np.clip(desired, min_r, max_r))

            # Convert desired replicas to action (delta)
            delta = desired - current_replicas
            # Map delta to discrete action space: {-2, -1, 0, +1, +2} → {0, 1, 2, 3, 4}
            if delta <= -2:
                action = 0
            elif delta == -1:
                action = 1
            elif delta == 0:
                action = 2
            elif delta == 1:
                action = 3
            else:  # delta >= 2
                action = 4

        obs_np, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        replicas.append(int(info["replicas"]))
        cpu_per_pod_list.append(float(info["cpu_per_pod"]))
        actions.append(int(action))
        slo_met.append(bool(info["slo_met"]))

        current_replicas = int(info["replicas"])
        step += 1

    n = max(len(rewards), 1)
    return {
        "rewards":      rewards,
        "replicas":     replicas,
        "cpu_per_pod":  cpu_per_pod_list,
        "actions":      actions,
        "slo_met":      slo_met,
        "total_reward": sum(rewards),
        "slo_pct":      sum(slo_met) / n * 100,
        "avg_replicas": np.mean(replicas),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _rolling_mean(x: list, w: int = 20) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_evaluation_plot(
    training_history: Optional[dict],
    ppo_result:       dict,
    random_result:    dict,
    hpa_result:       dict,
    slo_threshold:    float,
    output_path:      Path,
) -> None:
    """Generate the 4-panel evaluation figure and save as PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    # Style constants
    BG       = "#0d1117"
    PANEL_BG = "#161b22"
    GRID_C   = "#30363d"
    PPO_C    = "#58a6ff"
    RAND_C   = "#f78166"
    HPA_C    = "#a78bfa"     # purple for HPA
    SLO_C    = "#3fb950"
    MEAN_C   = "#ffa657"

    def _style_ax(ax, title: str) -> None:
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color="white", fontsize=12, pad=10, fontweight="bold")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)

    # ── Panel 1: Training Reward Curve ────────────────────────────────────────
    ax1 = axes[0, 0]
    _style_ax(ax1, "[1] Training Reward Curve")
    if training_history and training_history.get("episode_rewards"):
        ep_rewards = training_history["episode_rewards"]
        x_ep       = np.arange(1, len(ep_rewards) + 1)
        ax1.plot(x_ep, ep_rewards, color=PPO_C, alpha=0.3, linewidth=0.8, label="Episode")
        rm = _rolling_mean(ep_rewards, w=min(20, max(1, len(ep_rewards) // 5)))
        ax1.plot(x_ep, rm, color=MEAN_C, linewidth=2.0, label="Rolling mean (20 ep)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor="white", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No training history available",
                 ha="center", va="center", color="#8b949e", transform=ax1.transAxes)

    # ── Panel 2: Replica Trace ────────────────────────────────────────────────
    ax2 = axes[0, 1]
    _style_ax(ax2, "[2] Replica Trace -- PPO vs HPA vs Random")
    steps = np.arange(len(ppo_result["replicas"]))
    # Downsample if very long
    ds = max(1, len(steps) // 2000)
    ax2.plot(steps[::ds], np.array(ppo_result["replicas"])[::ds],
             color=PPO_C,  linewidth=1.4, label=f"PPO (avg={ppo_result['avg_replicas']:.1f})")
    ax2.plot(steps[::ds], np.array(hpa_result["replicas"])[::ds],
             color=HPA_C,  linewidth=1.2, alpha=0.85, label=f"HPA (avg={hpa_result['avg_replicas']:.1f})")
    steps_r = np.arange(len(random_result["replicas"]))
    ax2.plot(steps_r[::ds], np.array(random_result["replicas"])[::ds],
             color=RAND_C, linewidth=1.0, alpha=0.6, label=f"Random (avg={random_result['avg_replicas']:.1f})")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Replicas")
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor="white", fontsize=9)

    # ── Panel 3: CPU-per-Pod Trace ────────────────────────────────────────────
    ax3 = axes[1, 0]
    _style_ax(ax3, "[3] CPU-per-Pod -- PPO vs HPA vs Random")
    cpp_ppo  = np.array(ppo_result["cpu_per_pod"])
    cpp_hpa  = np.array(hpa_result["cpu_per_pod"])
    cpp_rand = np.array(random_result["cpu_per_pod"])
    steps_p  = np.arange(len(cpp_ppo))
    steps_h  = np.arange(len(cpp_hpa))
    steps_r2 = np.arange(len(cpp_rand))
    ds2 = max(1, max(len(steps_p), len(steps_h), len(steps_r2)) // 2000)
    ax3.plot(steps_p[::ds2], cpp_ppo[::ds2],
             color=PPO_C,  linewidth=1.2,
             label=f"PPO  SLO={ppo_result['slo_pct']:.1f}%")
    ax3.plot(steps_h[::ds2], cpp_hpa[::ds2],
             color=HPA_C,  linewidth=1.0, alpha=0.85,
             label=f"HPA  SLO={hpa_result['slo_pct']:.1f}%")
    ax3.plot(steps_r2[::ds2], cpp_rand[::ds2],
             color=RAND_C, linewidth=0.8, alpha=0.6,
             label=f"Random SLO={random_result['slo_pct']:.1f}%")
    ax3.axhline(slo_threshold, color=SLO_C, linestyle="--", linewidth=1.5,
                label=f"SLO threshold ({slo_threshold})")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("CPU / Pod")
    ax3.set_ylim(0, max(1.2, cpp_ppo.max() * 1.1, cpp_hpa.max() * 1.1, cpp_rand.max() * 1.1))
    ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor="white", fontsize=9)

    # ── Panel 4: Action Distribution ──────────────────────────────────────────
    ax4 = axes[1, 1]
    _style_ax(ax4, "[4] Action Distribution -- PPO vs Random")
    action_labels = ["-2", "-1", " 0", "+1", "+2"]
    x_pos         = np.arange(5)
    width         = 0.35

    ppo_counts  = np.bincount(ppo_result["actions"],  minlength=5)
    rand_counts = np.bincount(random_result["actions"], minlength=5)
    ppo_pct     = ppo_counts  / max(ppo_counts.sum(),  1) * 100
    rand_pct    = rand_counts / max(rand_counts.sum(), 1) * 100

    ax4.bar(x_pos - width / 2, ppo_pct,  width, color=PPO_C,  label="PPO",    alpha=0.9)
    ax4.bar(x_pos + width / 2, rand_pct, width, color=RAND_C, label="Random", alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(action_labels, color="#8b949e")
    ax4.set_xlabel("Action (Delta replicas)")
    ax4.set_ylabel("Usage (%)")
    ax4.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor="white", fontsize=9)

    # ── overall title ─────────────────────────────────────────────────────────
    ppo_vs_hpa = ppo_result["total_reward"] - hpa_result["total_reward"]
    sign = "+" if ppo_vs_hpa >= 0 else ""
    fig.suptitle(
        f"PPO Agent Evaluation   |   "
        f"PPO: {ppo_result['total_reward']:+.0f}   "
        f"HPA: {hpa_result['total_reward']:+.0f}   "
        f"Random: {random_result['total_reward']:+.0f}   "
        f"PPO vs HPA={sign}{ppo_vs_hpa:.0f}",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Plot saved -> {output_path.resolve()}", flush=True)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained PPO agent against a random baseline."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to PPO checkpoint (default: checkpoints/ppo_agent.pt).",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip loading LSTM model.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device. Auto-detected by default.",
    )
    parser.add_argument(
        "--benchmark", choices=["none", "azure", "alibaba", "all"], default="none",
        help="Evaluate on real cloud benchmark traces (azure, alibaba, all).",
    )
    parser.add_argument(
        "--eval-trace", default=None,
        help="Path to a custom CSV trace file for evaluation.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / args.config

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)

    # ── resolve checkpoint path ───────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dir  = project_root / cfg["ppo"]["checkpoint_dir"]
        ckpt_path = ckpt_dir / "ppo_agent.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "ppo_agent_final.pt"

    if not ckpt_path.exists():
        print(f"[ERROR] PPO checkpoint not found at {ckpt_path}", flush=True)
        print("  Run `python -m rl_agent.train_rl` first.", flush=True)
        raise SystemExit(1)

    print("=" * 65, flush=True)
    print("  K8s RL Autoscaling -- Phase 7: PPO Agent Evaluation", flush=True)
    print("=" * 65, flush=True)
    print(f"  Checkpoint : {ckpt_path}", flush=True)
    print(f"  Device     : {device}", flush=True)

    # ── load LSTM model (optional) ────────────────────────────────────────────
    lstm_model, scaler = None, None
    lstm_ckpt = project_root / cfg["training"]["checkpoint_dir"] / "best_model.pt"
    if not args.no_model and lstm_ckpt.exists():
        from model.inference import load_checkpoint as load_lstm
        lstm_model, scaler, _ = load_lstm(lstm_ckpt, device=device)
        print(f"  LSTM model : {lstm_ckpt.name}", flush=True)

    # ── load PPO policy ───────────────────────────────────────────────────────
    from rl_agent.ppo import load_ppo_checkpoint
    policy, raw_ckpt = load_ppo_checkpoint(ckpt_path, device=device)
    training_history = raw_ckpt.get("history", None)

    slo_threshold = float(cfg["rl_env"]["slo_threshold"])
    output_dir    = project_root / cfg["evaluation"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    def _eval_and_report(trace_data: pd.DataFrame, label: str, plot_file: str):
        from rl_env.k8s_env import K8sAutoscalingEnv

        env_ppo = K8sAutoscalingEnv(
            cfg=cfg, model=lstm_model, scaler=scaler,
            trace_df=trace_data, device=device,
        )
        env_rand = K8sAutoscalingEnv(
            cfg=cfg, model=None, scaler=None,
            trace_df=trace_data, device=device,
        )
        env_hpa = K8sAutoscalingEnv(
            cfg=cfg, model=None, scaler=None,
            trace_df=trace_data, device=device,
        )

        print(f"\n  Running PPO episode on {label} ...", flush=True)
        p_res = run_episode(env_ppo, policy=policy, seed=42)

        print(f"  Running HPA-baseline episode on {label} ...", flush=True)
        h_res = run_episode_hpa(env_hpa, cfg=cfg, seed=42)

        print(f"  Running random-baseline episode on {label} ...", flush=True)
        r_res = run_episode(env_rand, policy=None, seed=42)

        print("", flush=True)
        print(f"  Evaluation Comparison: {label}", flush=True)
        print(f"  {'Metric':<28} {'PPO':>12} {'HPA':>12} {'Random':>12}", flush=True)
        print(f"  {'-'*68}", flush=True)
        print(f"  {'Total reward':<28} {p_res['total_reward']:>+12.1f} {h_res['total_reward']:>+12.1f} {r_res['total_reward']:>+12.1f}", flush=True)
        print(f"  {'SLO compliance (%)':<28} {p_res['slo_pct']:>11.1f}% {h_res['slo_pct']:>11.1f}% {r_res['slo_pct']:>11.1f}%", flush=True)
        print(f"  {'Avg replicas':<28} {p_res['avg_replicas']:>12.2f} {h_res['avg_replicas']:>12.2f} {r_res['avg_replicas']:>12.2f}", flush=True)
        print(f"  {'Episode steps':<28} {len(p_res['rewards']):>12,} {len(h_res['rewards']):>12,} {len(r_res['rewards']):>12,}", flush=True)

        d_hpa = p_res["slo_pct"] - h_res["slo_pct"]
        s_hpa = "+" if d_hpa >= 0 else ""
        d_rand = p_res["slo_pct"] - r_res["slo_pct"]
        s_rand = "+" if d_rand >= 0 else ""
        print(f"\n  SLO improvement vs HPA     : {s_hpa}{d_hpa:.1f}%", flush=True)
        print(f"  SLO improvement vs random  : {s_rand}{d_rand:.1f}%", flush=True)

        p_plot = output_dir / plot_file
        make_evaluation_plot(
            training_history=training_history if plot_file == "rl_evaluation.png" else None,
            ppo_result=p_res,
            random_result=r_res,
            hpa_result=h_res,
            slo_threshold=slo_threshold,
            output_path=p_plot,
        )
        print(f"  Plot saved : {p_plot.resolve()}", flush=True)
        return p_res, h_res, r_res

    # ── 1. Evaluate on primary held-out trace split ───────────────────────────
    from data.loader import load_trace
    df = load_trace(cfg=cfg)
    train_frac = float(cfg["preprocessing"].get("train_frac", 0.8))
    split_idx  = int(len(df) * train_frac)
    eval_df    = df.iloc[split_idx:].reset_index(drop=True)

    print(f"  Trace (total)  : {len(df):,} timesteps", flush=True)
    print(f"  Eval slice     : [{split_idx:,} : {len(df):,}]  ({len(eval_df):,} steps, last {100*(1-train_frac):.0f}%)", flush=True)

    _eval_and_report(eval_df, "Held-out Evaluation Split", "rl_evaluation.png")

    # ── 2. Evaluate on benchmark traces if requested ──────────────────────────
    benchmarks_to_run = []
    if args.eval_trace:
        benchmarks_to_run.append(("Custom Trace", Path(args.eval_trace), "rl_eval_custom.png"))
    if args.benchmark in ("azure", "all"):
        benchmarks_to_run.append(("Azure VM Trace", project_root / "data" / "azure_vm_workload_trace.csv", "rl_eval_azure.png"))
    if args.benchmark in ("alibaba", "all"):
        benchmarks_to_run.append(("Alibaba Cluster Trace", project_root / "data" / "alibaba_cluster_trace.csv", "rl_eval_alibaba.png"))

    for label, path, plot_file in benchmarks_to_run:
        if not path.exists():
            print(f"\n  [WARN] Benchmark trace not found at {path}, skipping.")
            continue
        bench_df = pd.read_csv(path, parse_dates=["timestamp"])
        if bench_df["cpu_util"].max() > 1.5:
            bench_df["cpu_util"] = bench_df["cpu_util"] / 100.0
        bench_df["cpu_util"] = bench_df["cpu_util"].clip(0.0, 1.0).astype(np.float32)
        print(f"\n{'=' * 65}")
        print(f"  Benchmark Evaluation: {label} ({len(bench_df):,} timesteps)")
        print(f"{'=' * 65}")
        _eval_and_report(bench_df, label, plot_file)

    print("", flush=True)
    print("=" * 65, flush=True)
    print("  Evaluation complete.", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
