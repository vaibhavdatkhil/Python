"""
explainability/shap_explain.py
───────────────────────────────
Phase 8: SHAP-based explainability for:
  1. LSTM Forecaster  — KernelSHAP on the mean MC-Dropout output
  2. PPO Actor        — KernelSHAP on the actor logits

Algorithm
─────────
LSTM Explainer
  • Wraps the MC-Dropout mean-forecast as a callable that accepts a 2-D
    numpy matrix (n_samples × window_size) and returns a 1-D array
    (mean absolute forecast error proxy, or just mean of output).
  • Uses shap.KernelExplainer with a background of k random windows.
  • Saves a summary bar plot (feature importances = timestep importances).

PPO Explainer
  • Wraps the actor's softmax probabilities as a callable.
  • Observation vector: [cpu_window(W), forecast_mean(H), forecast_std(H),
    rep_norm(1), cpp_norm(1)].
  • Uses shap.KernelExplainer.
  • Saves a bar chart of per-feature mean |SHAP| value.

Usage
─────
  python -m explainability.shap_explain
  python -m explainability.shap_explain --no-model
  python -m explainability.shap_explain --n-bg 20 --n-explain 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

# ── SHAP import guard ─────────────────────────────────────�def explain_lstm(
    model,
    scaler,
    X_te:       np.ndarray,          # (N, window, 1)
    cfg:        dict,
    output_dir: Path,
    n_bg:       int = 30,            # background samples for GradientExplainer
    n_explain:  int = 60,            # samples to explain
    device:     torch.device = torch.device("cpu"),
) -> Path:
    """Run GradientExplainer on the LSTM forecaster and save a summary plot.

    Uses shap.GradientExplainer (gradient-based, ~1000x faster than KernelSHAP
    on CPU) to attribute each look-back timestep's contribution to the mean
    forecast output.

    Parameters
    ----------
    model      : trained LSTMForecast in eval mode.
    scaler     : fitted MinMaxScaler.
    X_te       : test windows, shape (N, window, 1).
    cfg        : full config dict.
    output_dir : directory for the output PNG.
    n_bg       : background samples for GradientExplainer baseline.
    n_explain  : number of test samples to explain.
    device     : torch device.

    Returns
    -------
    Path to the saved PNG.
    """
    if not _SHAP_OK:
        raise ImportError(
            "shap is not installed. Run: pip install shap>=0.45.0"
        )

    W = cfg["preprocessing"]["window_size"]
    H = cfg["preprocessing"]["horizon"]

    # GradientExplainer requires the model to be in train-compatible mode
    # (it uses autograd), but we disable MC dropout for stable gradients.
    model.eval()
    # Freeze dropout so gradients are deterministic
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    # ── Wrapper: scalar output (mean of horizon forecast) ─────────────────────
    class _LSTMWrapper(torch.nn.Module):
        """Wrap LSTMForecast: (B, W, 1) -> (B,) mean forecast."""
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)   # (B, H)
            return out.mean(dim=1, keepdim=True)  # (B, 1)

    wrapped = _LSTMWrapper(model).to(device)
    wrapped.eval()

    # ── prepare data ─────────────────────────────────────────────────────────
    X_raw = X_te[:, :, 0]   # (N, W)
    rng   = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_raw), size=min(n_bg, len(X_raw)), replace=False)
    ex_idx = rng.choice(len(X_raw), size=min(n_explain, len(X_raw)), replace=False)

    # Tensors: shape (N, W, 1) — LSTM input format
    X_bg_t = torch.from_numpy(
        X_raw[bg_idx].astype(np.float32)[:, :, np.newaxis]
    ).to(device)
    X_ex_t = torch.from_numpy(
        X_raw[ex_idx].astype(np.float32)[:, :, np.newaxis]
    ).to(device)

    print(f"  [SHAP/LSTM] GradientExplainer | bg={len(bg_idx)} explain={len(ex_idx)}", flush=True)

    explainer   = shap.GradientExplainer(wrapped, X_bg_t)
    shap_values = explainer.shap_values(X_ex_t)
    # shap_values: list of 1 array of shape (n_explain, W, 1)
    if isinstance(shap_values, list):
        sv = shap_values[0]          # (n_explain, W, 1)
    else:
        sv = shap_values             # (n_explain, W, 1)
    sv = sv[:, :, 0]                 # (n_explain, W) — drop last dim

n(n_bg, len(X_flat)), replace=False)
    ex_idx = rng.choice(len(X_flat), size=min(n_explain, len(X_flat)), replace=False)

    X_bg = X_flat[bg_idx]              # (n_bg, W)
    X_ex = X_flat[ex_idx]              # (n_explain, W)

    print(f"  [SHAP/LSTM] Background: {len(X_bg)}, Explain: {len(X_ex)}", flush=True)

    explainer   = shap.KernelExplainer(_predict, X_bg, link="identity")
    shap_values = explainer.shap_values(X_ex, nsamples=100, silent=True)
    # shap_values: (n_explain, W)

    # ── plot ──────────────────────────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)   # (W,)
    timestep_labels = [f"t-{W - i}" for i in range(W)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")

    BG       = "#0d1117"
    PANEL_BG = "#161b22"
    GRID_C   = "#30363d"
    BAR_C    = "#58a6ff"

    # ── Panel 1: Mean |SHAP| bar chart ───────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    # Show only top-20 timesteps for readability
    top_n = min(20, W)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    ax.barh(
        [timestep_labels[i] for i in top_idx],
        mean_abs[top_idx],
        color=BAR_C, alpha=0.9,
    )
    ax.set_xlabel("Mean |SHAP value|", color="#8b949e", fontsize=10)
    ax.set_title("LSTM Feature Importance (top timesteps)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(axis="x", color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)

    # ── Panel 2: SHAP importance over time ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(range(W), mean_abs, color=BAR_C, linewidth=1.5)
    ax2.fill_between(range(W), mean_abs, alpha=0.2, color=BAR_C)
    ax2.set_xlabel("Look-back timestep (0 = oldest)", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Mean |SHAP value|", color="#8b949e", fontsize=10)
    ax2.set_title("LSTM Importance Over Look-back Window",
                  color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_C)
    ax2.grid(color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)

    fig.suptitle(
        "SHAP Explainability — LSTM MC-Dropout Forecaster\n"
        f"KernelSHAP | bg={len(X_bg)} | explained={len(X_ex)}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "shap_lstm.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [SHAP/LSTM] Saved → {out_path.resolve()}", flush=True)
    return out_path


# ── PPO explainability ────────────────────────────────────────────────────────

def explain_ppo(
    policy,
    obs_samples:  np.ndarray,       # (N, obs_dim)
    cfg:          dict,
    output_dir:   Path,
    n_bg:         int = 30,
    n_explain:    int = 60,
    device:       torch.device = torch.device("cpu"),
) -> Path:
    """Run KernelSHAP on the PPO actor's action probabilities.

    Parameters
    ----------
    policy      : ActorCritic in eval mode.
    obs_samples : flattened observation vectors, shape (N, obs_dim).
    cfg         : full config dict.
    output_dir  : directory for the output PNG.
    n_bg / n_explain : KernelSHAP sizes.
    device      : torch device.

    Returns
    -------
    Path to the saved PNG.
    """
    if not _SHAP_OK:
        raise ImportError("shap is not installed. Run: pip install shap>=0.45.0")

    W = cfg["preprocessing"]["window_size"]
    H = cfg["preprocessing"]["horizon"]
    n_actions = 5  # [-2, -1, 0, +1, +2]

    # Build human-readable feature names
    feat_names = (
        [f"cpu_t-{W - i}" for i in range(W)]           # CPU window
        + [f"fc_mean_h{i+1}" for i in range(H)]        # forecast mean
        + [f"fc_std_h{i+1}"  for i in range(H)]        # forecast std
        + ["replicas_norm", "cpu_per_pod_norm"]         # scalar features
    )

    # ── wrap policy as numpy→numpy callable ──────────────────────────────────
    def _actor_probs(obs_flat: np.ndarray) -> np.ndarray:
        """
        obs_flat : (B, obs_dim)
        returns  : (B, n_actions) — softmax action probabilities
        """
        x = torch.from_numpy(obs_flat.astype(np.float32)).to(device)
        with torch.no_grad():
            features = policy.backbone(x)
            logits   = policy.actor_head(features)
            probs    = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # ── prepare background + explain data ─────────────────────────────────────
    rng    = np.random.default_rng(42)
    bg_idx = rng.choice(len(obs_samples), size=min(n_bg, len(obs_samples)), replace=False)
    ex_idx = rng.choice(len(obs_samples), size=min(n_explain, len(obs_samples)), replace=False)

    X_bg = obs_samples[bg_idx]
    X_ex = obs_samples[ex_idx]

    print(f"  [SHAP/PPO] Background: {len(X_bg)}, Explain: {len(X_ex)}", flush=True)

    # KernelExplainer for multi-output (action probabilities)
    explainer   = shap.KernelExplainer(_actor_probs, X_bg, link="identity")
    shap_values = explainer.shap_values(X_ex, nsamples=100, silent=True)
    # shap_values: list of (n_explain, obs_dim), one per action

    # Aggregate: mean |SHAP| across all actions and all explained samples
    shap_arr = np.array(shap_values)              # (n_actions, n_explain, obs_dim)
    mean_abs = np.abs(shap_arr).mean(axis=(0, 1)) # (obs_dim,)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")

    BG       = "#0d1117"
    PANEL_BG = "#161b22"
    GRID_C   = "#30363d"
    COLORS   = ["#58a6ff", "#3fb950", "#f78166", "#ffa657", "#bc8cff"]
    ACTION_NAMES = ["Scale -2", "Scale -1", "Hold 0", "Scale +1", "Scale +2"]

    # ── Panel 1: Top-20 features (aggregated across actions) ─────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    top_n   = min(20, len(feat_names))
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    ax.barh(
        [feat_names[i] for i in top_idx],
        mean_abs[top_idx],
        color="#58a6ff", alpha=0.9,
    )
    ax.set_xlabel("Mean |SHAP value|", color="#8b949e", fontsize=10)
    ax.set_title("PPO Actor Feature Importance (top-20)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(axis="x", color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)

    # ── Panel 2: Per-action importance for key feature groups ─────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)

    # Summarise by feature group
    groups = {
        "CPU window": list(range(W)),
        "Forecast mean": list(range(W, W + H)),
        "Forecast std": list(range(W + H, W + 2*H)),
        "Replicas": [W + 2*H],
        "CPU/pod": [W + 2*H + 1],
    }
    group_names = list(groups.keys())
    x_pos = np.arange(len(group_names))
    bar_w = 0.15

    for ai, (action_name, color) in enumerate(zip(ACTION_NAMES, COLORS)):
        sv_action = shap_arr[ai]  # (n_explain, obs_dim)
        group_means = []
        for g_idx in groups.values():
            group_means.append(np.abs(sv_action[:, g_idx]).mean())
        ax2.bar(
            x_pos + ai * bar_w - bar_w * 2,
            group_means, bar_w,
            color=color, label=action_name, alpha=0.85,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(group_names, color="#8b949e", fontsize=9, rotation=15)
    ax2.set_ylabel("Mean |SHAP value|", color="#8b949e", fontsize=10)
    ax2.set_title("Per-Action Feature Group Importance",
                  color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor="white", fontsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_C)
    ax2.grid(axis="y", color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)

    fig.suptitle(
        "SHAP Explainability — PPO Actor (Scaling Policy)\n"
        f"KernelSHAP | bg={len(X_bg)} | explained={len(X_ex)}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "shap_ppo.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [SHAP/PPO] Saved → {out_path.resolve()}", flush=True)
    return out_path


# ── Observation collector helper ──────────────────────────────────────────────

def collect_obs_samples(
    env,
    policy,
    n_samples: int = 300,
    device:    torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Roll out the PPO policy and collect observation vectors.

    Parameters
    ----------
    env      : K8sAutoscalingEnv instance (already reset not required).
    policy   : ActorCritic or None (random actions if None).
    n_samples: how many observations to collect.
    device   : torch device.

    Returns
    -------
    obs_arr : np.ndarray, shape (n_samples, obs_dim)
    """
    obs_list = []
    obs_np, _ = env.reset(seed=7)
    obs = torch.from_numpy(obs_np).float().to(device)

    while len(obs_list) < n_samples:
        obs_list.append(obs_np.copy())
        if policy is not None:
            with torch.no_grad():
                action_t, _, _, _ = policy.act(obs.unsqueeze(0))
                action = int(action_t.item())
        else:
            action = env.action_space.sample()

        obs_np, _, terminated, truncated, _ = env.step(action)
        obs = torch.from_numpy(obs_np).float().to(device)
        if terminated or truncated:
            obs_np, _ = env.reset()
            obs = torch.from_numpy(obs_np).float().to(device)

    return np.array(obs_list[:n_samples], dtype=np.float32)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the LSTM and PPO models."
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--no-model",  action="store_true",
                        help="Skip LSTM model (PPO still explained).")
    parser.add_argument("--no-lstm",   action="store_true",
                        help="Skip LSTM SHAP (faster).")
    parser.add_argument("--no-ppo",    action="store_true",
                        help="Skip PPO SHAP.")
    parser.add_argument("--n-bg",      type=int, default=30,
                        help="KernelSHAP background samples.")
    parser.add_argument("--n-explain", type=int, default=60,
                        help="Samples to explain.")
    parser.add_argument("--device",    default=None)
    args = parser.parse_args()

    if not _SHAP_OK:
        print("[ERROR] SHAP is not installed. Run: pip install shap>=0.45.0")
        raise SystemExit(1)

    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / args.config

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)

    output_dir = project_root / cfg.get("evaluation", {}).get("output_dir", "outputs")

    print("=" * 65, flush=True)
    print("  K8s RL Autoscaling - Phase 8: SHAP Explainability", flush=True)
    print("=" * 65, flush=True)
    print(f"  Device     : {device}", flush=True)
    print(f"  Output dir : {output_dir}", flush=True)

    # ── load trace ────────────────────────────────────────────────────────────
    from data.loader import load_trace
    df = load_trace(cfg=cfg)

    # ── optionally load LSTM ──────────────────────────────────────────────────
    lstm_model, scaler = None, None
    lstm_ckpt = project_root / cfg["training"]["checkpoint_dir"] / "best_model.pt"
    if not args.no_model and lstm_ckpt.exists():
        from model.inference import load_checkpoint as load_lstm
        lstm_model, scaler, _ = load_lstm(lstm_ckpt, device=device)
        print(f"  LSTM model : {lstm_ckpt.name}", flush=True)
    else:
        print("  LSTM model : not loaded", flush=True)

    # -- LSTM SHAP -------------------------------------------------------
    if not args.no_lstm and lstm_model is not None:
        print("\n  -- LSTM Explainability ---------------------------------", flush=True)
        from preprocessing.pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline(cfg)
        splits, _ = pipeline.fit_transform(df)
        _, _, _, _, X_te, _ = splits

        explain_lstm(
            model=lstm_model,
            scaler=scaler,
            X_te=X_te,
            cfg=cfg,
            output_dir=output_dir,
            n_bg=args.n_bg,
            n_explain=args.n_explain,
            device=device,
        )
    elif not args.no_lstm:
        print("  Skipping LSTM SHAP (no checkpoint).", flush=True)

    # -- PPO SHAP --------------------------------------------------------
    if not args.no_ppo:
        print("\n  -- PPO Explainability ----------------------------------", flush=True)
        ppo_ckpt_dir = project_root / cfg["ppo"]["checkpoint_dir"]
        ppo_ckpt     = ppo_ckpt_dir / "ppo_agent.pt"
        if not ppo_ckpt.exists():
            ppo_ckpt = ppo_ckpt_dir / "ppo_agent_final.pt"

        if not ppo_ckpt.exists():
            print("  [WARN] PPO checkpoint not found — skipping PPO SHAP.", flush=True)
        else:
            from rl_agent.ppo import load_ppo_checkpoint
            from rl_env.k8s_env import K8sAutoscalingEnv

            policy, _ = load_ppo_checkpoint(ppo_ckpt, device=device)

            env = K8sAutoscalingEnv(
                cfg=cfg, model=lstm_model, scaler=scaler,
                trace_df=df, device=device,
            )
            n_collect = min(args.n_bg + args.n_explain + 50, 400)
            print(f"  Collecting {n_collect} obs samples …", flush=True)
            obs_samples = collect_obs_samples(
                env, policy, n_samples=n_collect, device=device
            )
            env.close()

            explain_ppo(
                policy=policy,
                obs_samples=obs_samples,
                cfg=cfg,
                output_dir=output_dir,
                n_bg=args.n_bg,
                n_explain=args.n_explain,
                device=device,
            )

    print("\n" + "=" * 65, flush=True)
    print("  SHAP explainability complete.", flush=True)
    print(f"  Plots saved to: {output_dir.resolve()}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
