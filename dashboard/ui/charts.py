"""
dashboard/ui/charts.py
───────────────────────
All Plotly figure builders for the K8s RL Autoscaling Dashboard.

Every function accepts a ``theme: str`` argument and applies the active
theme palette. No hardcoded hex values — all colors come from get_palette().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import get_palette, get_plotly_template


# ── Private layout helper ─────────────────────────────────────────────────────

def _get_layout(
    title: str = "",
    height: int = 300,
    theme: str = "dark",
) -> dict:
    """Return a Plotly layout dict matching the active theme.

    Drop-in replacement for the old ``_dark_layout()`` in app.py.

    Parameters
    ----------
    title  : Chart title string.
    height : Chart height in pixels.
    theme  : ``"dark"`` or ``"light"``.

    Returns
    -------
    dict
        Keyword arguments for ``fig.update_layout(**...)``.
    """
    p = get_palette(theme)
    base = get_plotly_template(theme)
    base["title"] = title
    base["height"] = height
    # Use surface color so charts look polished on the background
    base["plot_bgcolor"] = p["surface"]
    base["paper_bgcolor"] = p["surface"]
    return base



# ── Composite scaling activity figure ────────────────────────────────────────

def make_scaling_activity_figure(
    hpa_replicas: np.ndarray,
    rl_replicas: np.ndarray,
    cpu_series: np.ndarray,
    mean_fc: Optional[np.ndarray],
    lower_fc: Optional[np.ndarray],
    upper_fc: Optional[np.ndarray],
    slo_threshold: float = 0.70,
    is_simulated: bool = True,
    theme: str = "dark",
) -> go.Figure:
    """Build the 3-panel composite Pod Scaling Activity figure.

    Panel A (row 1, 50%): Replica count hero chart — HPA vs RL Agent step-lines
    with scale-up / scale-down markers and a filled area under the RL line.

    Panel B (row 2, 30%): CPU utilisation + optional MC-Dropout forecast band
    + SLO threshold line.

    Panel C (row 3, 20%): Diverging bar chart of Δ replicas per timestep.

    Parameters
    ----------
    hpa_replicas   : 1D array of HPA replica counts.
    rl_replicas    : 1D array of RL agent replica counts.
    cpu_series     : 1D array of CPU utilisation values (0–1).
    mean_fc        : Optional forecast mean array (same length as cpu_series).
    lower_fc       : Optional forecast CI lower bound.
    upper_fc       : Optional forecast CI upper bound.
    slo_threshold  : SLO target utilisation (drawn as dashed hline on Panel B).
    is_simulated   : If True, adds a "Simulated" annotation on Panel A.
    theme          : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    n = len(rl_replicas)
    steps = np.arange(n)

    diffs = np.diff(rl_replicas, prepend=rl_replicas[0])
    up_idx   = np.where(diffs > 0)[0]
    down_idx = np.where(diffs < 0)[0]
    hold_idx = np.where(diffs == 0)[0]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.30, 0.20],
        vertical_spacing=0.04,
        subplot_titles=["Pod Replicas: HPA vs RL Agent", "CPU Utilisation & Forecast", "Scaling Events (Δ Replicas)"],
    )

    # ── Panel A: replica counts ───────────────────────────────────────────────

    # HPA dashed line
    fig.add_trace(go.Scatter(
        x=steps, y=hpa_replicas,
        mode="lines",
        name="HPA (reactive)",
        line=dict(color=p["danger"], width=2, shape="hv", dash="dash"),
        hovertemplate="Step %{x}<br>HPA Replicas: %{y}<extra>HPA</extra>",
    ), row=1, col=1)

    # RL filled area
    fig.add_trace(go.Scatter(
        x=np.concatenate([steps, steps[::-1]]),
        y=np.concatenate([rl_replicas, np.zeros(n)]),
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(p['accent'])},0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="RL fill",
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # RL solid line
    fig.add_trace(go.Scatter(
        x=steps, y=rl_replicas,
        mode="lines",
        name="RL Agent (predictive)",
        line=dict(color=p["accent"], width=2.5, shape="hv"),
        hovertemplate="Step %{x}<br>RL Replicas: %{y}<extra>RL Agent</extra>",
    ), row=1, col=1)

    # Scale-up markers
    if len(up_idx) > 0:
        fig.add_trace(go.Scatter(
            x=up_idx, y=rl_replicas[up_idx],
            mode="markers+text",
            name="Scale Up",
            marker=dict(symbol="triangle-up", size=11, color=p["success"]),
            text=[f"+{int(d)}" for d in diffs[up_idx]],
            textposition="top center",
            textfont=dict(size=9, color=p["success"]),
            hovertemplate="Step %{x}<br>Scale UP to %{y}<extra>Scale Up</extra>",
        ), row=1, col=1)

    # Scale-down markers
    if len(down_idx) > 0:
        fig.add_trace(go.Scatter(
            x=down_idx, y=rl_replicas[down_idx],
            mode="markers+text",
            name="Scale Down",
            marker=dict(symbol="triangle-down", size=11, color=p["warning"]),
            text=[f"{int(d)}" for d in diffs[down_idx]],
            textposition="bottom center",
            textfont=dict(size=9, color=p["warning"]),
            hovertemplate="Step %{x}<br>Scale DOWN to %{y}<extra>Scale Down</extra>",
        ), row=1, col=1)

    # Hold markers (subtle dots)
    if len(hold_idx) > 0:
        fig.add_trace(go.Scatter(
            x=hold_idx, y=rl_replicas[hold_idx],
            mode="markers",
            name="Hold",
            marker=dict(symbol="circle", size=3, color=p["text_secondary"], opacity=0.35),
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)

    # Vertical shaded bands where RL scaled before HPA reacted
    hpa_diffs = np.diff(hpa_replicas, prepend=hpa_replicas[0])
    proactive_idx = np.where((diffs > 0) & (hpa_diffs == 0))[0]
    for idx in proactive_idx:
        fig.add_vrect(
            x0=max(0, idx - 0.5), x1=min(n - 1, idx + 1.5),
            fillcolor=p["success"], opacity=0.07,
            layer="below", line_width=0,
            row=1, col=1,
        )

    # Dynamic y-axis range
    y_max = max(int(hpa_replicas.max()), int(rl_replicas.max())) + 2
    fig.update_yaxes(range=[0, y_max], title_text="Replicas", row=1, col=1)

    if is_simulated:
        fig.add_annotation(
            text="Simulated data",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=10, color=p["warning"]),
            bgcolor=f"rgba({_hex_to_rgb(p['warning'])},0.12)",
            bordercolor=p["warning"],
            borderwidth=1,
            borderpad=4,
        )

    # ── Panel B: CPU + forecast ───────────────────────────────────────────────
    cpu_n = min(len(cpu_series), n)

    # MC-Dropout CI band
    if mean_fc is not None and lower_fc is not None and upper_fc is not None:
        x_band = list(steps[:cpu_n]) + list(steps[:cpu_n][::-1])
        y_band = list(upper_fc[:cpu_n]) + list(lower_fc[:cpu_n][::-1])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(p['warning'])},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="MC-Dropout CI",
            hoverinfo="skip",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=steps[:cpu_n], y=mean_fc[:cpu_n],
            mode="lines",
            name="Forecast mean",
            line=dict(color=p["warning"], width=1.5, dash="dot"),
            hovertemplate="Step %{x}<br>Forecast: %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    # CPU utilisation
    fig.add_trace(go.Scatter(
        x=steps[:cpu_n], y=cpu_series[:cpu_n],
        mode="lines",
        name="CPU Util",
        line=dict(color=p["accent"], width=1.8),
        hovertemplate="Step %{x}<br>CPU: %{y:.3f}<extra>CPU</extra>",
    ), row=2, col=1)

    # SLO threshold
    fig.add_hline(
        y=slo_threshold,
        line_dash="dot", line_color=p["danger"], line_width=1.2,
        annotation_text=f"SLO {slo_threshold}",
        annotation_position="top right",
        annotation_font=dict(size=9, color=p["danger"]),
        row=2, col=1,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="CPU", row=2, col=1)

    # ── Panel C: diverging delta bars ─────────────────────────────────────────
    colors = [p["success"] if d > 0 else p["danger"] for d in diffs]
    fig.add_trace(go.Bar(
        x=steps, y=diffs,
        marker_color=colors,
        name="Δ Replicas",
        hovertemplate="Step %{x}<br>Δ Replicas: %{y:+d}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(title_text="Δ Pods", row=3, col=1)
    fig.update_xaxes(title_text="Timestep", row=3, col=1)

    # ── Global layout ─────────────────────────────────────────────────────────
    tpl = get_plotly_template(theme)
    fig.update_layout(
        plot_bgcolor=p["surface"],
        paper_bgcolor=p["surface"],
        font=tpl["font"],
        height=640,
        margin=dict(l=55, r=20, t=60, b=45),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=p["text_secondary"], size=11),
            orientation="h",
            y=1.04, x=0, xanchor="left",
        ),
        colorway=p["colorway"],
        showlegend=True,
    )
    # Apply grid colors to all axes
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["border"], row=i, col=1)
        fig.update_yaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["border"], row=i, col=1)

    # Style subplot title text
    for ann in fig.layout.annotations:
        ann.font.color = p["text_secondary"]
        ann.font.size = 11
        ann.font.family = "Inter, system-ui, sans-serif"

    return fig


# ── Workload trace figure ─────────────────────────────────────────────────────

def make_workload_trace_figure(
    df_sub: pd.DataFrame,
    cur_pos_index: object,
    theme: str = "dark",
) -> go.Figure:
    """CPU utilisation trace with a 'now' vline for the Forecasting tab.

    Parameters
    ----------
    df_sub        : Slice of the full DataFrame containing a ``cpu_util`` column.
    cur_pos_index : Index value (label) used for the vertical 'now' annotation.
    theme         : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sub.index, y=df_sub["cpu_util"],
        mode="lines",
        name="CPU Util",
        line=dict(color=p["accent"], width=1.8),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(p['accent'])},0.10)",
    ))
    if len(df_sub) > 0:
        fig.add_vline(
            x=cur_pos_index, line_width=1.5,
            line_dash="dash", line_color=p["warning"],
            annotation_text="now",
            annotation_position="top right",
            annotation_font=dict(size=9, color=p["warning"]),
        )
    fig.update_layout(**_get_layout("Workload Trace (replaying)", 280, theme))
    fig.update_yaxes(range=[0, 1.05], title_text="CPU Utilisation")
    return fig


# ── MC-Dropout forecast figure ────────────────────────────────────────────────

def make_forecast_figure(
    actual: np.ndarray,
    mean_fc: np.ndarray,
    lower_fc: np.ndarray,
    upper_fc: np.ndarray,
    horizon: int,
    ci_lower: int,
    ci_upper: int,
    theme: str = "dark",
) -> go.Figure:
    """MC-Dropout forecast figure for the Forecasting tab.

    Parameters
    ----------
    actual   : Ground-truth values (length = horizon).
    mean_fc  : Forecast mean (length = horizon).
    lower_fc : CI lower bound.
    upper_fc : CI upper bound.
    horizon  : Forecast horizon (number of steps ahead).
    ci_lower : Lower percentile used (for label).
    ci_upper : Upper percentile used (for label).
    theme    : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    x = list(range(1, horizon + 1))

    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(upper_fc) + list(lower_fc[::-1]),
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(p['warning'])},0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"CI band ({ci_lower}–{ci_upper}th)",
        hoverinfo="skip",
    ))
    # Actual
    fig.add_trace(go.Scatter(
        x=x, y=list(actual),
        mode="lines+markers",
        name="Actual",
        line=dict(color=p["accent"], width=2),
        marker=dict(size=5),
    ))
    # Forecast mean
    fig.add_trace(go.Scatter(
        x=x, y=list(mean_fc),
        mode="lines+markers",
        name="Forecast (mean)",
        line=dict(color=p["warning"], width=2, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
    ))

    fig.update_layout(**_get_layout(f"MC Dropout Forecast — next {horizon} steps", 340, theme))
    fig.update_xaxes(title_text="Horizon step")
    fig.update_yaxes(title_text="CPU Utilisation")
    return fig


# ── Rolling MAE figure ────────────────────────────────────────────────────────

def make_rolling_mae_figure(mae_series: list[float], theme: str = "dark") -> go.Figure:
    """Rolling-20 MAE chart for the Forecasting tab.

    Parameters
    ----------
    mae_series : List of MAE values accumulated over replay steps.
    theme      : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    rolling = pd.Series(mae_series).rolling(20, min_periods=1).mean().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rolling,
        mode="lines",
        name="MAE (rolling-20)",
        line=dict(color=p["colorway"][4], width=1.5),
    ))
    fig.update_layout(**_get_layout("Rolling MAE over Replay", 200, theme))
    fig.update_xaxes(title_text="Replay step")
    fig.update_yaxes(title_text="MAE")
    return fig


# ── Replica trace figure (RL tab) ─────────────────────────────────────────────

def make_replica_trace_figure(
    replicas: list[int],
    replay_s: int,
    theme: str = "dark",
) -> go.Figure:
    """Replica count over RL episode with replay vline.

    Parameters
    ----------
    replicas : List of replica counts per episode step.
    replay_s : Current replay position (for vline).
    theme    : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    n = len(replicas)
    steps = np.arange(n)
    ds = max(1, n // 2000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps[::ds], y=np.array(replicas)[::ds],
        mode="lines",
        name="Replicas",
        line=dict(color=p["accent"], width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(p['accent'])},0.08)",
    ))
    if replay_s > 0:
        fig.add_vline(
            x=replay_s, line_dash="dash",
            line_color=p["warning"], line_width=1.5,
        )
    fig.update_layout(**_get_layout("Replica Count over Episode", 280, theme))
    fig.update_yaxes(title_text="Replicas")
    return fig


# ── CPU-per-pod figure (RL tab) ────────────────────────────────────────────────

def make_cpu_per_pod_figure(
    cpu_per_pod: np.ndarray,
    slo_thr: float,
    replay_s: int,
    theme: str = "dark",
) -> go.Figure:
    """CPU-per-pod trace with SLO threshold and replay vline.

    Parameters
    ----------
    cpu_per_pod : Array of CPU-per-pod values.
    slo_thr     : SLO threshold for dashed hline.
    replay_s    : Current replay position.
    theme       : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    n = len(cpu_per_pod)
    steps = np.arange(n)
    ds = max(1, n // 2000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps[::ds], y=cpu_per_pod[::ds],
        mode="lines",
        name="CPU / Pod",
        line=dict(color=p["success"], width=1.2),
    ))
    fig.add_hline(
        y=slo_thr, line_dash="dash",
        line_color=p["danger"], line_width=1.5,
        annotation_text=f"SLO={slo_thr}",
        annotation_position="top right",
        annotation_font=dict(size=9, color=p["danger"]),
    )
    if replay_s > 0:
        fig.add_vline(
            x=replay_s, line_dash="dash",
            line_color=p["warning"], line_width=1.5,
        )
    fig.update_layout(**_get_layout("CPU-per-Pod (SLO threshold)", 280, theme))
    fig.update_yaxes(
        title_text="CPU / Pod",
        range=[0, max(1.2, float(cpu_per_pod.max()) * 1.1)],
    )
    return fig


# ── Action distribution figure (RL tab) ──────────────────────────────────────

def make_action_distribution_figure(
    actions: list[int],
    theme: str = "dark",
) -> go.Figure:
    """Bar chart of action distribution for the RL Episode Replay tab.

    Parameters
    ----------
    actions : List of integer actions (0–4 mapping to -2, -1, 0, +1, +2).
    theme   : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    action_labels = ["-2", "-1", "0", "+1", "+2"]
    counts = np.bincount(actions, minlength=5)
    pcts = counts / max(counts.sum(), 1) * 100

    bar_colors = [p["danger"], p["warning"], p["accent"], p["success"], p["colorway"][4]]

    fig = go.Figure(go.Bar(
        x=action_labels, y=pcts,
        marker_color=bar_colors,
        text=[f"{v:.0f}%" for v in pcts],
        textposition="outside",
    ))
    fig.update_layout(**_get_layout("Action Distribution", 280, theme))
    fig.update_xaxes(title_text="Δ Replicas")
    fig.update_yaxes(title_text="Usage (%)")
    return fig


# ── Cumulative reward figure (RL tab) ─────────────────────────────────────────

def make_cumulative_reward_figure(
    rewards: list[float],
    replay_s: int,
    theme: str = "dark",
) -> go.Figure:
    """Cumulative reward curve for the RL Episode Replay tab.

    Parameters
    ----------
    rewards  : List of per-step reward values.
    replay_s : Current replay position for vline.
    theme    : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    n = len(rewards)
    steps = np.arange(n)
    ds = max(1, n // 2000)
    cum_rew = np.cumsum(rewards)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps[::ds], y=cum_rew[::ds],
        mode="lines",
        name="Cumulative Reward",
        line=dict(color=p["warning"], width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(p['warning'])},0.08)",
    ))
    if replay_s > 0:
        fig.add_vline(
            x=replay_s, line_dash="dash",
            line_color=p["warning"], line_width=1.5,
        )
    fig.update_layout(**_get_layout("Cumulative Reward", 280, theme))
    fig.update_yaxes(title_text="Cumulative Reward")
    return fig


# ── Training reward convergence figure ────────────────────────────────────────

def make_training_reward_figure(
    df_log: pd.DataFrame,
    theme: str = "dark",
) -> go.Figure:
    """PPO training reward convergence chart from training_log.csv.

    Parameters
    ----------
    df_log : DataFrame with columns ``rollout`` and ``mean_reward``.
    theme  : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_log["rollout"], y=df_log["mean_reward"],
        mode="lines",
        name="Mean Reward (rolling-10 episodes)",
        line=dict(color=p["colorway"][4], width=2.5),
    ))
    fig.update_layout(**_get_layout("PPO Training Reward Convergence", 320, theme))
    fig.update_xaxes(title_text="Rollout")
    fig.update_yaxes(title_text="Mean Episode Reward (last 10 eps)")
    return fig


# ── Loss curves figure ────────────────────────────────────────────────────────

def make_loss_curves_figure(
    df_log: pd.DataFrame,
    theme: str = "dark",
) -> go.Figure:
    """Policy + value loss curves from training_log.csv.

    Parameters
    ----------
    df_log : DataFrame with columns ``rollout``, ``policy_loss``, ``value_loss``.
    theme  : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_log["rollout"], y=df_log["policy_loss"],
        mode="lines",
        name="Policy Loss",
        line=dict(color=p["danger"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df_log["rollout"], y=df_log["value_loss"],
        mode="lines",
        name="Value Loss",
        line=dict(color=p["accent"], width=1.5),
    ))
    fig.update_layout(**_get_layout("Training Loss Curves", 260, theme))
    fig.update_xaxes(title_text="Rollout")
    fig.update_yaxes(title_text="Loss")
    return fig


# ── Training history figure (from checkpoint) ─────────────────────────────────

def make_training_history_figure(
    ep_rew: list[float],
    w_rm: int,
    theme: str = "dark",
) -> go.Figure:
    """PPO checkpoint reward history with rolling mean smoothing.

    Parameters
    ----------
    ep_rew : List of episode rewards from the checkpoint history.
    w_rm   : Rolling-mean window size.
    theme  : ``"dark"`` or ``"light"``.

    Returns
    -------
    go.Figure
    """
    p = get_palette(theme)
    x_real = np.arange(1, len(ep_rew) + 1)
    rm = pd.Series(ep_rew).rolling(w_rm, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_real, y=ep_rew,
        mode="lines",
        name="Checkpoint Episode Reward",
        line=dict(color=p["accent"], width=0.8),
        opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=x_real, y=rm.tolist(),
        mode="lines",
        name=f"Rolling Mean ({w_rm} ep)",
        line=dict(color=p["warning"], width=2.0),
    ))
    fig.update_layout(**_get_layout("Trained PPO Checkpoint Progress", 260, theme))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Total Reward")
    return fig


# ── Utility ────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convert a ``#rrggbb`` hex string to a ``r,g,b`` string for rgba() CSS.

    Parameters
    ----------
    hex_color : Hex color string, with or without leading ``#``.

    Returns
    -------
    str
        Comma-separated RGB values, e.g. ``"88,166,255"``.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
