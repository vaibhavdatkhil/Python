"""
dashboard/ui/components.py
───────────────────────────
Reusable HTML/Streamlit component builders for the K8s RL Autoscaling Dashboard.

All color references come from get_palette() — no hardcoded hex values here.
"""

from __future__ import annotations

import datetime

import streamlit as st
import streamlit.components.v1 as components

from .theme import get_palette


# ── KPI / Metric card ─────────────────────────────────────────────────────────

def metric_card(
    label: str,
    value: str,
    delta: str = "",
    icon: str = "",
    status: str = "neutral",
    theme: str = "dark",
) -> str:
    """Return an HTML string for a glassmorphism KPI card.

    Parameters
    ----------
    label  : Card label (displayed in small caps).
    value  : Primary metric value (large, bold).
    delta  : Secondary / delta line shown below the value.
    icon   : Emoji icon shown at top-right of the card.
    status : ``"ok"`` | ``"warn"`` | ``"err"`` | ``"accent"`` | ``"neutral"``
             Controls the left-border accent color.
    theme  : ``"dark"`` or ``"light"``.

    Returns
    -------
    str
        HTML string — pass to ``st.markdown(..., unsafe_allow_html=True)``.
    """
    status_class = {
        "ok":     "status-ok",
        "warn":   "status-warn",
        "err":    "status-err",
        "accent": "status-accent",
    }.get(status, "")

    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""

    return (
        f'<div class="kpi-card {status_class}">'
        f"{icon_html}"
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}"
        f"</div>"
    )


def render_metric_row(cards: list[dict], theme: str = "dark") -> None:
    """Render a horizontal row of metric_card dicts using st.columns.

    Parameters
    ----------
    cards : list of dicts with keys: ``label``, ``value``, and optional
            ``delta``, ``icon``, ``status``.
    theme : ``"dark"`` or ``"light"``.
    """
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                metric_card(
                    label=card.get("label", ""),
                    value=card.get("value", ""),
                    delta=card.get("delta", ""),
                    icon=card.get("icon", ""),
                    status=card.get("status", "neutral"),
                    theme=theme,
                ),
                unsafe_allow_html=True,
            )


# ── Badge ─────────────────────────────────────────────────────────────────────

def badge(text: str, level: str = "ok") -> str:
    """Return a colored pill badge HTML string.

    Parameters
    ----------
    text  : Badge label text.
    level : ``"ok"`` | ``"warn"`` | ``"err"``.

    Returns
    -------
    str
        HTML ``<span>`` element.
    """
    return f'<span class="badge-{level}">{text}</span>'


# ── Render header ──────────────────────────────────────────────────────────────

def render_header(api_status: str = "unknown", theme: str = "dark") -> None:
    """Render the gradient hero banner with title, subtitle, and status pill.

    The dark/light toggle is rendered separately in the sidebar by app.py
    (Streamlit widgets cannot be embedded in raw HTML).

    Parameters
    ----------
    api_status : ``"healthy"`` | ``"offline"`` | ``"unknown"``.
    theme      : ``"dark"`` or ``"light"``.
    """
    if api_status == "healthy":
        pill_class = "status-pill"
        pill_text = '<span class="live-dot">●</span> API Healthy'
    elif api_status == "offline":
        pill_class = "status-pill-offline"
        pill_text = "● API Offline"
    else:
        pill_class = "status-pill-unknown"
        pill_text = "● API Unknown"

    st.markdown(
        f"""
        <div class="hero-banner">
          <div class="hero-left">
            <div class="hero-title">🚀 K8s RL Autoscaling Dashboard</div>
            <div class="hero-sub">
              Intelligent Kubernetes pod autoscaling using
              <strong>LSTM MC-Dropout forecasting</strong> and
              <strong>PPO reinforcement learning</strong>.
            </div>
          </div>
          <div class="hero-right">
            <span class="{pill_class}">{pill_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Simulated data badge ───────────────────────────────────────────────────────

def render_simulated_badge() -> None:
    """Render a small amber 'Simulated data' badge."""
    st.markdown(
        '<span class="badge-sim">🔵 Simulated data</span>',
        unsafe_allow_html=True,
    )


# ── Section header ─────────────────────────────────────────────────────────────

def render_section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section title with a subtle divider line.

    Parameters
    ----------
    title    : Section heading text.
    subtitle : Optional sub-label shown beneath the title.
    """
    sub_html = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
          <p class="section-title">{title}</p>
          {sub_html}
        </div>
        <hr class="section-divider" />
        """,
        unsafe_allow_html=True,
    )


# ── Pod grid ───────────────────────────────────────────────────────────────────

def render_pod_grid(
    current_rl: int,
    previous_rl: int,
    current_hpa: int,
    previous_hpa: int,
    theme: str = "dark",
    height: int = 170,
) -> None:
    """Render an animated pod grid showing RL Agent and HPA replicas.

    Uses ``st.components.v1.html`` for iframe isolation so CSS keyframe
    animations are not disrupted by Streamlit reruns.

    Newly added pods pulse green (``podIn`` + ``podPulse``), removed pods
    fade red (``podOut``). Max 12 pods displayed per row.

    Parameters
    ----------
    current_rl   : Current RL agent replica count.
    previous_rl  : RL replica count on the previous step (for diff detection).
    current_hpa  : Current HPA replica count.
    previous_hpa : HPA replica count on the previous step.
    theme        : ``"dark"`` or ``"light"``.
    height       : iframe height in pixels.
    """
    p = get_palette(theme)
    MAX_PODS = 12

    def _row_html(current: int, previous: int, is_hpa: bool = False) -> str:
        pods_html = ""
        n = min(current, MAX_PODS)
        prev = min(previous, MAX_PODS)
        for i in range(n):
            if is_hpa:
                cls = "pod-hpa-new" if i >= prev else "pod-hpa"
            else:
                cls = "pod-new" if i >= prev else "pod-icon"
            label = "P" if not is_hpa else "H"
            pods_html += f'<div class="{cls}" title="Pod {i+1}">{label}</div>'
        # Show fading pods that were removed
        if previous > current:
            fade_count = min(previous - current, MAX_PODS - n)
            for _ in range(fade_count):
                cls = "pod-hpa-removed" if is_hpa else "pod-removed"
                pods_html += f'<div class="{cls}"></div>'
        return pods_html

    rl_pods = _row_html(current_rl, previous_rl, is_hpa=False)
    hpa_pods = _row_html(current_hpa, previous_hpa, is_hpa=True)

    html = f"""
    <html><head>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
      body {{
        margin: 0; padding: 8px 14px;
        font-family: 'Inter', sans-serif;
        background: {p['surface']};
        color: {p['text_primary']};
      }}
      .pod-grid-wrapper {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 10px;
        padding: 10px 14px;
      }}
      .row-label {{
        font-size: 11px; font-weight: 600;
        color: {p['text_secondary']};
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 5px;
      }}
      .pod-row {{
        display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;
      }}
      .pod-icon, .pod-new, .pod-removed, .pod-hpa, .pod-hpa-new, .pod-hpa-removed {{
        width: 24px; height: 24px; border-radius: 5px;
        display: flex; align-items: center; justify-content: center;
        font-size: 10px; font-weight: 700; color: {p['bg']};
      }}
      .pod-icon   {{ background: {p['accent']}; opacity: 0.75; }}
      .pod-new    {{ background: {p['success']};
                    animation: podIn 0.55s ease-out forwards, podPulse 1.2s ease-in-out 0.55s 2; }}
      .pod-removed{{ background: {p['danger']};
                    animation: podOut 0.6s ease-in forwards; }}
      .pod-hpa         {{ background: {p['danger']}; opacity: 0.6; }}
      .pod-hpa-new     {{ background: {p['danger']};
                          animation: podIn 0.55s ease-out forwards; }}
      .pod-hpa-removed {{ background: {p['danger']}; opacity: 0.35;
                          animation: podOut 0.6s ease-in forwards; }}
      @keyframes podIn {{
        from {{ opacity: 0; transform: scale(0.25); }}
        to   {{ opacity: 0.9; transform: scale(1); }}
      }}
      @keyframes podOut {{
        from {{ opacity: 0.85; transform: scale(1); }}
        to   {{ opacity: 0; transform: scale(0.25); }}
      }}
      @keyframes podPulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 {p['success']}80; }}
        50%       {{ box-shadow: 0 0 0 5px {p['success']}00; }}
      }}
    </style></head>
    <body>
      <div class="pod-grid-wrapper">
        <div class="row-label">🤖 RL Agent — {current_rl} pod{'s' if current_rl != 1 else ''}</div>
        <div class="pod-row">{rl_pods}</div>
        <div class="row-label">⚙️ HPA — {current_hpa} pod{'s' if current_hpa != 1 else ''}</div>
        <div class="pod-row">{hpa_pods}</div>
      </div>
    </body></html>
    """
    components.html(html, height=height, scrolling=False)


# ── Event feed ─────────────────────────────────────────────────────────────────

def render_event_feed(events: list[dict], theme: str = "dark") -> None:
    """Render the 'Latest Scaling Decisions' event feed.

    Parameters
    ----------
    events : List of event dicts with keys: ``timestamp``, ``action``
             (``"UP"`` | ``"DOWN"`` | ``"HOLD"``), ``from_r``, ``to_r``, ``reason``.
             Displays up to 5 most-recent events.
    theme  : ``"dark"`` or ``"light"``.
    """
    if not events:
        return

    rows_html = ""
    for ev in events[:5]:
        action = ev.get("action", "HOLD")
        badge_cls = "badge-up" if action == "UP" else ("badge-down" if action == "DOWN" else "badge-hold")
        action_icon = "⬆" if action == "UP" else ("⬇" if action == "DOWN" else "—")
        rows_html += (
            f'<div class="event-feed-row">'
            f'<span class="event-ts">{ev.get("timestamp", "")}</span>'
            f'<span class="{badge_cls}">{action_icon} {action}</span>'
            f'<span class="event-delta">{ev.get("from_r", "?")} → {ev.get("to_r", "?")} replicas</span>'
            f'<span class="event-reason">{ev.get("reason", "")}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="event-feed">
          <div class="event-feed-header">⚡ Latest Scaling Decisions</div>
          {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Progress bar ──────────────────────────────────────────────────────────────

def render_progress_bar(current: int, total: int, theme: str = "dark") -> None:
    """Render a slim HTML progress bar showing replay step / total.

    Parameters
    ----------
    current : Current step (1-based).
    total   : Total number of steps.
    theme   : ``"dark"`` or ``"light"``.
    """
    pct = int(current / max(total, 1) * 100)
    st.markdown(
        f"""
        <div class="progress-outer">
          <div class="progress-inner" style="width:{pct}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── How-to-read expander ──────────────────────────────────────────────────────

def render_how_to_read(text: str) -> None:
    """Render an expandable 'How to read this chart' helper.

    Parameters
    ----------
    text : Markdown-formatted explanation text.
    """
    with st.expander("ℹ️ How to read this chart"):
        st.markdown(text)


# ── Footer ────────────────────────────────────────────────────────────────────

def render_footer(version: str = "v1.0.0", theme: str = "dark") -> None:
    """Render a simple app footer with version and last-updated timestamp.

    Parameters
    ----------
    version : Application version string (e.g. ``"v1.0.0"``).
    theme   : ``"dark"`` or ``"light"``.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(
        f"""
        <div class="app-footer">
          K8s RL Autoscaling Dashboard &nbsp;·&nbsp; {version}
          &nbsp;·&nbsp; Last updated: {now}
          &nbsp;·&nbsp; B.Tech Final Year Project
        </div>
        """,
        unsafe_allow_html=True,
    )
