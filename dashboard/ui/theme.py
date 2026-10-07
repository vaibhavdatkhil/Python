"""
dashboard/ui/theme.py
─────────────────────
Centralized theme module: color palettes, Plotly templates, and CSS injection.

All color constants live here. No hardcoded hex values in any other module.
"""

from __future__ import annotations

import streamlit as st

# ── Palette definitions ───────────────────────────────────────────────────────

DARK: dict = {
    "bg":             "#0a0e1a",
    "surface":        "#111827",
    "surface2":       "#1a2235",
    "border":         "#1e2d45",
    "border2":        "#2d3f5c",
    "text_primary":   "#e2e8f4",
    "text_secondary": "#7a8ba8",
    "accent":         "#4f9cf9",
    "accent2":        "#3b82f6",
    "success":        "#22c55e",
    "warning":        "#f59e0b",
    "danger":         "#ef4444",
    "grid":           "#1a2235",
    "colorway": [
        "#4f9cf9",
        "#22c55e",
        "#f59e0b",
        "#ef4444",
        "#a78bfa",
        "#06b6d4",
        "#f43f5e",
        "#34d399",
    ],
    # Extra tokens used only in CSS
    "shadow":           "rgba(0,0,0,0.55)",
    "shadow_card":      "rgba(0,0,0,0.35)",
    "hero_grad_a":      "#0a0e1a",
    "hero_grad_b":      "#0f1e3d",
    "hero_grad_c":      "#091428",
    "accent_glow":      "rgba(79,156,249,0.12)",
    "accent_glow2":     "rgba(79,156,249,0.06)",
    "glass_bg":         "rgba(17,24,39,0.80)",
    "glass_border":     "rgba(78,156,249,0.15)",
    "sidebar_bg":       "#080c16",
    "sidebar_border":   "#1e2d45",
    "success_bg":       "rgba(34,197,94,0.10)",
    "warning_bg":       "rgba(245,158,11,0.10)",
    "danger_bg":        "rgba(239,68,68,0.10)",
    "accent_bg":        "rgba(79,156,249,0.10)",
}

LIGHT: dict = {
    "bg":             "#f8faff",
    "surface":        "#ffffff",
    "surface2":       "#f1f5f9",
    "border":         "#e2e8f0",
    "border2":        "#cbd5e1",
    "text_primary":   "#0f172a",
    "text_secondary": "#64748b",
    "accent":         "#2563eb",
    "accent2":        "#3b82f6",
    "success":        "#16a34a",
    "warning":        "#d97706",
    "danger":         "#dc2626",
    "grid":           "#e8eef7",
    "colorway": [
        "#2563eb",
        "#16a34a",
        "#d97706",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
        "#e11d48",
        "#059669",
    ],
    # Extra tokens used only in CSS
    "shadow":           "rgba(15,23,42,0.10)",
    "shadow_card":      "rgba(15,23,42,0.07)",
    "hero_grad_a":      "#eff6ff",
    "hero_grad_b":      "#dbeafe",
    "hero_grad_c":      "#f0f7ff",
    "accent_glow":      "rgba(37,99,235,0.08)",
    "accent_glow2":     "rgba(37,99,235,0.04)",
    "glass_bg":         "rgba(255,255,255,0.92)",
    "glass_border":     "rgba(37,99,235,0.15)",
    "sidebar_bg":       "#f1f5f9",
    "sidebar_border":   "#e2e8f0",
    "success_bg":       "rgba(22,163,74,0.08)",
    "warning_bg":       "rgba(217,119,6,0.08)",
    "danger_bg":        "rgba(220,38,38,0.08)",
    "accent_bg":        "rgba(37,99,235,0.07)",
}


def get_palette(theme: str = "dark") -> dict:
    """Return the color palette dict for the given theme.

    Parameters
    ----------
    theme : str
        ``"dark"`` or ``"light"``.

    Returns
    -------
    dict
        Palette mapping token names to hex color strings.
    """
    return DARK if theme == "dark" else LIGHT


def get_plotly_template(theme: str = "dark") -> dict:
    """Return a Plotly layout dict matching the active theme palette.

    Intended for use as ``fig.update_layout(**get_plotly_template(theme))``.

    Parameters
    ----------
    theme : str
        ``"dark"`` or ``"light"``.

    Returns
    -------
    dict
        Plotly layout keyword arguments.
    """
    p = get_palette(theme)
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=p["text_secondary"], size=12),
        title_font=dict(family="Inter, system-ui, sans-serif", color=p["text_primary"], size=14),
        xaxis=dict(
            gridcolor=p["grid"],
            showgrid=True,
            zeroline=False,
            linecolor=p["border"],
            tickfont=dict(color=p["text_secondary"], size=11),
            title_font=dict(color=p["text_secondary"], size=11),
        ),
        yaxis=dict(
            gridcolor=p["grid"],
            showgrid=True,
            zeroline=False,
            linecolor=p["border"],
            tickfont=dict(color=p["text_secondary"], size=11),
            title_font=dict(color=p["text_secondary"], size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=p["text_secondary"], size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=52, r=20, t=50, b=48),
        colorway=p["colorway"],
    )


def inject_css(theme: str = "dark") -> str:
    """Build the full CSS block for the given theme and inject via st.markdown.

    Returns the ``<style>`` block string (also calls ``st.markdown`` for
    immediate injection).

    Parameters
    ----------
    theme : str
        ``"dark"`` or ``"light"``.

    Returns
    -------
    str
        The ``<style>...</style>`` HTML block.
    """
    p = get_palette(theme)
    css = f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800&display=swap');

/* ── CSS variables ── */
:root {{
  --bg:              {p['bg']};
  --surface:         {p['surface']};
  --surface2:        {p['surface2']};
  --border:          {p['border']};
  --border2:         {p['border2']};
  --text-primary:    {p['text_primary']};
  --text-secondary:  {p['text_secondary']};
  --accent:          {p['accent']};
  --accent2:         {p['accent2']};
  --success:         {p['success']};
  --warning:         {p['warning']};
  --danger:          {p['danger']};
  --grid:            {p['grid']};
  --shadow:          {p['shadow']};
  --shadow-card:     {p['shadow_card']};
  --hero-grad-a:     {p['hero_grad_a']};
  --hero-grad-b:     {p['hero_grad_b']};
  --hero-grad-c:     {p['hero_grad_c']};
  --accent-glow:     {p['accent_glow']};
  --glass-border:    {p['glass_border']};
  --sidebar-bg:      {p['sidebar_bg']};
  --sidebar-border:  {p['sidebar_border']};
  --success-bg:      {p['success_bg']};
  --warning-bg:      {p['warning_bg']};
  --danger-bg:       {p['danger_bg']};
  --accent-bg:       {p['accent_bg']};
}}

/* ── Base typography ── */
html, body, [class*="css"] {{
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
}}

/* ── Main app background ── */
.stApp {{
  background: var(--bg) !important;
}}
.main .block-container {{
  background: var(--bg) !important;
  padding-top: 0.75rem !important;
  max-width: 1400px;
}}

/* ── KPI / metric cards (glassmorphism) ── */
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.3rem 1rem 1.3rem;
  margin-bottom: 0.6rem;
  transition: transform 0.20s cubic-bezier(.25,.46,.45,.94),
              box-shadow 0.20s cubic-bezier(.25,.46,.45,.94),
              border-color 0.20s ease;
  position: relative;
  overflow: hidden;
  box-shadow: 0 2px 8px var(--shadow-card);
  min-height: 96px;
}}
.kpi-card::after {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent) 0%, transparent 70%);
  opacity: 0;
  transition: opacity 0.20s ease;
}}
.kpi-card:hover {{
  transform: translateY(-3px);
  box-shadow: 0 10px 36px var(--shadow);
  border-color: var(--border2);
}}
.kpi-card:hover::after {{
  opacity: 1;
}}
.kpi-card.status-ok   {{
  border-left: 3px solid var(--success);
  background: linear-gradient(135deg, var(--surface) 0%, var(--success-bg) 100%);
}}
.kpi-card.status-warn {{
  border-left: 3px solid var(--warning);
  background: linear-gradient(135deg, var(--surface) 0%, var(--warning-bg) 100%);
}}
.kpi-card.status-err  {{
  border-left: 3px solid var(--danger);
  background: linear-gradient(135deg, var(--surface) 0%, var(--danger-bg)  100%);
}}
.kpi-card.status-accent {{
  border-left: 3px solid var(--accent);
  background: linear-gradient(135deg, var(--surface) 0%, var(--accent-bg) 100%);
}}
.kpi-icon  {{
  position: absolute; top: 0.8rem; right: 0.9rem;
  font-size: 1.4rem; opacity: 0.40;
  transition: opacity 0.20s ease, transform 0.20s ease;
}}
.kpi-card:hover .kpi-icon {{ opacity: 0.72; transform: scale(1.12) rotate(-5deg); }}
.kpi-label {{
  font-size: 0.67rem; color: var(--text-secondary);
  letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;
  margin-bottom: 0.2rem;
}}
.kpi-value {{
  font-size: 1.75rem; font-weight: 800; color: var(--text-primary);
  line-height: 1.15; margin: 0.1rem 0;
  letter-spacing: -0.01em;
}}
.kpi-delta {{
  font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.2rem;
  line-height: 1.4;
}}
/* Legacy sub-label used by kpi() helper — matches .kpi-delta styling */
.kpi-sub {{
  font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.2rem;
  line-height: 1.4;
}}

/* ── Badges ── */
.badge-ok {{
  display:inline-flex; align-items:center; gap:4px;
  background: var(--success-bg);
  color: var(--success); border: 1px solid color-mix(in srgb, var(--success) 30%, transparent);
  border-radius: 100px; padding: 2px 10px; font-size: 0.74rem; font-weight: 600;
}}
.badge-warn {{
  display:inline-flex; align-items:center; gap:4px;
  background: var(--warning-bg);
  color: var(--warning); border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent);
  border-radius: 100px; padding: 2px 10px; font-size: 0.74rem; font-weight: 600;
}}
.badge-err {{
  display:inline-flex; align-items:center; gap:4px;
  background: var(--danger-bg);
  color: var(--danger); border: 1px solid color-mix(in srgb, var(--danger) 30%, transparent);
  border-radius: 100px; padding: 2px 10px; font-size: 0.74rem; font-weight: 600;
}}
.badge-up   {{
  display:inline-flex; align-items:center; gap:3px;
  background: var(--success-bg);
  color: var(--success); border-radius: 100px; padding: 2px 9px; font-size: 0.73rem; font-weight: 700;
}}
.badge-down {{
  display:inline-flex; align-items:center; gap:3px;
  background: var(--danger-bg);
  color: var(--danger); border-radius: 100px; padding: 2px 9px; font-size: 0.73rem; font-weight: 700;
}}
.badge-hold {{
  display:inline-flex; align-items:center; gap:3px;
  background: color-mix(in srgb, var(--text-secondary) 14%, transparent);
  color: var(--text-secondary); border-radius: 100px; padding: 2px 9px; font-size: 0.73rem; font-weight: 700;
}}
.badge-sim  {{
  display:inline-flex; align-items:center; gap:4px;
  background: var(--warning-bg);
  color: var(--warning);
  border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent);
  border-radius: 100px; padding: 3px 12px; font-size: 0.73rem; font-weight: 600;
}}

/* ── Hero banner ── */
.hero-banner {{
  background: linear-gradient(135deg, var(--hero-grad-a) 0%, var(--hero-grad-b) 55%, var(--hero-grad-c) 100%);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  padding: 1.6rem 2rem;
  margin-bottom: 1.4rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 32px var(--accent-glow), 0 1px 4px var(--shadow-card);
}}
.hero-banner::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at 75% 50%, var(--accent-glow) 0%, transparent 65%);
  pointer-events: none;
}}
.hero-icon {{
  font-size: 2.8rem;
  flex-shrink: 0;
  filter: drop-shadow(0 0 14px var(--accent));
  animation: rocketFloat 3.5s ease-in-out infinite;
}}
@keyframes rocketFloat {{
  0%, 100% {{ transform: translateY(0) rotate(0deg); }}
  33%  {{ transform: translateY(-5px) rotate(2deg); }}
  66%  {{ transform: translateY(-2px) rotate(-1deg); }}
}}
.hero-left  {{ display: flex; align-items: center; gap: 1rem; flex: 1; }}
.hero-text  {{ display: flex; flex-direction: column; gap: 0.25rem; }}
.hero-title {{
  font-size: 1.52rem; font-weight: 800; color: var(--text-primary);
  letter-spacing: -0.025em; line-height: 1.2;
}}
.hero-sub   {{
  font-size: 0.83rem; color: var(--text-secondary);
  max-width: 540px; line-height: 1.55;
}}
.hero-sub strong {{ color: var(--accent); font-weight: 600; }}
.hero-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.6rem; flex-shrink: 0; }}

/* ── Status pills ── */
.status-pill {{
  display: inline-flex; align-items: center; gap: 7px;
  background: var(--success-bg);
  border: 1px solid color-mix(in srgb, var(--success) 35%, transparent);
  color: var(--success); border-radius: 100px; padding: 5px 14px;
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.01em;
  box-shadow: 0 0 14px color-mix(in srgb, var(--success) 18%, transparent);
}}
.status-pill-offline {{
  display: inline-flex; align-items: center; gap: 7px;
  background: var(--danger-bg);
  border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
  color: var(--danger); border-radius: 100px; padding: 5px 14px;
  font-size: 0.75rem; font-weight: 700;
}}
.status-pill-unknown {{
  display: inline-flex; align-items: center; gap: 7px;
  background: color-mix(in srgb, var(--text-secondary) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--text-secondary) 25%, transparent);
  color: var(--text-secondary); border-radius: 100px; padding: 5px 14px;
  font-size: 0.75rem; font-weight: 700;
}}
@keyframes liveDot {{
  0%, 100% {{ opacity: 1; transform: scale(1); }}
  50%       {{ opacity: 0.35; transform: scale(0.8); }}
}}
.live-dot {{
  animation: liveDot 1.6s ease-in-out infinite;
  font-size: 0.62rem; display: inline-block;
}}

/* ── Section header / divider ── */
.section-header {{
  margin: 1.6rem 0 0.4rem 0;
}}
.section-title {{
  font-size: 1.0rem; font-weight: 700; color: var(--text-primary); margin: 0;
  letter-spacing: -0.01em;
}}
.section-sub {{
  font-size: 0.77rem; color: var(--text-secondary); margin: 0.12rem 0 0 0;
}}
.section-divider {{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, var(--accent) 0%, var(--border) 25%, transparent 100%);
  margin: 0.5rem 0 1rem 0;
  opacity: 0.55;
}}

/* ── Pod grid ── */
.pod-grid-wrapper {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 1.2rem;
  margin-bottom: 0.8rem;
}}
.pod-row-label {{
  font-size: 0.70rem; font-weight: 700; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;
}}
.pod-row {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 0.65rem; }}
.pod-icon {{
  width: 26px; height: 26px; border-radius: 6px;
  background: var(--accent); opacity: 0.72;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.65rem; font-weight: 700;
}}
.pod-icon.pod-new {{
  background: var(--success);
  animation: podIn 0.55s ease-out forwards, podPulse 1.4s ease-in-out 0.55s 2;
}}
.pod-icon.pod-removed {{
  background: var(--danger);
  animation: podOut 0.65s ease-in forwards;
}}
.pod-icon.pod-hpa         {{ background: var(--danger); opacity: 0.60; }}
.pod-icon.pod-hpa-new     {{ background: var(--danger); animation: podIn 0.55s ease-out forwards; }}
.pod-icon.pod-hpa-removed {{ background: var(--danger); opacity: 0.35; animation: podOut 0.65s ease-in forwards; }}
@keyframes podIn {{
  from {{ opacity: 0; transform: scale(0.2) rotate(-12deg); }}
  to   {{ opacity: 0.85; transform: scale(1) rotate(0deg); }}
}}
@keyframes podOut {{
  from {{ opacity: 0.85; transform: scale(1); }}
  to   {{ opacity: 0; transform: scale(0.15); }}
}}
@keyframes podPulse {{
  0%, 100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--success) 55%, transparent); }}
  50%       {{ box-shadow: 0 0 0 6px color-mix(in srgb, var(--success) 0%, transparent); }}
}}

/* ── Event feed ── */
.event-feed {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 0.8rem;
}}
.event-feed-header {{
  padding: 0.65rem 1rem;
  font-size: 0.70rem; font-weight: 700; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.08em;
  border-bottom: 1px solid var(--border);
  background: var(--surface2);
}}
.event-feed-row {{
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.55rem 1rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.82rem; color: var(--text-primary);
  transition: background 0.14s ease;
}}
.event-feed-row:hover {{ background: var(--surface2); }}
.event-feed-row:last-child {{ border-bottom: none; }}
.event-ts    {{ color: var(--text-secondary); font-size: 0.73rem; min-width: 52px; }}
.event-delta {{ color: var(--text-secondary); font-size: 0.80rem; }}
.event-reason {{ color: var(--text-secondary); font-size: 0.74rem; margin-left: auto; font-style: italic; }}

/* ── Progress bar ── */
.progress-outer {{
  background: var(--border); border-radius: 100px; height: 5px;
  overflow: hidden; margin: 0.35rem 0 0.8rem;
}}
.progress-inner {{
  height: 100%;
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
  border-radius: 100px;
  transition: width 0.4s cubic-bezier(.25,.46,.45,.94);
}}

/* ── Tab pill styling ── */
button[data-baseweb="tab"] {{
  font-size: 0.87rem !important; font-weight: 600 !important;
  border-radius: 100px !important;
  padding: 5px 16px !important;
  transition: background 0.18s ease, color 0.18s ease !important;
  letter-spacing: 0.01em;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--accent) !important;
  background: var(--accent-bg) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--sidebar-border) !important;
}}
[data-testid="stSidebar"] .block-container {{
  background: var(--sidebar-bg) !important;
}}

/* ── Button overrides ── */
.stButton > button {{
  border-radius: 10px;
  font-weight: 600; font-size: 0.86rem;
  transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease;
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text-primary);
}}
.stButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 4px 16px var(--shadow);
  border-color: var(--accent);
  color: var(--accent);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* ── Expander ── */
.streamlit-expanderHeader {{
  font-size: 0.84rem !important; font-weight: 600 !important;
  color: var(--text-secondary) !important; border-radius: 8px;
}}
.streamlit-expanderHeader:hover {{ color: var(--text-primary) !important; background: var(--surface2); }}

/* ── Footer ── */
.app-footer {{
  margin-top: 2.5rem;
  padding: 1rem 0;
  border-top: 1px solid var(--border);
  text-align: center;
  font-size: 0.73rem;
  color: var(--text-secondary);
  letter-spacing: 0.01em;
}}

/* ── Headings ── */
h1, h2, h3 {{ color: var(--text-primary) !important; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.05rem !important; font-weight: 700 !important; }}
hr {{ border-color: var(--border) !important; opacity: 0.6; }}

/* ── Alerts ── */
.stAlert {{ border-radius: 10px !important; }}

/* ── Custom scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-secondary); }}

/* ── Live indicator ── */
.live-indicator {{
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--success-bg);
  border: 1px solid color-mix(in srgb, var(--success) 28%, transparent);
  color: var(--success);
  border-radius: 100px; padding: 3px 10px;
  font-size: 0.70rem; font-weight: 700; letter-spacing: 0.05em;
}}

/* ── Comparison table ── */
.comparison-table {{
  width: 100%; border-collapse: collapse; border-radius: 10px; overflow: hidden; font-size: 0.84rem;
}}
.comparison-table th {{
  background: var(--surface2);
  color: var(--text-secondary);
  font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.07em;
  padding: 0.65rem 1rem; border-bottom: 1px solid var(--border); text-align: left;
}}
.comparison-table td {{
  padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); color: var(--text-primary);
}}
.comparison-table tr:last-child td {{ border-bottom: none; }}
.comparison-table tr:hover td {{ background: var(--surface2); }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
    return css
