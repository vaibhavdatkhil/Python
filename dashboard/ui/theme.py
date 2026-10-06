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
    "bg":             "#0d1117",
    "surface":        "#161b22",
    "border":         "#30363d",
    "text_primary":   "#f0f6fc",
    "text_secondary": "#8b949e",
    "accent":         "#58a6ff",
    "success":        "#3fb950",
    "warning":        "#ffa657",
    "danger":         "#f78166",
    "grid":           "#21262d",
    "colorway": [
        "#58a6ff",
        "#3fb950",
        "#ffa657",
        "#f78166",
        "#bc8cff",
        "#39d353",
    ],
    # Extra tokens used only in CSS
    "shadow":         "rgba(0,0,0,0.45)",
    "hero_grad_a":    "#0d1117",
    "hero_grad_b":    "#1a2744",
    "sidebar_bg":     "#0a0f1c",
    "sidebar_border": "#1e293b",
}

LIGHT: dict = {
    "bg":             "#ffffff",
    "surface":        "#f6f8fa",
    "border":         "#d0d7de",
    "text_primary":   "#24292f",
    "text_secondary": "#57606a",
    "accent":         "#0969da",
    "success":        "#1a7f37",
    "warning":        "#9a6700",
    "danger":         "#cf222e",
    "grid":           "#d0d7de",
    "colorway": [
        "#0969da",
        "#1a7f37",
        "#9a6700",
        "#cf222e",
        "#8250df",
        "#0550ae",
    ],
    # Extra tokens used only in CSS
    "shadow":         "rgba(0,0,0,0.12)",
    "hero_grad_a":    "#f0f7ff",
    "hero_grad_b":    "#dbeafe",
    "sidebar_bg":     "#f6f8fa",
    "sidebar_border": "#d0d7de",
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
        plot_bgcolor=p["bg"],
        paper_bgcolor=p["bg"],
        font=dict(family="Inter, sans-serif", color=p["text_secondary"], size=12),
        title_font=dict(family="Inter, sans-serif", color=p["text_primary"], size=14),
        xaxis=dict(
            gridcolor=p["grid"],
            showgrid=True,
            zeroline=False,
            linecolor=p["border"],
            tickfont=dict(color=p["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=p["grid"],
            showgrid=True,
            zeroline=False,
            linecolor=p["border"],
            tickfont=dict(color=p["text_secondary"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=p["border"],
            font=dict(color=p["text_secondary"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=48, r=20, t=44, b=48),
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── CSS variables ── */
:root {{
  --bg:            {p['bg']};
  --surface:       {p['surface']};
  --border:        {p['border']};
  --text-primary:  {p['text_primary']};
  --text-secondary:{p['text_secondary']};
  --accent:        {p['accent']};
  --success:       {p['success']};
  --warning:       {p['warning']};
  --danger:        {p['danger']};
  --grid:          {p['grid']};
  --shadow:        {p['shadow']};
  --hero-grad-a:   {p['hero_grad_a']};
  --hero-grad-b:   {p['hero_grad_b']};
  --sidebar-bg:    {p['sidebar_bg']};
  --sidebar-border:{p['sidebar_border']};
}}

/* ── Base typography ── */
html, body, [class*="css"] {{
  font-family: 'Inter', sans-serif;
  color: var(--text-primary);
}}

/* ── KPI / metric cards (glassmorphism) ── */
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.3rem;
  margin-bottom: 0.6rem;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
  position: relative;
  overflow: hidden;
}}
.kpi-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 28px var(--shadow);
}}
.kpi-card.status-ok   {{ border-left: 4px solid var(--success); }}
.kpi-card.status-warn {{ border-left: 4px solid var(--warning); }}
.kpi-card.status-err  {{ border-left: 4px solid var(--danger);  }}
.kpi-card.status-accent {{ border-left: 4px solid var(--accent); }}

.kpi-icon  {{
  position: absolute; top: 0.7rem; right: 0.9rem;
  font-size: 1.5rem; opacity: 0.55;
}}
.kpi-label {{
  font-size: 0.70rem; color: var(--text-secondary);
  letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600;
}}
.kpi-value {{
  font-size: 1.65rem; font-weight: 700; color: var(--text-primary);
  line-height: 1.2; margin: 0.15rem 0;
}}
.kpi-delta {{
  font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.1rem;
}}
/* Legacy sub-label used by kpi() helper — matches .kpi-delta styling */
.kpi-sub {{
  font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.1rem;
}}
/* Legacy sub-label used by kpi() helper — matches .kpi-delta styling */
.kpi-sub {{
  font-size: 0.68rem; color: var(--text-secondary); margin-top: 0.1rem;
}}

/* ── Badges ── */
.badge-ok {{
  display:inline-block; background: color-mix(in srgb, var(--success) 18%, transparent);
  color: var(--success); border: 1px solid color-mix(in srgb, var(--success) 35%, transparent);
  border-radius: 20px; padding: 2px 10px; font-size: 0.76rem; font-weight: 600;
}}
.badge-warn {{
  display:inline-block; background: color-mix(in srgb, var(--warning) 18%, transparent);
  color: var(--warning); border: 1px solid color-mix(in srgb, var(--warning) 35%, transparent);
  border-radius: 20px; padding: 2px 10px; font-size: 0.76rem; font-weight: 600;
}}
.badge-err {{
  display:inline-block; background: color-mix(in srgb, var(--danger) 18%, transparent);
  color: var(--danger); border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
  border-radius: 20px; padding: 2px 10px; font-size: 0.76rem; font-weight: 600;
}}
.badge-up   {{ display:inline-block; background: color-mix(in srgb, var(--success) 20%, transparent);
               color: var(--success); border-radius: 20px; padding: 2px 8px; font-size: 0.74rem; font-weight: 700; }}
.badge-down {{ display:inline-block; background: color-mix(in srgb, var(--danger) 20%, transparent);
               color: var(--danger);  border-radius: 20px; padding: 2px 8px; font-size: 0.74rem; font-weight: 700; }}
.badge-hold {{ display:inline-block; background: color-mix(in srgb, var(--text-secondary) 20%, transparent);
               color: var(--text-secondary); border-radius: 20px; padding: 2px 8px; font-size: 0.74rem; font-weight: 700; }}
.badge-sim  {{
  display:inline-block;
  background: color-mix(in srgb, var(--warning) 18%, transparent);
  color: var(--warning);
  border: 1px solid color-mix(in srgb, var(--warning) 35%, transparent);
  border-radius: 20px; padding: 2px 10px; font-size: 0.74rem; font-weight: 600;
}}

/* ── Hero banner ── */
.hero-banner {{
  background: linear-gradient(135deg, var(--hero-grad-a) 0%, var(--hero-grad-b) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem 1.8rem;
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.hero-left   {{ display: flex; flex-direction: column; gap: 0.2rem; }}
.hero-title  {{ font-size: 1.6rem; font-weight: 700; color: var(--text-primary); }}
.hero-sub    {{ font-size: 0.85rem; color: var(--text-secondary); max-width: 600px; line-height: 1.5; }}
.hero-right  {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.5rem; }}
.status-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  background: color-mix(in srgb, var(--success) 15%, transparent);
  border: 1px solid color-mix(in srgb, var(--success) 35%, transparent);
  color: var(--success); border-radius: 20px; padding: 4px 12px;
  font-size: 0.76rem; font-weight: 600;
}}
.status-pill-offline {{
  display: inline-flex; align-items: center; gap: 6px;
  background: color-mix(in srgb, var(--danger) 15%, transparent);
  border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
  color: var(--danger); border-radius: 20px; padding: 4px 12px;
  font-size: 0.76rem; font-weight: 600;
}}
.status-pill-unknown {{
  display: inline-flex; align-items: center; gap: 6px;
  background: color-mix(in srgb, var(--text-secondary) 15%, transparent);
  border: 1px solid color-mix(in srgb, var(--text-secondary) 35%, transparent);
  color: var(--text-secondary); border-radius: 20px; padding: 4px 12px;
  font-size: 0.76rem; font-weight: 600;
}}
@keyframes liveDot {{
  0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }}
}}
.live-dot {{ animation: liveDot 1.4s ease-in-out infinite; font-size: 0.7rem; }}

/* ── Section header / divider ── */
.section-header {{
  margin: 1.2rem 0 0.5rem 0;
}}
.section-title {{
  font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin: 0;
}}
.section-sub {{
  font-size: 0.78rem; color: var(--text-secondary); margin: 0.1rem 0 0 0;
}}
.section-divider {{
  border: none; border-top: 1px solid var(--border); margin: 0.4rem 0 0.9rem 0;
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
  font-size: 0.72rem; font-weight: 600; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;
}}
.pod-row {{
  display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 0.6rem;
}}
.pod-icon {{
  width: 26px; height: 26px; border-radius: 5px;
  background: var(--accent); opacity: 0.75;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 700;
}}
.pod-icon.pod-new {{
  background: var(--success);
  animation: podIn 0.6s ease-out forwards, podPulse 1.2s ease-in-out 0.6s 2;
}}
.pod-icon.pod-removed {{
  background: var(--danger);
  animation: podOut 0.7s ease-in forwards;
}}
.pod-icon.pod-hpa {{
  background: var(--danger); opacity: 0.65;
}}
.pod-icon.pod-hpa-new {{
  background: var(--danger);
  animation: podIn 0.6s ease-out forwards;
}}
.pod-icon.pod-hpa-removed {{
  background: var(--danger); opacity: 0.4;
  animation: podOut 0.7s ease-in forwards;
}}
@keyframes podIn {{
  from {{ opacity: 0; transform: scale(0.3); }}
  to   {{ opacity: 0.9; transform: scale(1); }}
}}
@keyframes podOut {{
  from {{ opacity: 0.9; transform: scale(1); }}
  to   {{ opacity: 0; transform: scale(0.3); }}
}}
@keyframes podPulse {{
  0%, 100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--success) 50%, transparent); }}
  50%       {{ box-shadow: 0 0 0 5px color-mix(in srgb, var(--success) 0%, transparent); }}
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
  padding: 0.6rem 1rem;
  font-size: 0.78rem; font-weight: 700; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border);
}}
.event-feed-row {{
  display: flex; align-items: center; gap: 0.7rem;
  padding: 0.5rem 1rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.82rem; color: var(--text-primary);
}}
.event-feed-row:last-child {{ border-bottom: none; }}
.event-ts  {{ color: var(--text-secondary); font-size: 0.75rem; min-width: 50px; }}
.event-delta {{ color: var(--text-secondary); font-size: 0.80rem; }}
.event-reason {{ color: var(--text-secondary); font-size: 0.76rem; margin-left: auto; font-style: italic; }}

/* ── Progress bar ── */
.progress-outer {{
  background: var(--border); border-radius: 4px; height: 5px;
  overflow: hidden; margin: 0.3rem 0 0.7rem;
}}
.progress-inner {{
  height: 100%; background: var(--accent);
  border-radius: 4px; transition: width 0.3s ease;
}}

/* ── Tab pill styling ── */
button[data-baseweb="tab"] {{
  font-size: 0.88rem; font-weight: 600;
  border-radius: 20px !important;
  padding: 4px 14px !important;
  transition: background 0.15s ease;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--accent) !important;
}}

/* ── Sidebar ── */
.stSidebar {{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--sidebar-border);
}}

/* ── Button overrides ── */
.stButton > button {{
  border-radius: 8px;
  font-weight: 600;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}}
.stButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--shadow);
}}

/* ── Footer ── */
.app-footer {{
  margin-top: 2rem;
  padding: 0.9rem 0;
  border-top: 1px solid var(--border);
  text-align: center;
  font-size: 0.75rem;
  color: var(--text-secondary);
}}

/* ── Headings ── */
h1, h2, h3 {{ color: var(--text-primary) !important; }}
hr {{ border-color: var(--border); }}

/* ── Custom scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-secondary); }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
    return css
