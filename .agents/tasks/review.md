# Dashboard Redesign: K8s RL Autoscaling Streamlit UI

The redesign restructures the Streamlit dashboard into a proper module hierarchy (`ui/theme.py`, `ui/components.py`, `ui/charts.py`, `app.py`), adds a dark/light theme system, and places the new "Pod Scaling Activity" composite visualization above the tabs. All six original tabs are present and all original session state keys are initialized. The primary concerns are a `kpi-sub` CSS class that is referenced but never defined, the Scaling Decisions tab rebuilding its chart from scratch instead of reusing the shared composite figure, the theme toggle placed in the sidebar rather than the header, and missing `@st.cache_data` on data transforms and `-> None` return type annotations on tab render functions.

Watch for:
- **`kpi-sub` undefined CSS** (confirmed) — the legacy `kpi()` helper emits `<div class="kpi-sub">` but the class is absent from `inject_css`; all legacy KPI sub-labels are invisible.
- **Scaling Decisions tab does not reuse the composite figure** (confirmed) — spec requires `make_scaling_activity_figure` and `render_pod_grid` in the tab; the tab builds a standalone `go.Figure` instead and skips the pod grid entirely.
- **`kpi_calibration` injects inline style** (confirmed) — palette-derived colors are used, so the hex values do come from `get_palette()`, but the function writes `style="border-left: 4px solid {color};"` directly into HTML rather than using the `.status-ok/.status-warn/.status-err` CSS classes that `metric_card()` provides. This means the calibration card does not respond to future CSS class refactors.

**Verdict**: NEEDS_CHANGES

---

## High-level view

Theme tokens are fully centralized in `DARK`/`LIGHT` dicts in `theme.py` and surfaced through `inject_css` as CSS variables. Every Plotly figure builder calls `get_palette(theme)` and `get_plotly_template(theme)`, so theme switching propagates through all charts. The one gap is the `kpi-sub` class: the legacy `kpi()` helper in `app.py` is preserved for backward compatibility but references a CSS class that was never added to the injected stylesheet, making its sub-label text invisible in both modes.

The three-panel `make_scaling_activity_figure` is present in `charts.py` with all specified elements: step-lines, filled RL area, scale-up/down markers with `+N`/`−N` labels, hold dots, proactive-scaling vrects, dynamic y-axis range, and the diverging bar panel. `render_pod_grid` in `components.py` uses `st.components.v1.html` for iframe isolation and implements `podIn`/`podOut`/`podPulse` CSS keyframe animations. Both are correctly called above the tabs in `main()`.

The Scaling Decisions tab (`render_scaling_decisions_tab`) does not reuse these components. It builds a separate `go.Figure` with a single replica-count panel, no pod grid, no event feed, and no diverging bar chart. The spec explicitly requires reuse; the duplication also means a second independent rendering path to maintain.

All eight session state keys (`lstm_step`, `lstm_running`, `mae_acc`, `rmse_acc`, `cov_acc`, `rl_episode`, `rl_step`, `rl_running`) are initialized in their respective tab blocks. The `simulate_hpa_replicas` / `simulate_predictive_rl_replicas` helpers in `app.py` are data-transform functions that run on every rerun; neither carries `@st.cache_data`, so the full simulation loop re-executes on every user interaction, including theme toggles.

---

<details>
<summary>Issues (5)</summary>

1. **`kpi-sub` class missing from CSS** — the legacy `kpi()` helper emits `<div class="kpi-sub">` but `inject_css` in `theme.py` defines no `.kpi-sub` rule. All sub-labels in the Forecasting, RL, and Training KPI rows are invisible. Add `.kpi-sub { font-size: 0.68rem; color: var(--text-secondary); }` to `inject_css`, or migrate all callers to `metric_card()` which uses the correctly defined `.kpi-delta`.

2. **Scaling Decisions tab does not reuse the composite figure or pod grid** — `render_scaling_decisions_tab` constructs a standalone `go.Figure` and omits `render_pod_grid`, `render_event_feed`, and `make_scaling_activity_figure`. The spec requires reuse of these components. Refactor the function to call `make_scaling_activity_figure` (passing available CPU/forecast arrays) and `render_pod_grid` for the current endpoint replicas.

3. **`kpi_calibration` bypasses component abstraction** — writes `style="border-left: 4px solid {color};"` inline rather than using `metric_card(..., status=badge_lvl)`. If the card CSS is refactored, this function will desync. Migrate to `metric_card()`.

4. **`simulate_hpa_replicas` and `simulate_predictive_rl_replicas` not cached** — both functions run O(n) loops on every rerun, including theme toggles. The spec requires `@st.cache_data` on data transforms. Wrap the simulation calls (or the functions themselves) with `@st.cache_data`.

5. **`render_scaling_decisions_tab` and `render_training_tab` missing return type annotation** — all other public functions in `components.py` and `charts.py` carry `-> str` or `-> go.Figure`. These two render functions lack `-> None`, breaking the consistent type-hint contract the spec requires.

</details>

---

<details>
<summary>Details</summary>

## `kpi-sub` CSS class absent from stylesheet

The `kpi()` helper at line 93 of `app.py` is preserved for backward compatibility and emits:

```html
<div class="kpi-sub">{sub}</div>
```

`inject_css` in `theme.py` defines `.kpi-label`, `.kpi-value`, `.kpi-delta`, and `.kpi-icon` — but not `.kpi-sub`. Any tab that still calls `kpi()` with a sub-label (Forecasting step/MAE/RMSE KPIs, RL tab reward/SLO/replicas KPIs, Training tab final-reward/SLO/replicas KPIs) renders those sub-labels as unstyled text inheriting nothing, which in dark mode makes them indistinguishable from the background. This is confirmed by tracing `inject_css` end-to-end; the class string does not appear anywhere in the generated `<style>` block.

## Scaling Decisions tab: duplicate figure builder, missing pod grid

`render_scaling_decisions_tab` creates `fig_scale = go.Figure()` and manually adds HPA and RL Agent scatter traces, then calls `_get_layout()` directly. This duplicates the replica-count logic already in `make_scaling_activity_figure` Panel A, but it:

- Has no Panel B (CPU/forecast overlay) or Panel C (diverging bar chart).
- Does not call `render_pod_grid`, so the tab has no animated pod visualisation.
- Does not call `render_event_feed`, so the tab shows no latest-decisions feed.
- Uses `st.caption(...)` for the simulated-data notice rather than the dedicated `render_simulated_badge()`.

The spec states "Scaling Decisions: reuse the new Pod Scaling Activity components, plus a comparison table". None of those reuse points are present.

## `kpi_calibration` inline style vs. component system

`kpi_calibration` computes `color = p["danger" | "warning" | "success"]` from the palette, then injects it as `style="border-left: 4px solid {color}; ... style="color: {color};"`. The component system's `metric_card()` already exposes a `status` parameter that maps `"ok"/"warn"/"err"` to the same CSS-variable-driven border rules without inline styles. The calibration card is the only card that writes inline styles; it will not benefit from future CSS-variable transitions and is invisible to the theme test path.

## Missing `@st.cache_data` on simulation helpers

`simulate_hpa_replicas` and `simulate_predictive_rl_replicas` are called unconditionally in `main()` on the main page on every Streamlit rerun. Each runs an O(n) Python loop (n=200 for the default slice). A theme toggle, sidebar slider move, or any widget interaction triggers a full rerun. The spec explicitly calls for `@st.cache_data` on data transforms. Neither function nor its call sites use it. For `render_scaling_decisions_tab`, the same functions are called again inside the tab, doubling the work.

## Tab render functions missing return type annotations

`render_scaling_decisions_tab` and `render_training_tab` are defined without a return type:

```python
def render_scaling_decisions_tab(cfg: dict, df_full: pd.DataFrame, rl_ep: dict | None = None, theme: str = "dark"):
def render_training_tab(ppo_history: dict | None = None, theme: str = "dark"):
```

All functions in `components.py` and `charts.py` are fully annotated. The spec requires type hints on new functions. Adding `-> None` is the fix.

</details>

---

<details>
<summary>File map</summary>

| File | What changed |
|---|---|
| `dashboard/ui/theme.py` | New file: `DARK`/`LIGHT` palette dicts, `get_palette()`, `get_plotly_template()`, `inject_css()` |
| `dashboard/ui/components.py` | New file: `metric_card`, `render_metric_row`, `render_pod_grid`, `render_header`, `render_event_feed`, `render_section_header`, `render_footer`, `render_progress_bar`, `render_how_to_read`, `render_simulated_badge`, `badge` |
| `dashboard/ui/charts.py` | New file: `make_scaling_activity_figure` (3-panel composite), plus all single-panel figure builders for Forecasting, RL, Training tabs |
| `dashboard/app.py` | Restructured entry point: sidebar toggle, Pod Scaling Activity section above tabs, 6 `with tab_*:` blocks, legacy `kpi()`/`badge()`/`_dark_layout()` wrappers preserved |
| `.streamlit/config.toml` | New file: dark base theme, primaryColor/backgroundColor/textColor tokens, headless server config |

Full diff: `git diff main -- dashboard/ .streamlit/`

</details>
