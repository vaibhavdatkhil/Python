# K8s RL Autoscaling Dashboard — Redesign

The redesign restructures a single-file Streamlit app into a `ui/` module layer
(`theme.py`, `components.py`, `charts.py`) with a centralized color system,
glassmorphism cards, animated pod grid, and a 3-panel composite scaling-activity
figure placed above the tabs. All existing session state keys, replay logic,
model loading, and API calls are preserved. The dark/light toggle is wired in the
sidebar and correctly persists in `st.session_state["theme"]`.

**Watch for:** Three hardcoded hex literals in `kpi_calibration()` bypass the
theme system (confirmed). A duplicate `badge()` function definition in `app.py`
silently overwrites the first and bypasses the `ui.components` delegation
(confirmed). `.streamlit/config.toml` sets `font = "sans serif"` rather than
Inter, and is missing Inter + custom theme color entries (confirmed).

**Verdict**: NEEDS_CHANGES

---

## High-level view

The color-token architecture is correct: `theme.py` holds every hex literal, both
palettes expose the same keys, and `inject_css` maps them to CSS custom
properties. Charts and components consume `get_palette()` exclusively — except
for `kpi_calibration()` in `app.py`, which hardcodes `#f87171`, `#fbbf24`, and
`#4ade80` directly into inline styles. In light mode those colors will not match
the LIGHT palette tokens, breaking the visual contract the theme system was built
to provide.

The duplicate `badge()` at line 177 of `app.py` shadows the properly-delegating
wrapper at line 106. Every call to `badge()` in `app.py` after that point hits
the raw inline implementation rather than the `ui.components` version. If the CSS
class definitions in `theme.py` ever change structure, this shadow copy will drift
silently.

The Pod Scaling Activity section is present above the tabs, with the 3-panel
composite figure (`make_scaling_activity_figure`), animated pod grid
(`render_pod_grid`), KPI metric row, simulated-data badge, and event feed — all
per spec. The composite figure uses `make_subplots` with shared x-axis, step
lines with filled area, scale-up/down markers with `+N`/`-N` labels, vertical
shaded proactive bands, and a diverging bar chart in Panel C.

The `render_header` component intentionally does not embed `st.toggle` — its
docstring explicitly calls out that Streamlit widgets can't live inside raw HTML.
The toggle is in the sidebar instead, which is within the spec's stated
alternatives ("top-right of the header **or sidebar**"). Session state is read
back after the sidebar block so the chosen theme propagates correctly.

The `.streamlit/config.toml` exists and sets `base = "dark"` and
`headless = true`. However `font = "sans serif"` does not load Inter, and there
are no `[theme]` color entries (`primaryColor`, `backgroundColor`, etc.) to
reinforce the custom palette for users who haven't run the app long enough to
receive the injected CSS. This is a minor gap — the CSS `@import` of Inter fires
on every render, so fonts load correctly in practice — but the config could
explicitly declare Inter and matching background/text colors.

All eight required session state keys (`lstm_step`, `lstm_running`, `mae_acc`,
`rmse_acc`, `cov_acc`, `rl_episode`, `rl_step`, `rl_running`) are initialized
and used in their expected tabs. All six tab functions are present as `with
tab_*:` blocks. Imports from `dashboard.ui.theme`, `dashboard.ui.components`, and
`dashboard.ui.charts` are all at the top of `app.py`.

---

<details>
<summary>Issues (3)</summary>

1. **Hardcoded hex colors in `kpi_calibration()`** — `#f87171`, `#fbbf24`, and `#4ade80` are hardcoded in `app.py` lines 153–161. In light mode these Tailwind-style reds/ambers/greens do not match the LIGHT palette's `danger`/`warning`/`success` tokens (`#cf222e`, `#9a6700`, `#1a7f37`). Replace with `p = get_palette(theme)` and reference `p["danger"]`, `p["warning"]`, `p["success"]` — requires passing `theme` to `kpi_calibration()`.

2. **Duplicate `badge()` definition shadows the delegating wrapper** — `app.py` defines `badge()` twice: the first (line 106) correctly delegates to `ui.components.badge`; the second (line 177) is a raw inline implementation that overwrites the first in Python's namespace. All `badge()` calls in `app.py` after line 177 bypass `ui.components`. Remove the second definition.

3. **`config.toml` does not declare Inter or theme color tokens** — `font = "sans serif"` will not load Inter for browsers that can't hit Google Fonts (air-gapped envs, CI). Add `font = "monospace"` or simply remove the font key and rely on the `@import` in `inject_css`. Optionally populate `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor` in `[theme]` so the Streamlit default chrome (sidebar, modals) also matches the custom palette from the first render frame.

</details>

---

<details>
<summary>Details</summary>

### Hardcoded hex colors in `kpi_calibration()` bypass theme system

`kpi_calibration()` in `app.py` (lines 153–161) constructs inline `style=` attributes with `#f87171` (Tailwind red-400), `#fbbf24` (Tailwind amber-400), and `#4ade80` (Tailwind green-400). These are dark-mode-friendly Tailwind values that do not correspond to the LIGHT palette's danger/warning/success tokens. In light mode, the CI Coverage card will show a washed-out pastel red against a white card surface rather than the palette's saturated `#cf222e`. The function also does not accept a `theme` argument, making the mismatch structural rather than accidental.

Fix: add `theme: str = "dark"` parameter, call `get_palette(theme)`, replace the three literals with `p["danger"]`, `p["warning"]`, `p["success"]`.

### Duplicate `badge()` silently breaks component isolation

`app.py` defines `badge()` at line 106 as a thin wrapper that calls `_badge_ui` from `ui.components`, preserving the single-source-of-truth contract. A second `def badge(...)` at line 177 — an unconditional inline reimplementation — overwrites the first name in the module namespace. Python executes both `def` statements; the second wins. From line 177 onward, every `badge("Online", "ok")` call in the API Status tab, the sidebar model status block, and anywhere else in `app.py` hits the bare implementation. The intent of the first wrapper is voided without any import or lint error.

The second definition appears to be a copy-paste survival from the pre-refactor code. It should be deleted.

### `config.toml` gap — font and palette tokens

`font = "sans serif"` in `[theme]` does not instruct the browser to load Inter — it tells Streamlit's component system to use a generic sans-serif stack. Inter arrives only through the `@import url(...)` inside `inject_css()`, which fires on every Streamlit render. For environments where Google Fonts is blocked this silently degrades. Adding `font = "sans serif"` is harmless but provides no benefit over the existing `@import`; removing it or leaving it as-is are both acceptable. The more meaningful gap is the absence of `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, and `textColor` under `[theme]`. Without these, the Streamlit-native chrome (sidebar top bar, modal dialogs, file uploader) renders in Streamlit's own dark defaults on the very first frame before `inject_css` fires, producing a flash of un-themed UI.


</details>

---

<details>
<summary>File map</summary>

- `dashboard/app.py` — main entry; contains hardcoded hex colors in `kpi_calibration()`, duplicate `badge()` definition; all session state, tabs, Pod Scaling Activity section present
- `dashboard/ui/theme.py` — centralized color palettes (DARK/LIGHT), `get_palette()`, `get_plotly_template()`, `inject_css()` — fully compliant, no stray hex outside palette definitions
- `dashboard/ui/components.py` — `metric_card`, `render_metric_row`, `badge`, `render_header`, `render_pod_grid`, `render_event_feed`, `render_progress_bar`, `render_how_to_read`, `render_footer`, `render_simulated_badge`, `render_section_header` — all present with type hints and docstrings
- `dashboard/ui/charts.py` — `make_scaling_activity_figure`, `make_workload_trace_figure`, `make_forecast_figure`, `make_rolling_mae_figure`, `make_replica_trace_figure`, `make_cpu_per_pod_figure`, `make_action_distribution_figure`, `make_cumulative_reward_figure`, `make_training_reward_figure`, `make_loss_curves_figure`, `make_training_history_figure` — all present with type hints and docstrings
- `.streamlit/config.toml` — exists; base dark mode, headless server; missing Inter font entry and palette color tokens

Full diff: `git diff main -- dashboard/ .streamlit/`

</details>
