"""
tests/test_dashboard.py
───────────────────────
Unit tests for the dashboard UI module: theme, components, charts, and helpers.
"""

import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.ui.theme import DARK, LIGHT, get_palette, get_plotly_template
from dashboard.ui.components import (
    metric_card,
    badge,
)
from dashboard.ui.charts import (
    make_scaling_activity_figure,
    make_workload_trace_figure,
    make_forecast_figure,
    make_rolling_mae_figure,
    make_replica_trace_figure,
    make_cpu_per_pod_figure,
    make_action_distribution_figure,
    make_cumulative_reward_figure,
    make_training_reward_figure,
    make_loss_curves_figure,
    make_training_history_figure,
)
from dashboard.app import compute_calibration, kpi, simulate_hpa_replicas, simulate_predictive_rl_replicas


class TestThemeModule(unittest.TestCase):
    def test_palette_keys(self):
        required_keys = [
            "bg", "surface", "surface2", "border", "border2",
            "text_primary", "text_secondary", "accent", "success",
            "warning", "danger", "grid", "colorway",
        ]
        for key in required_keys:
            self.assertIn(key, DARK, f"Missing key '{key}' in DARK palette")
            self.assertIn(key, LIGHT, f"Missing key '{key}' in LIGHT palette")

    def test_get_palette(self):
        self.assertEqual(get_palette("dark")["bg"], DARK["bg"])
        self.assertEqual(get_palette("light")["bg"], LIGHT["bg"])
        # Fallback to light if not dark
        self.assertEqual(get_palette("other")["bg"], LIGHT["bg"])

    def test_plotly_template(self):
        dark_tpl = get_plotly_template("dark")
        self.assertIn("font", dark_tpl)
        self.assertIn("colorway", dark_tpl)
        self.assertEqual(dark_tpl["colorway"], DARK["colorway"])


class TestComponentsModule(unittest.TestCase):
    def test_metric_card_html(self):
        html = metric_card(
            label="TEST_LABEL",
            value="42",
            delta="+5",
            icon="🚀",
            status="ok",
            theme="dark",
        )
        self.assertIn("TEST_LABEL", html)
        self.assertIn("42", html)
        self.assertIn("+5", html)
        self.assertIn("🚀", html)
        self.assertIn("status-ok", html)

    def test_badge_html(self):
        self.assertIn("badge-ok", badge("Healthy", "ok"))
        self.assertIn("badge-err", badge("Down", "err"))

    def test_kpi_helper(self):
        card = kpi("REPLICAS", "5", "steady", icon="🤖")
        self.assertIn("REPLICAS", card)
        self.assertIn("5", card)


class TestChartsModule(unittest.TestCase):
    def setUp(self):
        self.n = 50
        self.hpa = np.ones(self.n, dtype=int) * 3
        self.rl = np.ones(self.n, dtype=int) * 3
        self.rl[20:25] = 5
        self.cpu = np.linspace(0.2, 0.8, self.n)

    def test_scaling_activity_figure(self):
        fig = make_scaling_activity_figure(
            hpa_replicas=self.hpa,
            rl_replicas=self.rl,
            cpu_series=self.cpu,
            mean_fc=None,
            lower_fc=None,
            upper_fc=None,
            slo_threshold=0.70,
            is_simulated=True,
            theme="dark",
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 2)

    def test_workload_trace_figure(self):
        df = pd.DataFrame({"cpu_util": np.random.rand(30)})
        fig = make_workload_trace_figure(df, cur_pos_index=15, theme="dark")
        self.assertIsInstance(fig, go.Figure)

    def test_forecast_figure(self):
        actual = np.random.rand(10)
        mean_fc = np.random.rand(10)
        lower_fc = mean_fc - 0.1
        upper_fc = mean_fc + 0.1
        fig = make_forecast_figure(actual, mean_fc, lower_fc, upper_fc, horizon=10, ci_lower=1, ci_upper=99)
        self.assertIsInstance(fig, go.Figure)

    def test_replica_and_cpu_figures(self):
        fig_rep = make_replica_trace_figure([2, 3, 4, 3], replay_s=2)
        self.assertIsInstance(fig_rep, go.Figure)

        fig_cpp = make_cpu_per_pod_figure(np.array([0.4, 0.5, 0.7]), slo_thr=0.70, replay_s=1)
        self.assertIsInstance(fig_cpp, go.Figure)

    def test_action_and_reward_figures(self):
        fig_act = make_action_distribution_figure([0, 1, 2, 2, 3, 4])
        self.assertIsInstance(fig_act, go.Figure)

        fig_rew = make_cumulative_reward_figure([1.0, -0.5, 2.0], replay_s=1)
        self.assertIsInstance(fig_rew, go.Figure)

    def test_training_figures(self):
        df_log = pd.DataFrame({
            "rollout": [1, 2, 3],
            "mean_reward": [10.0, 20.0, 30.0],
            "slo_pct": [80.0, 85.0, 92.0],
            "avg_replicas": [3.2, 3.0, 2.8],
            "policy_loss": [0.5, 0.3, 0.2],
            "value_loss": [0.8, 0.4, 0.1],
        })
        fig_tr = make_training_reward_figure(df_log)
        self.assertIsInstance(fig_tr, go.Figure)

        fig_loss = make_loss_curves_figure(df_log)
        self.assertIsInstance(fig_loss, go.Figure)

        fig_hist = make_training_history_figure([10.0, 15.0, 20.0], w_rm=2)
        self.assertIsInstance(fig_hist, go.Figure)


class TestCalibrationAndSimulations(unittest.TestCase):
    def test_compute_calibration(self):
        actuals = np.array([0.5, 0.6, 0.7, 0.8])
        lower = np.array([0.4, 0.4, 0.4, 0.4])
        upper = np.array([0.9, 0.9, 0.9, 0.9])
        cov = compute_calibration(actuals, lower, upper)
        self.assertEqual(cov, 100.0)

        upper_low = np.array([0.45, 0.45, 0.45, 0.45])
        cov_zero = compute_calibration(actuals, lower, upper_low)
        self.assertEqual(cov_zero, 0.0)

    def test_simulations(self):
        cpu = np.array([0.2, 0.3, 0.8, 0.9, 0.3, 0.2])
        hpa = simulate_hpa_replicas(cpu, min_replicas=1, max_replicas=10)
        rl = simulate_predictive_rl_replicas(cpu, min_replicas=1, max_replicas=10)
        self.assertEqual(len(hpa), len(cpu))
        self.assertEqual(len(rl), len(cpu))
        self.assertTrue(np.all(hpa >= 1))
        self.assertTrue(np.all(rl >= 1))


if __name__ == "__main__":
    unittest.main()
