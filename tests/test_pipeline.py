"""
tests/test_pipeline.py
──────────────────────
Comprehensive Unit & Integration Test Suite for K8s RL Autoscaler.
Tests Phases 1 through 11:
  - Data loading & preprocessing
  - LSTM inference & MC-dropout uncertainty
  - Gymnasium RL environment
  - PPO Actor-Critic policy inference
  - Kubernetes cluster interfaces (Mock & Live fallback)
  - FastAPI schemas
  - Locust workload configuration
"""

import unittest
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from data.loader import load_trace
from preprocessing.pipeline import normalize, make_windows
from model.lstm_model import LSTMForecast
from model.inference import load_checkpoint, mc_predict
from rl_env.k8s_env import K8sAutoscalingEnv
from rl_agent.policy_net import ActorCritic
from rl_agent.ppo import load_ppo_checkpoint
from k8s.k8s_adapter import MockK8sCluster, RLAutoscalingController
from api.schemas import ForecastRequest, ScaleRequest


class TestDataAndPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PROJECT_ROOT / "config.yaml") as f:
            cls.cfg = yaml.safe_load(f)

    def test_data_loader(self):
        df = load_trace(cfg=self.cfg)
        self.assertIn("cpu_util", df.columns)
        self.assertGreater(len(df), 100)

    def test_preprocessing_pipeline(self):
        df = load_trace(cfg=self.cfg)
        series = df["cpu_util"].to_numpy()
        scaled, scaler = normalize(series)
        self.assertAlmostEqual(float(scaled.min()), 0.0, places=1)
        self.assertAlmostEqual(float(scaled.max()), 1.0, places=1)

        W = self.cfg["preprocessing"]["window_size"]
        H = self.cfg["preprocessing"]["horizon"]
        X, Y = make_windows(scaled, window=W, horizon=H)
        self.assertEqual(X.shape[1], W)
        self.assertEqual(Y.shape[1], H)


class TestLSTMForecasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        cls.device = torch.device("cpu")

    def test_checkpoint_and_mc_inference(self):
        if not self.ckpt_path.exists():
            self.skipTest("LSTM checkpoint best_model.pt not found")

        model, scaler, cfg = load_checkpoint(self.ckpt_path, device=self.device)
        self.assertIsInstance(model, LSTMForecast)

        W = cfg["preprocessing"]["window_size"]
        H = cfg["preprocessing"]["horizon"]
        dummy_window = np.linspace(0.2, 0.6, W)

        mean_fc, lower_fc, upper_fc = mc_predict(
            model, dummy_window, n_samples=10, scaler=scaler, device=self.device
        )
        self.assertEqual(len(mean_fc), H)
        self.assertEqual(len(lower_fc), H)
        self.assertEqual(len(upper_fc), H)
        self.assertTrue(np.all(upper_fc >= lower_fc))


class TestGymnasiumEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PROJECT_ROOT / "config.yaml") as f:
            cls.cfg = yaml.safe_load(f)
        cls.df = load_trace(cfg=cls.cfg)

    def test_env_lifecycle(self):
        env = K8sAutoscalingEnv(self.cfg, model=None, scaler=None, trace_df=self.df)
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.ndim, 1)

        # Execute 5 random steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIsInstance(reward, float)
            self.assertIn("replicas", info)
            if terminated or truncated:
                break


class TestPPOAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = PROJECT_ROOT / "checkpoints" / "ppo_agent.pt"
        if not cls.ckpt_path.exists():
            cls.ckpt_path = PROJECT_ROOT / "checkpoints" / "ppo_agent_final.pt"

    def test_ppo_policy_inference(self):
        if not self.ckpt_path.exists():
            self.skipTest("PPO checkpoint not found")

        policy, _ = load_ppo_checkpoint(self.ckpt_path, device=torch.device("cpu"))
        self.assertIsInstance(policy, ActorCritic)

        # Dummy observation matching obs_dim (60 + 15 + 15 + 1 + 1 = 92)
        dummy_obs = torch.zeros(1, 92, dtype=torch.float32)
        with torch.no_grad():
            action, log_prob, value, entropy = policy.act(dummy_obs)
        self.assertIn(action.item(), [0, 1, 2, 3, 4])


class TestK8sAdapterAndController(unittest.TestCase):
    def test_mock_cluster_lifecycle(self):
        cluster = MockK8sCluster(initial_replicas=3, lag_steps=1)
        self.assertEqual(cluster.get_replicas("cloud-service-app"), 3)

        # Scale up to 5
        scaled = cluster.scale("cloud-service-app", 5)
        self.assertTrue(scaled)

        # Before lag advances, replicas should remain 3
        self.assertEqual(cluster.get_replicas("cloud-service-app"), 3)

        # Advance time by 1 step
        cluster.advance_time(0.5)
        self.assertEqual(cluster.get_replicas("cloud-service-app"), 5)

    def test_controller_decision(self):
        cluster = MockK8sCluster(initial_replicas=3)
        controller = RLAutoscalingController(cluster=cluster)
        dummy_history = np.full(60, 0.40)

        res = controller.decide_and_scale(dummy_history, step_idx=0)
        self.assertIn("target_replicas", res)
        self.assertIn("delta", res)
        self.assertIn("slo_met", res)
        self.assertIn("uncertainty", res)


class TestAPISchemas(unittest.TestCase):
    def test_forecast_schema(self):
        req = ForecastRequest(cpu_window=[0.35] * 60, n_samples=20)
        self.assertEqual(len(req.cpu_window), 60)

    def test_scale_action_schema(self):
        req = ScaleRequest(
            cpu_window=[0.50] * 60,
            replicas_norm=0.3,
            cpu_per_pod_norm=0.65,
        )
        self.assertEqual(req.replicas_norm, 0.3)


class TestFixesVerification(unittest.TestCase):
    def test_env_scaling_lag_queue(self):
        """Fix 1: verify maxlen=lag + 1 buffers deltas properly for exact lag steps."""
        import copy
        with open(PROJECT_ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg_lag = copy.deepcopy(cfg)
        cfg_lag["rl_env"]["scale_lag_steps"] = 2
        cfg_lag["rl_env"]["initial_replicas"] = 3

        df = load_trace(cfg=cfg)
        env = K8sAutoscalingEnv(cfg_lag, model=None, scaler=None, trace_df=df)
        env.reset(seed=42)

        # Action 4 is DELTA=+2 (ACTION_TO_DELTA[4] == +2)
        # Action 2 is DELTA=0  (ACTION_TO_DELTA[2] == 0)

        # Step 1: delta +2 queued, effective delta applied is 0 (from initial fill)
        _, _, _, _, info1 = env.step(4)
        self.assertEqual(info1["replicas"], 3)

        # Step 2: delta 0 queued, effective delta applied is still 0
        _, _, _, _, info2 = env.step(2)
        self.assertEqual(info2["replicas"], 3)

        # Step 3: effective delta applied should now be +2 (lag=2 elapsed)
        _, _, _, _, info3 = env.step(2)
        self.assertEqual(info3["replicas"], 5)

    def test_azure_benchmark_trace_monday_and_weekends(self):
        """Fix 2: verify Azure trace starts on Monday and weekends are computed correctly."""
        from data.benchmark_traces import generate_azure_trace
        df_azure = generate_azure_trace(n_days=7, seed=42)
        first_ts = df_azure["timestamp"].iloc[0]
        self.assertEqual(first_ts.day_name(), "Monday")

        weekends = df_azure[df_azure["timestamp"].dt.dayofweek >= 5]
        weekdays = df_azure[df_azure["timestamp"].dt.dayofweek < 5]
        self.assertGreater(len(weekends), 0)
        self.assertGreater(len(weekdays), 0)
        # Weekday load should exceed weekend load due to weekly_factor
        self.assertGreater(weekdays["cpu_util"].mean(), weekends["cpu_util"].mean())

    def test_locust_task_includes_work_endpoint(self):
        """Fix 3: verify Locust user has a task hitting the /work endpoint."""
        locust_path = PROJECT_ROOT / "locust" / "locustfile.py"
        self.assertTrue(locust_path.exists())
        content = locust_path.read_text(encoding="utf-8")
        self.assertIn('"/work"', content)
        self.assertIn("compute_work", content)

    def test_benchmark_files_and_alibaba_evaluation(self):
        """Fix 4: verify Azure and Alibaba trace CSVs exist and can be loaded for evaluation."""
        import pandas as pd
        azure_path = PROJECT_ROOT / "data" / "azure_vm_workload_trace.csv"
        alibaba_path = PROJECT_ROOT / "data" / "alibaba_cluster_trace.csv"
        self.assertTrue(azure_path.exists())
        self.assertTrue(alibaba_path.exists())

        df_ali = pd.read_csv(alibaba_path, parse_dates=["timestamp"])
        self.assertIn("cpu_util", df_ali.columns)
        self.assertGreater(len(df_ali), 1000)


if __name__ == "__main__":
    unittest.main()
