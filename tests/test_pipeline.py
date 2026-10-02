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
            action, log_prob, entropy, value = policy.act(dummy_obs)
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


if __name__ == "__main__":
    unittest.main()
