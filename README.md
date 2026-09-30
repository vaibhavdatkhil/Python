# K8s RL Autoscaling — Complete Project (Phases 1–10)

> **Final-year B.Tech project.**  
> This repository contains the **complete implementation** of an intelligent
> Kubernetes pod autoscaling system powered by **LSTM + MC-Dropout forecasting**
> and a **PPO reinforcement learning agent**, with **SHAP explainability** and
> a **FastAPI REST backend**.  All 10 phases are implemented and working.

---

## Project Structure

```
BtechProject/
├── config.yaml               ← central config (edit me)
├── requirements.txt
├── README.md
├── data/
│   ├── synthetic_generator.py  ← synthetic cloud workload trace
│   ├── benchmark_traces.py     ← Azure VM & Alibaba Cluster benchmark generators
│   ├── azure_vm_workload_trace.csv ← 14-day Azure cloud benchmark trace
│   ├── alibaba_cluster_trace.csv   ← 14-day Alibaba cloud benchmark trace
│   └── loader.py               ← unified data-loading interface
├── preprocessing/
│   └── pipeline.py             ← resample → normalize → sliding windows
├── model/
│   ├── lstm_model.py           ← LSTM + MC Dropout architecture
│   ├── train.py                ← training loop + early stopping
│   ├── inference.py            ← stochastic MC forward passes → CI bands
│   └── evaluate.py             ← metrics + plot generation
├── rl_env/                     ← Phase 6 — Gymnasium RL environment
│   ├── k8s_env.py              ← K8sAutoscalingEnv (main Gymnasium env)
│   ├── spaces.py               ← observation/action space definitions
│   └── wrappers.py             ← RecordEpisodeStats wrapper
├── rl_agent/                   ← Phase 7 — PPO RL agent (from scratch)
│   ├── policy_net.py           ← Actor-Critic MLP (shared backbone)
│   ├── ppo.py                  ← RolloutBuffer, GAE, PPO-Clip update loop
│   ├── train_rl.py             ← CLI: train PPO agent
│   └── evaluate_rl.py          ← CLI: evaluate + plot generation
├── explainability/             ← Phase 8 — SHAP explainability
│   └── shap_explain.py         ← KernelSHAP for LSTM + PPO
├── api/                        ← Phase 9 — FastAPI REST backend
│   ├── main.py                 ← FastAPI app (forecast, scale-action, metrics)
│   └── schemas.py              ← Pydantic request/response models
├── dashboard/
│   └── app.py                  ← Multi-tab Streamlit dashboard (Phase 10)
├── k8s/                        ← Kubernetes Adapter & Controller
│   ├── k8s_adapter.py          ← Intelligent controller (Mock + Live kubectl modes)
│   └── demo_deployment.yaml    ← Sample microservice deployment manifest
├── checkpoints/                ← created automatically at train/rl-train time
└── outputs/                    ← plots: evaluation, rl_evaluation, shap_*
```

---

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Python 3.11** is recommended.

---

## Running Each Stage

Run the stages **in order** from the project root.

### 1 — Data Pipeline (sanity check)
```bash
python -m data.loader
python -m preprocessing.pipeline
```
Expected output: trace shape, head, and window array shapes printed to stdout.

### 2 — Train the LSTM Model
```bash
python -m model.train
```
- Reads `config.yaml` for all hyperparameters.
- Saves the best checkpoint to `checkpoints/best_model.pt`.
- Prints epoch-by-epoch train/val loss.

### 3 — Evaluate
```bash
python -m model.evaluate
```
- Loads `checkpoints/best_model.pt`.
- Prints **MAE**, **RMSE**, and **interval coverage** (% of actuals inside the
  predicted 5th–95th percentile band).
- Saves the forecast-vs-actual plot with shaded CI band to
  `outputs/evaluation_plot.png`.

### 4 — Dashboard
```bash
streamlit run dashboard/app.py
```
- Opens at `http://localhost:8501`.
- Use the sidebar sliders to control replay speed and MC sample count.

### 5 — RL Environment smoke test  _(Phase 6)_
```bash
python -m rl_env.k8s_env              # 200 random steps (no model needed)
python -m rl_env.k8s_env --steps 500  # longer run
python -m rl_env.k8s_env --no-model   # force zero-forecast mode
```
- Runs 200 random-action steps and asserts all invariants.
- Prints SLO compliance %, avg replica count, and total reward.
- Also calls `gymnasium.utils.env_checker.check_env` for spec compliance.

> **Tip**: Train the model first (`python -m model.train`) so the environment
> uses the LSTM forecast in its observation vector.

### 6 — Train PPO Agent  _(Phase 7)_
```bash
python -m rl_agent.train_rl
# Quick smoke-test (2048 steps, logs every rollout):
python -m rl_agent.train_rl --total-timesteps 2048 --log-interval 1
# Skip LSTM forecast (faster, no checkpoint needed):
python -m rl_agent.train_rl --no-model
# Override total training budget:
python -m rl_agent.train_rl --total-timesteps 500000
```
- Loads `config.yaml` for all PPO hyperparameters (see `ppo:` section).
- Optionally loads the LSTM checkpoint for richer observations.
- Saves the best checkpoint to `checkpoints/ppo_agent.pt` and the final
  weights to `checkpoints/ppo_agent_final.pt`.
- Logs per-rollout: reward, SLO%, avg replicas, policy loss, value loss,
  entropy, FPS, elapsed time.

### 8 — SHAP Explainability  _(Phase 8)_
```bash
python -m explainability.shap_explain
# Faster: skip LSTM (PPO only)
python -m explainability.shap_explain --no-lstm
# More accurate: larger background + explain sets
python -m explainability.shap_explain --n-bg 50 --n-explain 100
```
- Applies **KernelSHAP** to both the LSTM forecaster and the PPO actor.
- Saves `outputs/shap_lstm.png` — timestep importances in the CPU look-back window.
- Saves `outputs/shap_ppo.png` — observation feature importances per scaling action.

### 9 — FastAPI REST Backend  _(Phase 9)_
```bash
uvicorn api.main:app --reload --port 8000
# or:
python -m api.main
```
- Opens at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs` (Swagger) or `/redoc`
- Key endpoints:
  - `GET /health` — liveness probe
  - `POST /forecast` — LSTM MC-Dropout forecast for a CPU window
  - `POST /scale-action` — PPO agent picks scaling action
  - `GET /metrics` — latest episode stats
  - `POST /run-episode` — run a full PPO episode via the API

### 10 — Multi-Tab Dashboard  _(Phase 10)_
```bash
streamlit run dashboard/app.py
```
- Opens at `http://localhost:8501`
- **Tab 1 — 📈 Forecasting**: live workload replay + MC-Dropout CI forecast with calibration tuning
- **Tab 2 — ⚙️ Scaling Decisions**: HPA baseline vs predictive RL step comparison with action markers
- **Tab 3 — 🎓 Training**: reward-curve convergence mockup + checkpoint telemetry
- **Tab 4 — 🤖 RL Episode Replay**: full PPO episode execution, replica trace, SLO gauge
- **Tab 5 — 🔍 SHAP Insights**: embedded SHAP explanation plots
- **Tab 6 — 🌐 API Status**: ping FastAPI backend, run episodes remotely

### 11 — Kubernetes Adapter & Controller  _(Live / Mock Cluster)_
```bash
# Run 30 steps of mock autoscaling against simulated cloud workload:
python -m k8s.k8s_adapter --mode mock --steps 30

# Deploy sample service and run live against a real Minikube / K8s cluster:
kubectl apply -f k8s/demo_deployment.yaml
python -m k8s.k8s_adapter --mode live --deployment cloud-service-app --namespace default
```
- Real-time actuation of pod replicas based on PPO policy and LSTM predictions.
- Emits realistic Kubernetes events (`ScalingReplicaSet`, `SuccessfulCreate`).
- Built-in safety guards: min/max bounds, flapping cooldown, and uncertainty-aware reactive fallback.



## Swapping In a Real CSV Trace

The loader accepts any CSV with a `timestamp` column (parseable by pandas) and a
`cpu_util` column (float, 0–100 or 0–1).  Change **one line** in `config.yaml`:

```yaml
data:
  source: "csv"
  csv_path: "path/to/your/trace.csv"
```

No downstream code changes needed.

---

## Project Status

| Phase | Item | Status |
|-------|------|--------|
| 1–2 | Data pipeline + synthetic & real benchmark traces | ✅ Complete |
| 3–4 | LSTM + MC Dropout model (train, infer, evaluate) | ✅ Complete |
| 5 | Streamlit replay dashboard (Review I) | ✅ Complete |
| 6 | Gymnasium RL environment (`rl_env/`) | ✅ Complete |
| 7 | PPO RL agent — train + evaluate (`rl_agent/`) | ✅ Complete |
| **8** | **SHAP explainability** (`explainability/`) | ✅ **Complete** |
| **9** | **FastAPI REST backend** (`api/`) | ✅ **Complete** |
| **10** | **Multi-tab Streamlit dashboard** | ✅ **Complete** |
| **11** | **Kubernetes Adapter & Controller (`k8s/`)** | ✅ **Complete (Mock + Live kubectl)** |


---

## Configuration Reference

All tuneable parameters live in [`config.yaml`](config.yaml).  
Key fields:

| Key | Default | Description |
|-----|---------|-------------|
| `data.n_days` | 30 | Days of synthetic trace |
| `data.freq` | `1min` | Resampling interval |
| `preprocessing.window_size` | 60 | Look-back window (timesteps) |
| `preprocessing.horizon` | 15 | Forecast horizon (timesteps) |
| `model.dropout` | 0.3 | MC Dropout rate |
| `training.epochs` | 100 | Max training epochs |
| `inference.n_mc_samples` | 50 | Stochastic passes per prediction |
| `inference.ci_lower_pct` | 5 | Lower CI percentile |
| `inference.ci_upper_pct` | 95 | Upper CI percentile |
| `rl_env.min_replicas` | 1 | Minimum pod count |
| `rl_env.max_replicas` | 10 | Maximum pod count |
| `rl_env.initial_replicas` | 3 | Replicas at episode start |
| `rl_env.slo_threshold` | 0.70 | CPU-per-pod SLO threshold |
| `rl_env.scale_lag_steps` | 2 | Steps before scaling takes effect |
| `rl_env.reward_slo_met` | 1.0 | Reward for meeting SLO |
| `rl_env.reward_slo_violated` | −2.0 | Penalty for SLO violation |
| `rl_env.reward_cost_per_replica` | −0.05 | Per-replica cost penalty |
| `rl_env.reward_stability_penalty` | −0.3 | Penalty when replicas change |
| `ppo.rollout_steps` | 512 | Env steps per PPO rollout |
| `ppo.n_epochs` | 4 | Gradient update epochs per rollout |
| `ppo.gamma` | 0.99 | Discount factor |
| `ppo.gae_lambda` | 0.95 | GAE λ |
| `ppo.clip_eps` | 0.2 | PPO surrogate clip epsilon |
| `ppo.entropy_coef` | 0.01 | Entropy bonus coefficient |
| `ppo.total_timesteps` | 200000 | Total training timesteps |
| `ppo.learning_rate` | 0.0003 | AdamW learning rate |
