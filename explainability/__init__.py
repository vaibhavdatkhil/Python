"""
explainability/
────────────────
Phase 8: SHAP-based explainability for the LSTM forecaster and PPO agent.

Public functions
────────────────
  explain_lstm(model, X_te, cfg, output_dir) → saves shap_lstm.png
  explain_ppo(policy, obs_samples, cfg, output_dir) → saves shap_ppo.png
"""
