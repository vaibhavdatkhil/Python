"""
rl_agent/
──────────
Phase 7: PPO (Proximal Policy Optimization) agent for K8s autoscaling.

Modules
───────
  policy_net  : Actor-Critic MLP (shared backbone + separate heads)
  ppo         : Rollout buffer, GAE, PPO update loop
  train_rl    : CLI training entry-point
  evaluate_rl : Evaluation + plot generation
"""
