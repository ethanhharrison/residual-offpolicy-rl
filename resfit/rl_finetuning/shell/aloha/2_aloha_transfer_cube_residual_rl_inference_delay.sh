#!/bin/bash
# Residual TD3 on Aloha Sim transfer cube with simulated base-policy inference delay.

export MUJOCO_GL=egl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/scripts/activate_residual_env.sh"
source "$REPO_ROOT/scripts/scratch_env.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/kinetix/jax_cuda_env.sh"

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_aloha_transfer_cube_config \
    camera_size=64 \
    algo.prefetch_batches=4 \
    algo.offline_fraction=0.0 \
    algo.learning_starts=10000 \
    algo.total_timesteps=1000000 \
    algo.use_base_policy_for_warmup=true \
    algo.warmup_pure_base_policy=true \
    algo.warmup_min_success_episodes=1 \
    algo.stddev_min=0.002 \
    algo.stddev_max=0.002 \
    agent.enc_type=conv \
    agent.actor.hidden_dim=128 \
    agent.actor.num_layers=2 \
    agent.critic.hidden_dim=128 \
    agent.critic.num_layers=2 \
    agent.actor.action_scale=0.03 \
    base_policy.inference_delay=20 \
    base_policy.n_action_steps=25 \
    eval_num_episodes=50 \
    wandb.project=aloha-transfer-cube-residual-td3 \
    wandb.name=residual-rl-aloha-as0.1-updated-delay20-n25 \
    wandb.notes="aloha/transfer_cube residual TD3 with inference delay 20 and n_action_steps 25" \
    wandb.group=residual-rl \
    debug=false
