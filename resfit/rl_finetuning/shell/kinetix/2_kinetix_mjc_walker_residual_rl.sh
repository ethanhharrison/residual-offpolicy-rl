#!/bin/bash
# Residual TD3 on Kinetix mjc_walker with rtc-kinetix flow base policy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/scripts/activate_residual_env.sh"
source "$REPO_ROOT/scripts/scratch_env.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/kinetix/jax_cuda_env.sh"
bash "$REPO_ROOT/kinetix/setup_kinetix.sh"

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_kinetix_mjc_walker_config \
    base_policy.kinetix_checkpoint=https://storage.googleapis.com/rtc-assets/bc/24/policies/worlds_l_mjc_walker.pkl \
    algo.offline_fraction=0.0 \
    algo.learning_starts=2000 \
    algo.total_timesteps=60000 \
    eval_interval_every_steps=2000 \
    eval_interval_every_steps=5000 \
    algo.use_base_policy_for_warmup=true \
    algo.warmup_pure_base_policy=true \
    algo.warmup_min_success_episodes=1 \
    algo.stddev_min=0.002 \
    algo.stddev_max=0.002 \
    agent.actor.action_scale=0.3 \
    eval_num_episodes=50 \
    wandb.project=kinetix-mjc-walker-residual-td3 \
    wandb.name=residual-rl-kinetix-mjc-walker \
    wandb.group=residual-rl-kinetix \
    debug=false
