#!/bin/bash
# Match interactive GPU settings from: alias gpush='srun -A co_rail -p savio4_gpu ...'
#SBATCH --job-name=resfit-train
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=60G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

REPO_ROOT="/global/home/users/ehharrison/residual-offpolicy-rl"
cd "${REPO_ROOT}"

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Start: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

export MUJOCO_GL=egl

# Scratch caches + Python env (see resfit/scripts/)
source resfit/scripts/scratch_env.sh
source resfit/scripts/activate_residual_env.sh

bash train_commands.sh

echo "End: $(date)"
