#!/bin/bash
# Setup Aloha Sim dependencies for residual TD3 experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Fix pip script installs when ~/.local/lib is symlinked to scratch storage.
source "$REPO_ROOT/resfit/scripts/ensure_pip_dirs.sh"
# Put large artifacts (HF cache, replay buffers) on scratch.
source "$REPO_ROOT/resfit/scripts/scratch_env.sh"
echo "==> Creating/updating scratch conda env (Python 3.11 + OpenPI)..."
bash "$REPO_ROOT/resfit/scripts/create_scratch_env.sh"

echo "Aloha Sim setup complete."
echo "Activate and train with:"
echo "  source resfit/scripts/activate_residual_env.sh"
echo "  bash resfit/rl_finetuning/shell/aloha/1_aloha_transfer_cube_residual_rl.sh"
