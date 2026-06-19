#!/bin/bash
# Get the directory of this script and navigate to repo root
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"
LEROBOT_COMMIT="69901b9b6a2300914ca3de0ea14b6fa6e0203bd4"

# Fix pip script installs when ~/.local/lib is symlinked to scratch storage.
source "$REPO_ROOT/resfit/scripts/ensure_pip_dirs.sh"

# Create deps directory if it doesn't exist
mkdir -p "$DEPS_DIR"

# Clone or update lerobot
if [ -d "$DEPS_DIR/lerobot/.git" ]; then
    echo "LeRobot repo already exists, checking out pinned commit..."
    git -C "$DEPS_DIR/lerobot" fetch --depth 1 origin "$LEROBOT_COMMIT" 2>/dev/null || true
    git -C "$DEPS_DIR/lerobot" checkout "$LEROBOT_COMMIT"
else
    git clone https://github.com/huggingface/lerobot.git "$DEPS_DIR/lerobot"
    git -C "$DEPS_DIR/lerobot" checkout "$LEROBOT_COMMIT"
fi

# Install lerobot
python -m pip install -e "$DEPS_DIR/lerobot" --no-deps

# Install pinned dependencies (datasets version must match training code)
python -m pip install -r "$REPO_ROOT/resfit/lerobot/lerobot_requirements.txt"

# Only upgrade torch packages if they are not already importable
if ! python -c "import torch, torchvision, torchcodec" 2>/dev/null; then
    python -m pip install --upgrade torch torchvision torchcodec
fi
