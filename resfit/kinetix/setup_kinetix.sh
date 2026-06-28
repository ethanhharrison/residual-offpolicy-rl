#!/bin/bash
# Install Kinetix + rtc-kinetix dependencies for residual RL experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RTC_ROOT="${REPO_ROOT}/third_party/real-time-chunking-kinetix"
KINETIX_ROOT="${RTC_ROOT}/third_party/kinetix"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/jax_cuda_env.sh"

echo "Installing Kinetix dependencies into the active Python environment..."

# rtc-kinetix checkpoints require flax 0.10.2 state layout (0.10.4 breaks LayerNorm loading).
pip install "jax[cuda12]==0.4.35" "flax==0.10.2" einops tyro

if ! python -c "import kinetix.environment.env" 2>/dev/null; then
  mkdir -p "${REPO_ROOT}/third_party"
  if [ ! -d "${KINETIX_ROOT}" ]; then
    echo "Cloning real-time-chunking-kinetix into ${RTC_ROOT}..."
    git clone --depth 1 https://github.com/Physical-Intelligence/real-time-chunking-kinetix.git "${RTC_ROOT}"
    git -C "${RTC_ROOT}" submodule update --init third_party/kinetix
  fi
  pip install -e "${KINETIX_ROOT}"
fi

python - <<'PY'
from resfit.kinetix.utils.deps import configure_jax_gpu
import flax
import jax
import kinetix.environment.env  # noqa: F401

configure_jax_gpu()
print("Flax version:", flax.__version__)
print("CUDA_ROOT:", __import__("os").environ.get("CUDA_ROOT"))
print("JAX devices:", jax.devices())
print("Kinetix import OK")
PY

echo "Kinetix setup complete."
echo "Pretrained BC checkpoints: https://storage.googleapis.com/rtc-assets/bc/24/policies/"
