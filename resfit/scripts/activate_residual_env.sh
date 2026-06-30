#!/bin/bash
# Activate the scratch-based Python 3.11 env for residual RL (OpenPI + Aloha).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/conda_scratch.sh"

ENV_NAME="${RESFIT_ENV_NAME:-resfit}"
ENV_PATH="${CONDA_ENVS_PATH}/${ENV_NAME}"

if [ ! -x "${ENV_PATH}/bin/python" ]; then
    echo "Scratch env not found at ${ENV_PATH}." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${ENV_PATH}/bin/activate"

export PYTHONNOUSERSITE=1
export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

# OpenPI (Aloha pi0) and Kinetix flow base policies use JAX with cuda12 wheels.
# Without CUDA_ROOT, jax import can fail with a partially initialized module error.
JAX_ENV="${SCRIPT_DIR}/../kinetix/jax_cuda_env.sh"
if [ -f "${JAX_ENV}" ]; then
    # shellcheck disable=SC1091
    source "${JAX_ENV}"
fi

# uv-managed Python on HPC often lacks system CA certs; gcsfs/HTTPS need certifi.
if python -c "import certifi" >/dev/null 2>&1; then
    export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
    export REQUESTS_CA_BUNDLE="${SSL_CERT_FILE}"
fi

echo "Using Python: $(which python) ($(python --version))"
echo "Env path: ${ENV_PATH}"
