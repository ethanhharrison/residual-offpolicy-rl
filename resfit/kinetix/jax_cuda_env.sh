#!/bin/bash
# Export CUDA_ROOT and JAX GPU settings for Kinetix + flow base policy.

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Leave headroom on the GPU for PyTorch RL networks alongside JAX env + base policy.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"

if [ -z "${CUDA_ROOT:-}" ]; then
  _CUDA_NVCC_DIR="$(python - <<'PY'
import site
from pathlib import Path

for site_dir in site.getsitepackages():
    candidate = Path(site_dir) / "nvidia" / "cuda_nvcc"
    if candidate.is_dir():
        print(candidate)
        break
PY
)"
  if [ -n "${_CUDA_NVCC_DIR}" ]; then
    export CUDA_ROOT="${_CUDA_NVCC_DIR}"
  fi
  unset _CUDA_NVCC_DIR
fi

# Do not force CPU JAX when this file is sourced.
unset JAX_PLATFORMS
