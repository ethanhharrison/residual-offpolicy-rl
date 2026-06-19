#!/bin/bash
# Use Savio scratch for large caches (checkpoints, HF downloads, replay buffers).
# Source this before setup or training:
#   source resfit/scripts/scratch_env.sh
#
# Override the root with:
#   export RESFIT_SCRATCH_ROOT=/path/to/your/scratch/cache

if [ -z "${RESFIT_SCRATCH_ROOT:-}" ]; then
    if [ -d "/global/scratch/users/${USER}" ]; then
        RESFIT_SCRATCH_ROOT="/global/scratch/users/${USER}/resfit_cache"
    else
        # Fallback for local/non-HPC machines.
        RESFIT_SCRATCH_ROOT="${HOME}/resfit_cache"
    fi
fi

export RESFIT_SCRATCH_ROOT
export CACHE_DIR="${CACHE_DIR:-${RESFIT_SCRATCH_ROOT}}"
# Keep HF auth in ~/.cache/huggingface (where `huggingface-cli login` stores the token).
# Only redirect large model caches to scratch.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${RESFIT_SCRATCH_ROOT}/huggingface/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${RESFIT_SCRATCH_ROOT}/huggingface/transformers}"
export BASE_POLICY_PATH="${BASE_POLICY_PATH:-${RESFIT_SCRATCH_ROOT}/base_policy}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${RESFIT_SCRATCH_ROOT}/openpi}"

mkdir -p \
    "${CACHE_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${BASE_POLICY_PATH}" \
    "${OPENPI_DATA_HOME}"

echo "Using scratch cache root: ${RESFIT_SCRATCH_ROOT}"
echo "  CACHE_DIR=${CACHE_DIR}"
echo "  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "  BASE_POLICY_PATH=${BASE_POLICY_PATH}"
echo "  OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
