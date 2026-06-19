#!/bin/bash
# Keep conda envs, package caches, and pip caches on Savio scratch (not home).
set -euo pipefail

if [ -d "/global/scratch/users/${USER}" ]; then
    SCRATCH_ROOT="/global/scratch/users/${USER}"
else
    SCRATCH_ROOT="${HOME}/scratch"
fi

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH_ROOT}/conda_pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${SCRATCH_ROOT}/conda_envs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip_cache}"

mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"
