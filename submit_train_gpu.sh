#!/bin/bash
# Submit train_commands.sh to a Savio GPU node (non-interactive).
#
# Usage:
#   bash submit_train_gpu.sh
#   bash submit_train_gpu.sh --time=10-00:00:00 --partition savio4_gpu
#
# Monitor:
#   squeue -u "$USER"
#   tail -f logs/train_<jobid>.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${REPO_ROOT}/train_gpu.sbatch"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

JOB_ID="$(sbatch "$@" "${SBATCH_SCRIPT}" | awk '{print $NF}')"
echo "Submitted job ${JOB_ID}"
echo "  stdout: ${LOG_DIR}/train_${JOB_ID}.out"
echo "  stderr: ${LOG_DIR}/train_${JOB_ID}.err"
echo ""
echo "Monitor with:  squeue -j ${JOB_ID}"
echo "Follow logs:   tail -f ${LOG_DIR}/train_${JOB_ID}.out"
