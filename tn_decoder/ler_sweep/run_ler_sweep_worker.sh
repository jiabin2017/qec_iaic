#!/bin/bash
set -euo pipefail

WORKER_ID="${1:?worker id is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID=$((WORKER_ID % GPUS_PER_NODE))
CUDAQX_LOG_ARGS=()
if [ "${SUPPRESS_CUDAQX_LOGS:-1}" = "0" ]; then
  CUDAQX_LOG_ARGS+=(--no-suppress-cudaqx-logs)
else
  CUDAQX_LOG_ARGS+=(--suppress-cudaqx-logs)
fi

echo "launch worker=${WORKER_ID} CUDA_VISIBLE_DEVICES=${GPU_ID}"
cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
PYTHON_BIN="${PYTHON_BIN:?Python path was not passed to the remote worker}"

exec "${PYTHON_BIN}" bb72_tn_ler_sweep_worker.py \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --p-grid "${P_GRID}" \
  --shots-per-p "${SHOTS_PER_P}" \
  --base-seed "${BASE_SEED}" \
  --device "${DEVICE}" \
  --error-type "${ERROR_TYPE}" \
  --path-cache "${PATH_CACHE}" \
  --output-dir "${OUT_DIR}" \
  "${CUDAQX_LOG_ARGS[@]}"
