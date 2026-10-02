#!/bin/bash
set -euo pipefail

WORKER_ID="${1:?worker id is required}"
LOCAL_GPU_INDEX="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${LOCAL_GPU_INDEX}" ]]; then
  LOCAL_GPU_INDEX=$((WORKER_ID % GPUS_PER_NODE))
fi
CUDAQX_LOG_ARGS=()
if [ "${SUPPRESS_CUDAQX_LOGS:-1}" = "0" ]; then
  CUDAQX_LOG_ARGS+=(--no-suppress-cudaqx-logs)
else
  CUDAQX_LOG_ARGS+=(--suppress-cudaqx-logs)
fi
TASK_RANGE_ARGS=(--task-start "${TASK_START:-0}")
if [[ -n "${TASK_STOP:-}" ]]; then
  TASK_RANGE_ARGS+=(--task-stop "${TASK_STOP}")
fi
LOGICAL_BASIS_ARGS=()
if [[ -n "${LOGICAL_BASIS_SOURCE:-}" ]]; then
  LOGICAL_BASIS_ARGS+=(--logical-basis-source "${LOGICAL_BASIS_SOURCE}")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  if (( LOCAL_GPU_INDEX >= ${#VISIBLE_GPUS[@]} )); then
    echo "GPU index ${LOCAL_GPU_INDEX} outside scheduler mask ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
  GPU_ID="${VISIBLE_GPUS[$LOCAL_GPU_INDEX]}"
else
  GPU_ID="${LOCAL_GPU_INDEX}"
fi

echo "launch worker=${WORKER_ID} CUDA_VISIBLE_DEVICES=${GPU_ID}"
cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:?Python path was not passed to the remote worker}"

exec "${PYTHON_BIN}" -u bb72_tn_ler_sweep_worker.py \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --p-grid "${P_GRID}" \
  --shots-per-p "${SHOTS_PER_P}" \
  --base-seed "${BASE_SEED}" \
  --device "${DEVICE}" \
  --error-type "${ERROR_TYPE}" \
  "${LOGICAL_BASIS_ARGS[@]}" \
  "${TASK_RANGE_ARGS[@]}" \
  --path-cache "${PATH_CACHE}" \
  --output-dir "${OUT_DIR}" \
  "${CUDAQX_LOG_ARGS[@]}"
