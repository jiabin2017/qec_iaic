#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN_DECODER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

NUM_WORKERS="${NUM_WORKERS:-4}"
SHOTS_PER_WORKER="${SHOTS_PER_WORKER:-2}"
P_VALUE="${P_VALUE:-0.02}"
BASE_SEED="${BASE_SEED:-20260901}"
ERROR_TYPE="${ERROR_TYPE:-Z}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/run_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_PATH_CACHE="${TN_DECODER_ROOT}/tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl"
LEGACY_PATH_CACHE_IN_PLACE="${TN_DECODER_ROOT}/tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_reconstruct_cached_Z_weight6-x_sample_path_cache.pkl"
LEGACY_PATH_CACHE_OLD_RESULTS="${TN_DECODER_ROOT}/results/bb72_cudaqx_nv_tn_reconstruct_cached/bb72_cudaqx_nv_tn_reconstruct_cached_Z_weight6-x_sample_path_cache.pkl"
PATH_CACHE="${PATH_CACHE:-${DEFAULT_PATH_CACHE}}"
SUPPRESS_CUDAQX_LOGS="${SUPPRESS_CUDAQX_LOGS:-1}"

if [ ! -f "${PATH_CACHE}" ] && [ "${PATH_CACHE}" = "${DEFAULT_PATH_CACHE}" ] && [ -f "${LEGACY_PATH_CACHE_IN_PLACE}" ]; then
  echo "using_legacy_path_cache=${LEGACY_PATH_CACHE_IN_PLACE}"
  PATH_CACHE="${LEGACY_PATH_CACHE_IN_PLACE}"
elif [ ! -f "${PATH_CACHE}" ] && [ "${PATH_CACHE}" = "${DEFAULT_PATH_CACHE}" ] && [ -f "${LEGACY_PATH_CACHE_OLD_RESULTS}" ]; then
  echo "using_legacy_path_cache=${LEGACY_PATH_CACHE_OLD_RESULTS}"
  PATH_CACHE="${LEGACY_PATH_CACHE_OLD_RESULTS}"
fi

CUDAQX_LOG_ARGS=()
if [ "${SUPPRESS_CUDAQX_LOGS}" = "0" ]; then
  CUDAQX_LOG_ARGS+=(--no-suppress-cudaqx-logs)
else
  CUDAQX_LOG_ARGS+=(--suppress-cudaqx-logs)
fi

mkdir -p "${OUT_DIR}"

echo "working_dir=${SCRIPT_DIR}"
echo "tn_decoder_root=${TN_DECODER_ROOT}"
echo "output_dir=${OUT_DIR}"
echo "path_cache=${PATH_CACHE}"
echo "num_workers=${NUM_WORKERS}"
echo "shots_per_worker=${SHOTS_PER_WORKER}"
echo "p=${P_VALUE}"
echo "base_seed=${BASE_SEED}"
echo "suppress_cudaqx_logs=${SUPPRESS_CUDAQX_LOGS}"
echo "PBS_JOBID=${PBS_JOBID:-<unset>}"
echo "PBS_NODEFILE=${PBS_NODEFILE:-<unset>}"

if [ ! -f "${PATH_CACHE}" ]; then
  echo "missing path cache: ${PATH_CACHE}" >&2
  exit 2
fi

if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
  echo "allocated_nodes:"
  sort -u "${PBS_NODEFILE}"
fi

PIDS=()
for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
  GPU_ID="${WORKER_ID}"
  echo "launch worker=${WORKER_ID} CUDA_VISIBLE_DEVICES=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python bb72_tn_ler_pilot_worker.py \
    --worker-id "${WORKER_ID}" \
    --num-workers "${NUM_WORKERS}" \
    --shots-per-worker "${SHOTS_PER_WORKER}" \
    --p "${P_VALUE}" \
    --base-seed "${BASE_SEED}" \
    --device "${DEVICE}" \
    --error-type "${ERROR_TYPE}" \
    --path-cache "${PATH_CACHE}" \
    --output-dir "${OUT_DIR}" \
    "${CUDAQX_LOG_ARGS[@]}" &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

if [ "${FAILED}" -ne 0 ]; then
  echo "at least one worker failed" >&2
  exit 1
fi

python merge_bb72_tn_ler_pilot.py \
  --output-dir "${OUT_DIR}" \
  --expected-workers "${NUM_WORKERS}" \
  --shots-per-worker "${SHOTS_PER_WORKER}"
