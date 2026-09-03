#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN_DECODER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

NUM_WORKERS="${NUM_WORKERS:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
P_GRID="${P_GRID:-0.005,0.01,0.02,0.03,0.04,0.05}"
SHOTS_PER_P="${SHOTS_PER_P:-5000}"
BASE_SEED="${BASE_SEED:-20260902}"
ERROR_TYPE="${ERROR_TYPE:-Z}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/sweep_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_PATH_CACHE="${TN_DECODER_ROOT}/tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl"
PATH_CACHE="${PATH_CACHE:-${DEFAULT_PATH_CACHE}}"
SUPPRESS_CUDAQX_LOGS="${SUPPRESS_CUDAQX_LOGS:-1}"

mkdir -p "${OUT_DIR}"

echo "working_dir=${SCRIPT_DIR}"
echo "tn_decoder_root=${TN_DECODER_ROOT}"
echo "output_dir=${OUT_DIR}"
echo "path_cache=${PATH_CACHE}"
echo "num_workers=${NUM_WORKERS}"
echo "gpus_per_node=${GPUS_PER_NODE}"
echo "p_grid=${P_GRID}"
echo "shots_per_p=${SHOTS_PER_P}"
echo "base_seed=${BASE_SEED}"
echo "error_type=${ERROR_TYPE}"
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

export NUM_WORKERS GPUS_PER_NODE P_GRID SHOTS_PER_P BASE_SEED ERROR_TYPE DEVICE PATH_CACHE OUT_DIR
export SUPPRESS_CUDAQX_LOGS

PYTHON_BIN="$(command -v python)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Cannot locate Python after conda activation" >&2
  exit 3
fi
echo "python_bin=${PYTHON_BIN}"

PIDS=()
if command -v pbsdsh >/dev/null 2>&1 && [ "${USE_PBSDSH:-1}" = "1" ]; then
  echo "launcher=pbsdsh"
  for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    pbsdsh -n "${WORKER_ID}" -- env \
      "NUM_WORKERS=${NUM_WORKERS}" \
      "GPUS_PER_NODE=${GPUS_PER_NODE}" \
      "P_GRID=${P_GRID}" \
      "SHOTS_PER_P=${SHOTS_PER_P}" \
      "BASE_SEED=${BASE_SEED}" \
      "ERROR_TYPE=${ERROR_TYPE}" \
      "DEVICE=${DEVICE}" \
      "PATH_CACHE=${PATH_CACHE}" \
      "OUT_DIR=${OUT_DIR}" \
      "PYTHON_BIN=${PYTHON_BIN}" \
      "SUPPRESS_CUDAQX_LOGS=${SUPPRESS_CUDAQX_LOGS}" \
      bash "${SCRIPT_DIR}/run_ler_sweep_worker.sh" "${WORKER_ID}" &
    PIDS+=("$!")
  done
else
  echo "launcher=local_background"
  for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    bash "${SCRIPT_DIR}/run_ler_sweep_worker.sh" "${WORKER_ID}" &
    PIDS+=("$!")
  done
fi

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

python merge_bb72_tn_ler_sweep.py \
  --output-dir "${OUT_DIR}" \
  --expected-workers "${NUM_WORKERS}" \
  --p-grid "${P_GRID}" \
  --shots-per-p "${SHOTS_PER_P}"
