#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN_DECODER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-7}"
P_GRID="${P_GRID:-0.005,0.01,0.02,0.03,0.04,0.05}"
SHOTS_PER_P="${SHOTS_PER_P:-5000}"
BASE_SEED="${BASE_SEED:-20260902}"
ERROR_TYPE="${ERROR_TYPE:-Z}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/sweep_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_PATH_CACHE="${TN_DECODER_ROOT}/tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl"
PATH_CACHE="${PATH_CACHE:-${DEFAULT_PATH_CACHE}}"
SUPPRESS_CUDAQX_LOGS="${SUPPRESS_CUDAQX_LOGS:-1}"
TASK_START="${TASK_START:-0}"
TASK_STOP="${TASK_STOP:-}"

USE_PBS=0
HOSTS=()
if [[ -n "${PBS_JOBID:-}" ]]; then
  command -v pbsdsh >/dev/null || { echo "PBS job requires pbsdsh" >&2; exit 2; }
  [[ -f "${PBS_NODEFILE:-}" ]] || { echo "Missing PBS_NODEFILE" >&2; exit 2; }
  mapfile -t HOSTS < "${PBS_NODEFILE}"
  CAPACITY="${#HOSTS[@]}"
  USE_PBS=1
else
  CAPACITY="${GPUS_PER_NODE}"
fi

NUM_WORKERS="${NUM_WORKERS:-${CAPACITY}}"
[[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid worker or GPU count" >&2
  exit 2
}
(( NUM_WORKERS <= CAPACITY )) || { echo "NUM_WORKERS exceeds allocated PBS slots" >&2; exit 2; }

GPU_IDS=()
declare -A HOST_COUNTS=()
for (( SLOT=0; SLOT<NUM_WORKERS; SLOT++ )); do
  if (( USE_PBS )); then
    HOST="${HOSTS[$SLOT]}"
    [[ -n "${HOST}" ]] || { echo "Empty PBS_NODEFILE host" >&2; exit 2; }
    LOCAL_INDEX="${HOST_COUNTS[$HOST]:-0}"
    (( LOCAL_INDEX < GPUS_PER_NODE )) || { echo "More PBS ranks than GPUs on a host" >&2; exit 2; }
    HOST_COUNTS[$HOST]=$((LOCAL_INDEX+1))
  else
    LOCAL_INDEX="${SLOT}"
  fi
  GPU_IDS+=("${LOCAL_INDEX}")
done

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
echo "task_start=${TASK_START}"
echo "task_stop=${TASK_STOP:-<auto>}"
echo "suppress_cudaqx_logs=${SUPPRESS_CUDAQX_LOGS}"
echo "PBS_JOBID=${PBS_JOBID:-<unset>}"
echo "PBS_NODEFILE=${PBS_NODEFILE:-<unset>}"

if [ ! -f "${PATH_CACHE}" ]; then
  echo "missing path cache: ${PATH_CACHE}" >&2
  exit 2
fi

if (( USE_PBS )); then
  echo "allocated_nodes:"
  sort -u "${PBS_NODEFILE}"
fi

export NUM_WORKERS GPUS_PER_NODE P_GRID SHOTS_PER_P BASE_SEED ERROR_TYPE DEVICE PATH_CACHE OUT_DIR
export SUPPRESS_CUDAQX_LOGS TASK_START TASK_STOP

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Cannot locate Python" >&2
  exit 3
fi
echo "python_bin=${PYTHON_BIN}"

REMOTE_ENV=(env
  "NUM_WORKERS=${NUM_WORKERS}"
  "GPUS_PER_NODE=${GPUS_PER_NODE}"
  "P_GRID=${P_GRID}"
  "SHOTS_PER_P=${SHOTS_PER_P}"
  "BASE_SEED=${BASE_SEED}"
  "ERROR_TYPE=${ERROR_TYPE}"
  "DEVICE=${DEVICE}"
  "PATH_CACHE=${PATH_CACHE}"
  "OUT_DIR=${OUT_DIR}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "SUPPRESS_CUDAQX_LOGS=${SUPPRESS_CUDAQX_LOGS}"
  "TASK_START=${TASK_START}"
  "TASK_STOP=${TASK_STOP}"
)
for KEY in LER_CONTAINER_NAME LER_ENV_FILE LOGICAL_BASIS_SOURCE PYTHONNOUSERSITE; do
  if [[ -v "${KEY}" ]]; then REMOTE_ENV+=("${KEY}=${!KEY}"); fi
done

if [[ "${PYTHON_BIN}" == "${SCRIPT_DIR}/run_container_python.sh" ]]; then
  if (( USE_PBS )); then
    declare -A CHECKED_HOSTS=()
    for (( SLOT=0; SLOT<NUM_WORKERS; SLOT++ )); do
      HOST="${HOSTS[$SLOT]}"
      [[ -z "${CHECKED_HOSTS[$HOST]:-}" ]] || continue
      CHECKED_HOSTS[$HOST]=1
      echo "Checking container, CUDA, decoder and cache on ${HOST}"
      pbsdsh -n "${SLOT}" -- "${REMOTE_ENV[@]}" "${PYTHON_BIN}" --check "${PATH_CACHE}" "${HOST_COUNTS[$HOST]}"
    done
  else
    "${PYTHON_BIN}" --check "${PATH_CACHE}" "${NUM_WORKERS}"
  fi
fi

mkdir -p "${OUT_DIR}"

PIDS=()
if (( USE_PBS )) && [ "${USE_PBSDSH:-1}" = "1" ]; then
  echo "launcher=pbsdsh"
  for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    pbsdsh -n "${WORKER_ID}" -- "${REMOTE_ENV[@]}" \
      bash "${SCRIPT_DIR}/run_ler_sweep_worker.sh" "${WORKER_ID}" "${GPU_IDS[$WORKER_ID]}" &
    PIDS+=("$!")
  done
else
  echo "launcher=local_background"
  for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    bash "${SCRIPT_DIR}/run_ler_sweep_worker.sh" "${WORKER_ID}" "${GPU_IDS[$WORKER_ID]}" &
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

MERGE_ARGS=(
  --output-dir "${OUT_DIR}"
  --expected-workers "${NUM_WORKERS}"
  --p-grid "${P_GRID}"
  --shots-per-p "${SHOTS_PER_P}"
  --task-start "${TASK_START}"
)
if [[ -n "${TASK_STOP}" ]]; then
  MERGE_ARGS+=(--task-stop "${TASK_STOP}")
fi

"${PYTHON_BIN}" merge_bb72_tn_ler_sweep.py "${MERGE_ARGS[@]}"
