#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

P_GRID="${P_GRID:-0.005,0.01,0.02,0.03,0.04,0.05}"
SHOTS_PER_P="${SHOTS_PER_P:-5000}"
NUM_BATCHES="${NUM_BATCHES:-12}"
BATCH_INDEX="${BATCH_INDEX:-${PBS_ARRAY_INDEX:?Set BATCH_INDEX or submit as a PBS array job}}"

IFS=',' read -r -a P_VALUES <<< "${P_GRID}"
TOTAL_TASKS=$((${#P_VALUES[@]} * SHOTS_PER_P))
TASKS_PER_BATCH="${TASKS_PER_BATCH:-$(((TOTAL_TASKS + NUM_BATCHES - 1) / NUM_BATCHES))}"

TASK_START=$((BATCH_INDEX * TASKS_PER_BATCH))
TASK_STOP=$((TASK_START + TASKS_PER_BATCH))
if (( TASK_START >= TOTAL_TASKS )); then
  echo "batch_index=${BATCH_INDEX} starts past total_tasks=${TOTAL_TASKS}" >&2
  exit 2
fi
if (( TASK_STOP > TOTAL_TASKS )); then
  TASK_STOP="${TOTAL_TASKS}"
fi

JOB_STEM="$(printf '%s' "${PBS_JOBID:-manual}" | sed 's/\..*$//; s/\[.*$//; s/[^A-Za-z0-9_]/_/g')"
SWEEP_DIR="${SWEEP_DIR:-results/sweep_h100_7gpu_${JOB_STEM}}"
BATCH_DIR="${SWEEP_DIR}/batches/batch_$(printf '%03d' "${BATCH_INDEX}")"

export GPUS_PER_NODE="${GPUS_PER_NODE:-7}"
export NUM_WORKERS="${NUM_WORKERS:-7}"
export P_GRID SHOTS_PER_P TASK_START TASK_STOP
export OUT_DIR="${BATCH_DIR}"

echo "batch_index=${BATCH_INDEX}"
echo "num_batches=${NUM_BATCHES}"
echo "total_tasks=${TOTAL_TASKS}"
echo "tasks_per_batch=${TASKS_PER_BATCH}"
echo "task_range=[${TASK_START},${TASK_STOP})"
echo "sweep_dir=${SWEEP_DIR}"
echo "batch_dir=${BATCH_DIR}"

exec bash run_ler_sweep.sh
