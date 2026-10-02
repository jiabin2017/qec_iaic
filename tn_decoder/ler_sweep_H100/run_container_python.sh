#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${LER_CONTAINER_NAME:-cudaqx_ler}"
ENV_FILE="${LER_ENV_FILE:-/home/users/astar/ihpc/youjb/cudaqx_env.sh}"

command -v enroot >/dev/null || { echo "enroot is unavailable on this execution node" >&2; exit 2; }

ENV_ARGS=()
for KEY in CUDA_VISIBLE_DEVICES OMP_NUM_THREADS PYTHONNOUSERSITE P_GRID SHOTS_PER_P BASE_SEED ERROR_TYPE DEVICE LOGICAL_BASIS_SOURCE SUPPRESS_CUDAQX_LOGS TASK_START TASK_STOP; do
  if [[ -v "${KEY}" ]]; then ENV_ARGS+=(--env "${KEY}=${!KEY}"); fi
done

printf -v SCRIPT_DIR_Q "%q" "${SCRIPT_DIR}"
printf -v ENV_FILE_Q "%q" "${ENV_FILE}"
ARGS_Q=()
for ARG in "$@"; do
  printf -v ARG_Q "%q" "${ARG}"
  ARGS_Q+=("${ARG_Q}")
done
ARGS_JOINED="${ARGS_Q[*]}"

read -r -d "" COMMAND <<EOF || true
set -eo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE="\${PYTHONNOUSERSITE:-1}"
SCRIPT_DIR=${SCRIPT_DIR_Q}
ENV_FILE=${ENV_FILE_Q}
if [ -r "\${ENV_FILE}" ]; then
  source "\${ENV_FILE}" >&2
fi
cd "\${SCRIPT_DIR}"
set -- ${ARGS_JOINED}
if [ "\${1:-}" = "--check" ]; then
  shift
  python - "\$@" <<'PY'
import json
import platform
import sys
from pathlib import Path

import torch

from bb72_tn_ler_sweep_worker import build_expected_cache, load_path_cache

cache_path = Path(sys.argv[1])
expected_gpus = int(sys.argv[2])
if not torch.cuda.is_available() or torch.cuda.device_count() < expected_gpus:
    raise RuntimeError(
        f"Expected at least {expected_gpus} visible CUDA GPU(s), "
        f"got {torch.cuda.device_count()}"
    )
import os

context = build_expected_cache(
    error_type=os.environ.get("ERROR_TYPE", "Z"),
    p=float(os.environ.get("P_GRID", "0.005").split(",")[0]),
    device=os.environ.get("DEVICE", "cuda"),
    logical_basis_source_arg=os.environ.get("LOGICAL_BASIS_SOURCE", "auto"),
    suppress_cudaqx_logs=True,
)
cache, status = load_path_cache(cache_path, context[5])
if status["state"] != "loaded" or len(cache["paths"]) != 4095:
    raise RuntimeError(f"Incomplete or incompatible path cache: {status}")
if context[6]["actual_device"] != "cuda" or context[6]["contractor"] != "cutensornet":
    raise RuntimeError(f"Unexpected TN backend: {context[6]}")
print(json.dumps({
    "preflight": "PASS",
    "host": platform.node(),
    "python": sys.executable,
    "visible_gpus": torch.cuda.device_count(),
    "gpu0": torch.cuda.get_device_name(0),
}))
PY
else
  exec python "\$@"
fi
EOF

exec enroot start "${ENV_ARGS[@]}" "${CONTAINER_NAME}" -c "${COMMAND}"
