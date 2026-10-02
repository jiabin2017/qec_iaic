# BB72 CUDA-QX TN LER Sweep, H100

This folder runs BB72 tensor-network logical-error-rate sweeps from the cached
4095-mask path file.

Default production layout:

```text
6 p values x 5000 shots per p = 30000 shots
12 PBS array batches
7 H100 GPUs per batch
```

Each batch runs only one contiguous `global_task_id` range. The default split is:

```text
batch 000: tasks     0..2499
batch 001: tasks  2500..4999
...
batch 011: tasks 27500..29999
```

Within one batch, workers are assigned by:

```text
global_task_id % NUM_WORKERS == worker_id
```

With `NUM_WORKERS=7`, this keeps the work balanced across the 7 GPUs requested
from `aiq1`.

## Required Cache

The default cache path is:

```text
../tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl
```

It must contain all 4095 nonzero logical-mask paths before starting the sweep.

## Production Run On ASPIRE 2A+

Submit from this directory:

```bash
cd /home/users/astar/ihpc/youjb/tn_decoder/ler_sweep_H100
qsub submit_ler_sweep_7gpu_array.txt
```

The submit file requests:

```text
12 array jobs
1 node x 7 H100 GPUs per array job
24 hour walltime
container image: /home/users/astar/ihpc/youjb/containers/cudaqx-qec-0.7.0-tn-cu13.sqsh
```

This is intended to route as small 7-GPU batch jobs rather than one 64-GPU job.

Outputs are written under one stable sweep directory:

```text
results/sweep_h100_7gpu_JOBID/
  batches/batch_000/
  batches/batch_001/
  ...
  batches/batch_011/
```

Each batch directory contains its own:

```text
worker_000.csv ... worker_006.csv
all_shots.csv
ler_summary.csv
sweep_summary.json
```

## Final Merge

After all array jobs finish, run:

```bash
cd /home/users/astar/ihpc/youjb/tn_decoder/ler_sweep_H100

python3 merge_ler_sweep_batches.py \
  --sweep-dir results/sweep_h100_7gpu_JOBID \
  --p-grid 0.005,0.01,0.02,0.03,0.04,0.05 \
  --shots-per-p 5000 \
  --expected-batches 12
```

Final outputs are written at the sweep-directory root:

```text
results/sweep_h100_7gpu_JOBID/all_shots.csv
results/sweep_h100_7gpu_JOBID/ler_summary.csv
results/sweep_h100_7gpu_JOBID/sweep_summary.json
```

`ler_summary.csv` is the file to plot:

```text
p, shots, failures, logical_error_rate, binomial_stderr
```

## Resume

To resume the same sweep, submit with the same `SWEEP_DIR`:

```bash
qsub -v SWEEP_DIR=results/sweep_h100_7gpu_JOBID submit_ler_sweep_7gpu_array.txt
```

Completed `global_task_id` values are skipped by worker checkpoint files. Do not
change `P_GRID`, `SHOTS_PER_P`, `NUM_BATCHES`, `NUM_WORKERS`, `BASE_SEED`,
`ERROR_TYPE`, or `PATH_CACHE` while resuming a sweep.

To rerun one failed batch only:

```bash
qsub -J 5-5 \
  -v SWEEP_DIR=results/sweep_h100_7gpu_JOBID \
  submit_ler_sweep_7gpu_array.txt
```

## Small Interactive Test

On an allocated 2-GPU H100 interactive node:

```bash
cd /home/users/astar/ihpc/youjb/tn_decoder/ler_sweep_H100

enroot start \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS=1 \
  cudaqx_tn_verify -c '
set -euo pipefail
unset PYTHONPATH
cd /home/users/astar/ihpc/youjb/tn_decoder/ler_sweep_H100
export PYTHON_BIN=python
export GPUS_PER_NODE=2
export NUM_WORKERS=2
export P_GRID=0.005,0.01
export SHOTS_PER_P=1
export OUT_DIR=results/sweep_h100_interactive_2gpu_test
export SUPPRESS_CUDAQX_LOGS=1
env -u PBS_JOBID -u PBS_NODEFILE bash run_ler_sweep.sh
'
```
