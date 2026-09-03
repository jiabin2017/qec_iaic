# BB72 CUDA-QX TN LER Sweep

This folder runs a full BB72 tensor-network logical error-rate sweep from the
cached 4095-mask path file.

Default run:

```text
6 p values x 5000 shots per p = 30000 shots
64 CUDA-QX workers
```

Each worker receives tasks by:

```text
global_task_id % NUM_WORKERS == worker_id
```

where `global_task_id` enumerates all `(p, shot)` pairs. This keeps the work
balanced across GPUs.

## Required Cache

The default cache path is:

```text
../tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl
```

It must contain all 4095 nonzero logical-mask paths before starting the sweep.

## Run On ASPIRE 2A

Submit from this directory:

```bash
cd /path/to/tn_decoder/ler_sweep
qsub submit_ler_sweep_64gpu.txt
```

The PBS log is written next to the submit file:

```text
output-bb72-tn-ler-sweep-64gpu.txt
```

Results are written under:

```text
results/sweep_YYYYMMDD_HHMMSS/
```

Important outputs:

```text
worker_000.csv ... worker_063.csv
all_shots.csv
ler_summary.csv
sweep_summary.json
sweep_summary.csv
```

`ler_summary.csv` is the file to plot:

```text
p, shots, failures, logical_error_rate, binomial_stderr
```

## Resume After Interruption

Each worker writes one row immediately after every completed shot to:

```text
worker_000_checkpoint.csv ... worker_063_checkpoint.csv
```

To resume, submit the job with the same output directory. Completed
`global_task_id` values are skipped and only unfinished shots are run:

```bash
qsub -v OUT_DIR=results/sweep_YYYYMMDD_HHMMSS submit_ler_sweep_64gpu.txt
```

Do not change the p-grid, shots per p, number of workers, base seed, error type,
or path cache when resuming an existing output directory.

## Small Test

Before a 64-GPU run, test with fewer workers and shots:

```bash
NUM_WORKERS=4 GPUS_PER_NODE=4 P_GRID=0.02 SHOTS_PER_P=8 bash run_ler_sweep_64gpu.sh
```

For a local single-GPU smoke test:

```bash
NUM_WORKERS=1 GPUS_PER_NODE=1 P_GRID=0.02 SHOTS_PER_P=1 bash run_ler_sweep_64gpu.sh
```
