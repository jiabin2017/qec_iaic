# BB72 CUDA-QX TN LER Pilot

This is a small 4-GPU pilot for the cached BB72 tensor-network decoder. It is
designed to test whether four CUDA-QX workers can run concurrently on one ASPIRE
2A GPU node without interfering with each other.

The pilot runs:

```text
4 workers x 2 shots per worker = 8 total Monte Carlo shots
```

Each worker:

- binds to one GPU through `CUDA_VISIBLE_DEVICES`
- reads the same existing path cache
- samples two physical Z-error patterns at `p=0.02`
- computes the syndrome for each sampled error
- runs full 4096-sector TN reconstruction from 4095 cached flip-probability queries
- writes one compact worker CSV

The merge step checks all worker outputs and reports the pilot logical error
rate.

By default the worker suppresses repeated CUDA-QX decoder-construction logs such
as contractor setup and dtype messages. The shot-level progress, CSV files, JSON
summaries, and Python tracebacks are still preserved. To re-enable the raw
CUDA-QX logs for debugging, run with:

```bash
SUPPRESS_CUDAQX_LOGS=0 bash run_pilot_4gpu.sh
```

## Required Existing Cache

The pilot expects this cache file by default:

```text
../tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl
```

That file should contain 4095 cached paths. The pilot only reads it.
For transition, `run_pilot_4gpu.sh` will also fall back to the old pre-refactor
cache path if the new default cache file is missing and the old file still
exists.

## Run On ASPIRE 2A

```bash
cd /path/to/this-folder/ler_pilot
qsub submit_tn_decoder.txt
```

After completion:

```bash
cat output-bb72-tn-ler-pilot-4gpu.txt
ls -lh results
```

Success means:

- all 4 worker CSV files exist
- each worker used CUDA
- 8 total shots completed
- no duplicate global shot ids
- merge completed with `success=True`
