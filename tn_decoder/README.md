# Self-Contained BB72 TN Decoder Workspace

This folder is organized so it can be copied to the Linux cluster as one unit.
Run PBS submit scripts from their own subdirectories.

## Layout

```text
this folder/
  common/
    bb72_baseline_common.py
    bb72_tn_common.py
    local_code_capacity_runtime.py

  examples/
    qrc311_cudaqx_nv_tn_flip_prob_example.py
    bb72_cudaqx_nv_tn_flip_prob_example.py
    results/

  bp_osd_baselines/
    bb72_qldpc_bp_osd_baseline.py
    bb72_cudaqx_nv_qldpc_bp_osd_baseline.py
    results/

  tn_reconstruct/
    bb72_cudaqx_nv_tn_flip_to_coset_reconstruct.py
    bb72_cudaqx_nv_tn_flip_to_coset_reconstruct_cached.py
    submit_tn_decoder.txt
    path_cache/
    results/

  ler_pilot/
    bb72_tn_ler_pilot_worker.py
    merge_bb72_tn_ler_pilot.py
    run_pilot_4gpu.sh
    submit_tn_decoder.txt
    results/

  ler_sweep/
    bb72_tn_ler_sweep_worker.py
    merge_bb72_tn_ler_sweep.py
    run_ler_sweep_64gpu.sh
    submit_ler_sweep_64gpu.txt
    results/
```

By default, each runnable script writes result data under the `results/`
directory next to that script. PBS run logs are written as `output-*.txt` in the
directory where you submit the job. Submit from the script directory to keep the
log next to the code. The only exception is reusable tensor-network path cache,
which is stored under `tn_reconstruct/path_cache/`.

## Typical Cluster Commands

Generate or reuse the BB72 TN path cache:

```bash
cd /path/to/this-folder/tn_reconstruct
qsub submit_tn_decoder.txt
```

Run the 4-GPU LER pilot:

```bash
cd /path/to/this-folder/ler_pilot
qsub submit_tn_decoder.txt
```

Run the 64-GPU LER sweep:

```bash
cd /path/to/this-folder/ler_sweep
qsub submit_ler_sweep_64gpu.txt
```

Run BP+OSD baselines manually:

```bash
cd /path/to/this-folder/bp_osd_baselines
python bb72_qldpc_bp_osd_baseline.py --error-type X --p-grid 0.005,0.01,0.02 --shots 5000
python bb72_cudaqx_nv_qldpc_bp_osd_baseline.py --error-type X --p-grid 0.005,0.01,0.02 --shots 5000 --cuda-device-id 0
```

The PBS submit scripts assume you run `qsub submit_tn_decoder.txt` from the
script directory. They do not assume the parent folder is named `tn_decoder`.

## Cache Location

The cached TN reconstruction writes one total path-cache file here by default:

```text
tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl
```

The LER pilot reads the same file. If you already have the old cache from before
the directory cleanup, copy or rename it to this path before running the pilot.

## Dependencies

The source files are self-contained, but the cluster environment still needs:

- `qldpc`, `ldpc`, `numpy`, `scipy`, `sympy` for BB72 construction and BP+OSD.
- `cudaq-qec[tensor-network-decoder]` for CUDA-QX TN decoding.
- A CUDA-capable GPU node for the CUDA-QX `cutensornet` runs.
