# BB72 v0.5 — multi-syndrome P5 validation (single GPU, resumable)

This package validates whether the P5 winner is robust across a large syndrome distribution rather than only the original easy/hard pair.

## Dataset

Part A (natural iid X errors): 5000 shots at each

`p = 0.005, 0.01, 0.02, 0.03, 0.04, 0.05`

for 30,000 natural shots total.

Part B (stress): 5000 fixed-weight X errors, balanced over weights 2, 3, 4, 5 (1250 each by default). The decoder prior for the stress set is `p=0.005` by default.

Total per layout: **35,000 decodes**.

The exact error vectors, syndromes and true logical sectors are generated once under `output-dir/datasets/` and reused for every layout, so comparisons are paired shot-by-shot.

## Recommended layout set (`validation6`)

1. `stageA-b9-identity` — current P5.1 winner
2. `stageA-b10-identity` — close competitor
3. `stageA-b6-identity` — earlier strong configuration
4. `stageA-b12-identity` — one full 4096-output contraction, used as a control
5. `stageB-b9-random-1` — best non-identity b=9 partition from P5.1
6. `stageB-b9-reverse` — deliberately different 9+3 partition

This gives a useful mix of block-size and partition tests while remaining around the scale of a ~120 GPU-hour campaign based on the two-syndrome timings. Higher-p syndromes may make BnB slower, so one 120 h job is not guaranteed to finish; resume is automatic.

To run every one of the 15 P5.1 candidates instead, use `--layout-selection all15`. Based only on the previous two-syndrome timing, all15 is already roughly >600 GPU-hours for 35,000 shots/layout, before any high-p slowdown.

## Resume / checkpoint behavior

`results.sqlite` is the source of truth. Every completed shot is committed immediately under a unique `(layout_tag, dataset_name, shot_id)` key. Re-running the same command:

- reuses the exact same dataset files;
- skips every completed shot;
- resumes the first incomplete shot;
- recreates decoder/network state as needed;
- never repeats path search (compiled P5.1 profiles are loaded).

`--max-wall-hours` asks the script to stop cleanly after the current shot. The included 24 h PBS file uses `23.5` h so it can be submitted repeatedly even if the single-GPU queue does not allow a 120 h job.

## Main run

```bash
python run_bb72_multisyndrome_validation.py \
  --scan-dir p5_1_scan \
  --output-dir p5_multisyndrome_validation \
  --layout-selection validation6 \
  --p-values 0.005,0.01,0.02,0.03,0.04,0.05 \
  --natural-shots 5000 \
  --stress-shots 5000 \
  --stress-weights 2,3,4,5 \
  --stress-p 0.005
```

If your queue allows a single 120 h job, omit `--max-wall-hours` (or use a suitable value below scheduler walltime). Otherwise use the included PBS repeatedly.

## What is recorded per shot

- online BnB wall time and oracle time
- oracle calls and calls by logical topology
- leaves evaluated / pruned leaves / expanded nodes
- physical-error weight and syndrome weight
- ML logical sector and its normalized probability
- true logical sector and logical success/failure
- numerical cancellation diagnostics

The call-count dictionary directly measures how many prefixes survive into each later block. For a two-block b=9 layout, calls to the second block are the operational `N_survive` discussed in the P5 analysis.

## Outputs

- `results.sqlite` — resumable per-shot database
- `summary.csv` — mean/median/P90/P95/P99 latency, oracle calls and LER by layout/dataset
- `consistency_vs_b12.csv` — shot-matched comparison against the b=12 full-logical-tensor control
- `run_manifest.json` — run settings
- `datasets/*.npz` — frozen paired datasets

Rebuild summaries at any point with:

```bash
python summarize_bb72_multisyndrome.py --output-dir p5_multisyndrome_validation
```

## Timing convention

Path search, decoder construction and one warm-up decode per `(layout,p)` are excluded from online latency. The timed quantity is the exact BnB search from a ready decoder receiving a syndrome to an ML logical sector.

P4 scratch retention is off by default because the v0.3 experiment showed ~22.7 GB of retained scratch with negligible latency gain. P0/P1/P2/P3 and compiled P5.1 paths remain active.
