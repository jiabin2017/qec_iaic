# v0.5_final_production

Frozen BB72 exact logical-sector branch-and-bound production baseline.

This package intentionally keeps only the validated **P0-P5** production path
and the winning canonical b9 profile.  P6/P7 experimental code is not included.
P8 is deferred until larger-code scaling makes multi-GPU slicing worthwhile.

## Frozen production configuration

```text
Code:            BB72 [[72,12,6]]
Noise:           homogeneous iid X, default p = 0.005
Logical order:   0,1,...,11
Block layout:    [0..8] | [9..11]
Block size:      9
P2 cache:        ON
P3 GPU resident: ON
P4 scratch:      OFF by default
P6 batching:     NOT PRESENT
P7 reuse:        NOT PRESENT
```

The retained compiled profile is:

```text
profiles/canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl
```

## What is retained

The runtime carries forward the validated cumulative P0-P5 implementation:

- **P0**: offline compiled cuTensorNet contraction path/slicing profile.
- **P1**: persistent prepared Network reuse across oracle calls.
- **P2**: constant-operand/cache support.  It had little additional BB72 speedup,
  but is stable and remains enabled by default.
- **P3**: GPU-resident base operands with in-place syndrome/boundary updates.
  It had little additional BB72 speedup, but is stable and remains enabled.
- **P4**: scratch-retention capability remains in the decoder implementation,
  but production default is OFF because the BB72 benchmark showed essentially
  no benefit.
- **P5**: winning b9 identity layout and canonical compiled path.

The major measured gains came from P0/P1 and P5.

## What is deliberately removed

This production package does not include:

```text
multisyndrome validation scripts
P5 scan/grid-search utilities
b5/b6/b7/b8/b10/b11/b12 profiles
random/reverse-order profiles
numerical-cleanup replay tooling
P6 batching code
P7 pairwise/environment-reuse code
P8 multi-GPU experiments
historical scan/checkpoint outputs
```

Those belong in the research archive, not the production runtime.

## Reference smoke cases

The production runner uses the two long-standing BB72 cases:

```text
easy: physical X error [28]
hard: physical X error [0,27,30]
```

The hard syndrome has a true three-fold ML degeneracy under homogeneous iid X
noise.  Any of the following is therefore accepted as an exact ML answer:

```text
001110011011
111011000000
100110011001
```

Reference probabilities at p=0.005 are approximately:

```text
easy: 0.9999999955333834
hard: 0.3333164874732356
```

Historical A100 steady-state timings for the frozen b9 baseline were roughly:

```text
easy: ~0.21 s
hard: ~0.39 s
```

Timing is not used as a correctness gate because it depends on system load and
GPU/runtime details.

## Run the production smoke benchmark

From this directory:

```bash
python run_bb72_easy_hard.py \
  --warmup 1 \
  --repeats 5 \
  --output-dir production_smoke_results
```

Outputs:

```text
production_smoke_results/
├── production_easy_hard_summary.csv
└── production_easy_hard_results.json
```

The script exits non-zero if the reference ML/probability checks fail.

## CPU-only BnB logic tests

```bash
pytest -q test_exact_bnb.py
```

These tests exercise the exact sequential BnB logic without CUDA-QX/qLDPC.

## Directory

```text
v0.5_final_production/
├── README.md
├── exact_block_bnb.py
├── multi_logical_tensor_network_decoder.py
├── run_bb72_easy_hard.py
├── test_exact_bnb.py
└── profiles/
    └── canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl
```

## Development rule

Use this directory as the clean starting point for later larger-code scaling
and real-time decoder work.  Keep P6/P7 negative-result implementations in the
archive rather than merging them back into this production branch.
