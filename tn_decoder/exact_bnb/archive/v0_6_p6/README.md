# BB72 exact BnB v0.6 = P6

This is the cleaned P6-only package.

Validated logical layout:

```text
[0,1,2,3,4,5,6,7,8] | [9,10,11]
```

P6 adds optional same-level final-parent micro-batching to the validated b=9 baseline.

## Package contents

```text
v0_6_p6_clean/
├── README.md
├── exact_block_bnb.py
├── multi_logical_tensor_network_decoder.py
├── run_bb72_p6_easy_hard.py
├── test_p6_bnb.py
├── submit_p6_easy_hard_single_gpu.pbs
└── profiles/
    └── canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl
```

No multisyndrome validation scripts, P5 scan utilities, old profiles, or P5 scan result files are included.

## P6 switch

Sequential baseline:

```python
ExactBlockBnB(..., parent_batch_size=1)
```

P6 batching:

```python
ExactBlockBnB(
    ...,
    batch_oracle=batch_oracle,
    parent_batch_size=2,
)
```

The first surviving parent is still evaluated sequentially to establish the incumbent. Only subsequent competitive parents are micro-batched.

## Easy/hard benchmark

The runner uses the original two BB72 test cases:

```text
easy physical X error: [28]
hard physical X error: [0,27,30]
```

Run on an allocated GPU:

```bash
python run_bb72_p6_easy_hard.py \
  --parent-execution compare \
  --batch-sizes 2,4 \
  --warmup 1 \
  --repeats 5 \
  --output-dir p6_easy_hard_results
```

Or submit:

```bash
qsub submit_p6_easy_hard_single_gpu.pbs
```

Expected qualitative behavior:

- easy: no batch call should be needed;
- hard: batching should reduce repeated tail executions if multiple prefixes survive;
- compare wall-clock latency, not only oracle-call count.

Outputs:

```text
p6_easy_hard_results/p6_easy_hard_summary.csv
p6_easy_hard_results/p6_easy_hard_results.json
```

## CPU search-logic test

```bash
pytest -q test_p6_bnb.py
```

## Scope

This package is intentionally limited to P6 and the single validated b9 compiled profile. If P6 gives no meaningful wall-time improvement on the hard case, proceed to P7 environment reuse.
