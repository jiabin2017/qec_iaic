# v0.4.2 / P5.1 — canonical topology path pool

This is a cleanup of P5, not a new decoder approximation. Exact block BnB and
its numerical guard are unchanged.

## Why this patch exists

The v0.4 scan showed that, for `b=6`,

- identity uses blocks `(0,1,2,3,4,5)` then `(6,7,8,9,10,11)`;
- reverse uses `(11,10,9,8,7,6)` then `(5,4,3,2,1,0)`.

Within each block those are the same logical **sets**. The tensor connectivity,
operand list and dimensions are the same; only the output-axis order changes.
Nevertheless v0.4 tuned them independently, so random path-search quality was
mixed together with the real BnB ordering/grouping effect.

P5.1 uses the canonical key

```text
canonical(open_logicals) = tuple(sorted(open_logicals))
```

Thus

```text
(6,7,8,9,10,11)
(11,10,9,8,7,6)
```

share one path pool entry `(6,7,8,9,10,11)`.

The prepared cuTensorNet `Network` is still order-specific because the returned
array axes must follow the requested `open_logicals`. Only the compiled
`path+slicing` plan is shared.

## What is new

- `scan_bb72_p5_1_canonical.py`
  - Stage A defaults to `b=5,6,7,8,9,10,11,12`.
  - Stage B compares logical grouping/block order at the winning `b`.
  - one global canonical path pool is shared by all configurations;
  - within-block permutations do not trigger another random path search;
  - block order is preserved as a real BnB variable;
  - optional old `p5_scan/paths_b*.pkl` profiles are imported;
  - imported plans are **re-benchmarked on the canonical equation** before
    competing with a fresh canonical tune;
  - the fastest measured valid plan is retained for each canonical logical set.
- `multi_logical_tensor_network_decoder.py`
  - compiled path lookup first tries the exact ordered key and then the sorted
    canonical key.
- `p5_utils.py`
  - adds `canonical_topology()` and `canonical_block_signature()`.

## Recommended run using the previous P5 results

If the previous output directory is available as `p5_scan`:

```bash
python scan_bb72_p5_1_canonical.py \
  --block-sizes 5,6,7,8,9,10,11,12 \
  --import-profile-dir p5_scan \
  --path-trials 6 \
  --path-samples 64 \
  --path-benchmark-repeats 2 \
  --decoder-repeats 2 \
  --output-dir p5_1_scan
```

The old identity/reverse paths are treated only as candidate plans. For example,
if the old reverse run found a 0.164 s plan for `(11,10,9,8,7,6)`, P5.1 first
tries that plan on canonical `(6,7,8,9,10,11)`. If valid, it competes directly
against the old identity plan and the new fresh tune. This removes the old
`ordering effect + random path quality effect` confound.

Outputs include:

```text
p5_1_scan/p5_1_scan_results.json
p5_1_scan/p5_1_best_config.json
p5_1_scan/canonical_path_pool.pkl
p5_1_scan/canonical_paths_b*.pkl
```

The best config can be checked with the existing runner:

```bash
python run_bb72_exact_bnb_p5.py \
  --config p5_1_scan/p5_1_best_config.json
```

## Memory policy

As in the P5 scan, P4 scratch retention is off by default because it consumed
~22.7 GB on the old b=3 setup without a measurable latency gain. Add
`--retain-scratch` only for an explicit ablation.

## Local checks

```text
py_compile: PASS
pytest: 9 passed
```

GPU/cuTensorNet path portability and timings must be validated on the A100 run;
imported plans that are not valid for the canonical equation are automatically
rejected and fresh tuning remains available.
