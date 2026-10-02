# BB72 exact logical-sector BnB — v0.7 = P7

Baseline: **v0.6 with `parent_batch_size=1`**.

P0–P5 production optimizations are carried forward.  The P6 batching code
remains inside `exact_block_bnb.py`, but P7 tests explicitly keep
`parent_batch_size=1`, because P6 was correct but did not improve BB72 wall time.

## P7 idea

P6 showed that merely batching two tail contractions reduced the number of
physical oracle launches but did not reduce tensor work.

P7 therefore attacks repeated work *inside* the contraction tree.

For the validated b9 layout

```text
[0,1,2,3,4,5,6,7,8] | [9,10,11]
```

the root `[0..8]` query stays on the existing P0–P5 cuTensorNet path.

Repeated tail queries have the same tensor topology and differ only in the
nine fixed logical-boundary projectors.  P7 executes the already-validated P5
pairwise tail path and memoizes an intermediate by

```text
(contraction step, values of exactly the prefix bits that intermediate depends on)
```

So if two surviving prefixes agree on the bits required by an intermediate,
that GPU intermediate is reused rather than recomputed.

The cache is **per decode / per syndrome**.  It is never reused across shots.

## Modes

`baseline`
: ordinary v0.6 behavior, `parent_batch_size=1`.

`p7_always`
: every tail query uses the P7 memoized pairwise executor.  This shows the raw
  reuse potential but may add overhead to easy cases with only one tail query.

`p7_adaptive`
: the first tail query uses the ordinary cuTensorNet fast path.  P7 is started
  only if a second tail query is needed.  This deliberately preserves the
  easy-syndrome baseline; only the third and later tail queries can reuse
  intermediates created by the second query.

## First experiment

Exactly the original two syndromes are used:

```text
easy: X error [28]
hard: X error [0,27,30]
```

Run:

```bash
qsub submit_p7_easy_hard_single_gpu.pbs
```

or interactively:

```bash
python run_bb72_p7_easy_hard.py \
  --modes baseline,p7_always,p7_adaptive \
  --warmup 1 \
  --repeats 5 \
  --p7-cache-budget 8GB \
  --p7-max-entry 2GB \
  --p7-max-dependency-bits 6 \
  --output-dir p7_easy_hard_results
```

## What to inspect

The main timing comparison is:

```text
baseline hard
vs
p7_always hard
vs
p7_adaptive hard
```

Also inspect:

```text
p7_cache_hits
p7_pair_contractions
p7_bytes_reused
p7_cache_peak_bytes
```

Interpretation:

* no cache hits -> this particular compiled tail path exposes little reusable
  substructure for the tested prefixes;
* cache hits but no speedup -> reusable environments exist but the manual
  pairwise executor / launch overhead dominates;
* cache hits + hard-case speedup -> P7 is worth developing further;
* adaptive easy should remain essentially the v0.6 easy baseline.

## Numerical/correctness guard

The runner compares P7's final ML sector and probability with the ordinary
baseline.  The current pass criterion is identical sector and absolute
probability difference <= 1e-10.

## Important implementation scope

This first P7 implementation is intentionally conservative:

* only the **unsliced** b9 tail topology `[9,10,11]` uses environment reuse;
* the root contraction remains the validated cuTensorNet P0–P5 path;
* the validated b9 tail profile has zero slices, so no slicing semantics need
  to be reimplemented in this experiment;
* P6 remains disabled.

If this targeted experiment shows no meaningful reuse or no wall-time gain,
do not expand it mechanically.  The next P7 attempt should instead change the
contraction tree / separator so that large reusable environments are created
deliberately.

## Files

```text
v0_7_p7/
├── README.md
├── exact_block_bnb.py
├── multi_logical_tensor_network_decoder.py
├── p7_environment_reuse.py
├── run_bb72_p7_easy_hard.py
├── submit_p7_easy_hard_single_gpu.pbs
├── test_p7_environment_reuse.py
└── profiles/
    └── canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl
```
