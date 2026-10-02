# BB72 exact TN-BnB v0.4 — P5 block size + logical ordering

v0.4 is built on v0.3 and keeps P0/P1/P2/P3/P4 available.  The new work is
**P5**: change the exact BnB search geometry so that the decoder performs less
real tensor-contraction work.

The mathematics is unchanged.  For every partial assignment `A`, the TN oracle
still returns exact subtree masses `W(A)`, and pruning is still based on the
rigorous upper bound `max_leaf(A) <= W(A)` plus the existing floating-point
numerical guard.  P5 only changes which logical bits are opened together and in
which block order they are visited.

## What is new

### 1. Arbitrary block size

`ExactBlockBnB` already supported arbitrary block size.  v0.4 exposes it in the
BB72 runner and path compiler:

```bash
--block-size 1
--block-size 2
--block-size 3
--block-size 4
--block-size 6
```

For `k=12`, the identity-order topologies are for example:

```text
b=3: (0,1,2) (3,4,5) (6,7,8) (9,10,11)
b=4: (0,1,2,3) (4,5,6,7) (8,9,10,11)
b=6: (0,1,2,3,4,5) (6,7,8,9,10,11)
```

A larger block means fewer BnB levels/oracle calls, but each contraction has
more open output legs and can be much more expensive.  Therefore the optimum is
measured, not assumed.

### 2. Arbitrary logical ordering/grouping

The runner and path compiler accept:

```text
identity
reverse
weight-asc
weight-desc
overlap-max
overlap-min
random
```

or an explicit permutation such as

```bash
--logical-order 0,4,7,1,9,3,2,5,6,8,10,11
```

`weight-*` sorts logical-Z rows by physical support size. `overlap-max` greedily
groups rows with strongly overlapping physical support, while `overlap-min`
groups rows with weak overlap.  These are **offline topology heuristics** only;
they do not look at the syndrome and add no online decoding work.

Because one TN oracle returns all `2^b` children of a block at once, the order
*inside* one block is usually less important than (i) which logicals are grouped
into the same block and (ii) which block is visited earlier in the BnB tree.

### 3. Generic P5 path compiler

`tune_bb72_p5_paths.py` replaces the b=3-only compiler.  The profile records
`block_size`, `logical_order`, and the corresponding open-logical topologies.
The decoder now accepts both legacy profile format v1 and P5 profile format v2.

Example for `b=6`:

```bash
python tune_bb72_p5_paths.py \
  --block-size 6 \
  --logical-order identity \
  --trials 6 \
  --samples 32 \
  --benchmark-repeats 1 \
  --memory-limit 50% \
  --output bb72_p5_b6_identity.pkl
```

Then benchmark the two established syndromes:

```bash
python run_bb72_exact_bnb_p5.py \
  --block-size 6 \
  --logical-order identity \
  --path-profile bb72_p5_b6_identity.pkl
```

The old `b=3` v0.1 path profile is still usable with
`--block-size 3 --logical-order identity`.

## Recommended P5 workflow: two-stage scan

`scan_bb72_p5.py` automates the experiment without attempting an impossible
search over all `12!` logical permutations.

**Stage A** scans block size with identity order.  Default:

```text
b = 1, 2, 3, 4, 6
```

**Stage B** fixes the best measured block size and scans grouping/order
heuristics, plus a small number of random permutations.

A first inexpensive scan is:

```bash
python scan_bb72_p5.py \
  --block-sizes 1,2,3,4,6 \
  --path-trials 3 \
  --path-samples 16 \
  --path-benchmark-repeats 1 \
  --decoder-repeats 2 \
  --output-dir p5_scan
```

This is still an **offline GPU autotuning job** and can take a while because
every candidate topology needs path compilation.  It does not belong to online
decoding latency.

The scan writes:

```text
p5_scan/p5_scan_results.json
p5_scan/p5_best_config.json
p5_scan/paths_b...pkl
```

Only after the scan selects the winner does it also compile the full 12-open
reference topology for the winning path profile.

Run the selected configuration directly with:

```bash
python run_bb72_exact_bnb_p5.py \
  --config p5_scan/p5_best_config.json
```

The final runner again performs the easy and hard syndrome twice and, unless
`--no-full-reference` is supplied, checks the winning BnB result against the
full 4096-sector tensor.

## P0–P4 remain available

v0.4 still contains:

- P0 offline compiled contraction paths.
- P1 persistent Network reuse across syndromes.
- P3 GPU-resident operands.
- P2 constant-intermediate cache.
- P4 optional retained SCRATCH workspace with a global memory budget.

P4 is on by default in the standalone runner to preserve v0.3 behavior.  The
P5 scanner leaves P4 off by default because v0.3 showed almost no timing gain
from retaining ~22.7 GB of BB72 scratch, and keeping it off reduces memory
pressure while many candidate topologies are being tested.  Add
`--retain-scratch` to the scan if an apples-to-apples P4-on comparison is
desired.

## Files

- `multi_logical_tensor_network_decoder.py` — v0.3 decoder plus P5 profile-v2 compatibility.
- `exact_block_bnb.py` — exact BnB engine; already supports arbitrary block size/order.
- `p5_utils.py` — P5 topology/order parsing and deterministic heuristics.
- `tune_bb72_p5_paths.py` — compile paths for one chosen P5 configuration.
- `run_bb72_exact_bnb_p5.py` — run/validate one chosen P5 configuration.
- `scan_bb72_p5.py` — two-stage block-size and logical-order autotuner.
- `test_exact_block_bnb.py` — exact-search regression tests, including nontrivial order and a partial last block.
- `test_p5_utils.py` — P5 utility tests.

The package focuses on the P5 entry points above; the earlier v0.3-specific
runner/tuner are not required.

## Local checks completed

The package was checked with:

```text
python -m py_compile ...
pytest -q test_exact_block_bnb.py test_p5_utils.py
7 passed
```

GPU/cuTensorNet timing and path quality must still be measured on the Aspire
GPU node; this environment cannot validate those runtime-specific results.

---

## v0.4.1 scan-stability hotfix

This hotfix does not change P5, the BnB mathematics, or path-selection criteria.
It fixes long single-process scan memory accumulation observed after completing
`b=1`, where the first `b=2` topology could fail every trial with
`Failed to allocate memory from pool` on a 40-GB A100.

Changes:

- aggressively release CuPy/Torch allocator caches between independent path trials
  and between P5 configurations;
- print CUDA/CuPy memory diagnostics around configurations and after OOMs;
- reuse a compatible existing `paths_b*.pkl` in `--output-dir` by default;
- write `p5_scan_checkpoint.json` after each completed configuration.

The cleanup runs outside the measured contraction interval, so it does not alter
the benchmark timing used to select a path.

For the failed run in which `b=1` already completed, the simplest continuation is
to omit `b=1`, since its measured mean latency was already much worse than the
existing b=3 baseline:

```bash
python scan_bb72_p5.py \
  --block-sizes 2,3,4,6 \
  --path-trials 6 \
  --path-samples 64 \
  --path-benchmark-repeats 2 \
  --decoder-repeats 2 \
  --output-dir p5_scan
```

Alternatively, copy the old compatible `paths_b1_identity.pkl` into the new
`p5_scan/` directory and include `1,2,3,4,6`; it will be reused automatically.
Use `--no-reuse-existing-profiles` to force recompilation.
