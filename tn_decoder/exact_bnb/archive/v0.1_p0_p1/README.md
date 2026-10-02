# v0.1 = P0 + P1

This directory is based directly on the supplied **v0** baseline.  The BnB
mathematics, block size (`b=3`), logical ordering, numerical guard, and
`release_workspace=True` policy are unchanged.

v0.1 adds only the first two performance items:

- **P0 — offline path compilation**: search several cuTensorNet contraction
  paths for each fixed b=3 topology, benchmark them on the current GPU, save
  the fastest path+slicing plan to `bb72_b3_paths.pkl`, and load that plan in
  later decoder runs.  A compiled path is reusable for all branch values and
  all syndromes because the equation and tensor shapes do not change.
- **P1 — cross-syndrome persistent Network reuse**: a syndrome change no
  longer clears ordinary fast Network objects.  The new syndrome values are
  written into the syndrome tensors, and cache hits call `reset_operands(...)`
  on the already prepared Network.  Experimental constant qualifiers remain
  disabled; qualifier-enabled Networks are still invalidated on syndrome
  changes for safety.

## Files

- `multi_logical_tensor_network_decoder.py` — v0.1 TN oracle with compiled-path
  profiles and cross-syndrome Network reuse.
- `exact_block_bnb.py` — unchanged BnB core from v0.
- `run_bb72_exact_bnb_b3_two_syndromes.py` — easy + symmetry-degenerate hard
  syndrome benchmark, two runs each, now without clearing the topology cache
  between the two syndromes.
- `tune_bb72_b3_paths.py` — new P0 offline path compiler/benchmark.
- `test_exact_block_bnb.py` — unchanged CPU-side BnB tests.

## Step 1: compile paths once on the target GPU

Run this once on the same GPU class / environment that will be used for the
benchmark:

```bash
python tune_bb72_b3_paths.py \
  --trials 6 \
  --samples 32 \
  --benchmark-repeats 1 \
  --memory-limit 50% \
  --output bb72_b3_paths.pkl
```

For a more serious offline search, increase trials/samples, for example:

```bash
python tune_bb72_b3_paths.py \
  --trials 12 \
  --samples 64 \
  --benchmark-repeats 2 \
  --memory-limit 50% \
  --output bb72_b3_paths.pkl
```

The compiler tunes these topologies by default:

```text
(0,1,2)
(3,4,5)
(6,7,8)
(9,10,11)
(0,1,2,3,4,5,6,7,8,9,10,11)   # full-4096 reference
```

Each trial runs the cuTensorNet hyperoptimizer with a different seed, executes
that candidate path, and chooses the candidate with the lowest measured median
contraction time.  The output pickle also records FLOP cost, largest
intermediate, slices, GPU name, cuQuantum version, and benchmark timings.

## Step 2: run v0.1

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl
```

If the profile file is absent, the code still runs, but the first use of each
topology falls back to native runtime path search.

### What to look for

In `fast cache info`, first-use entries should show:

```text
path_source: 'compiled_profile'
```

rather than:

```text
path_source: 'runtime_search'
```

For **CASE A**, RUN 1 creates the persistent Networks from the compiled paths;
RUN 2 should hit them.

For **CASE B**, RUN 1 uses a *different syndrome* but should still hit the same
four b=3 Networks.  That is the direct P1 test.  In other words, the CASE B
cache-hit deltas should be positive immediately instead of returning to zero.

## Important scope boundary

v0.1 deliberately does **not** implement P2/P3/P4 yet:

- no constant-intermediate cache,
- operands are not yet permanently GPU-resident,
- workspace is still released after each contraction.

This makes the v0 -> v0.1 comparison interpretable: measured changes come from
**compiled path quality + cross-syndrome topology reuse only**.

## Why P1 is valid

CUDA-QX `flip_syndromes()` modifies only the syndrome tensor values.  For a
fixed code and fixed set of open logical legs, the tensor equation, connectivity,
and operand shapes are unchanged.  Therefore the same contraction path and
persistent `Network` topology remain valid; v0.1 refreshes the operand values
through `reset_operands(...)` before contraction.
