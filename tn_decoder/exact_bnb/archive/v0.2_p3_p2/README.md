# v0.2 = P3 + P2 (built on v0.1 = P0 + P1)

This version keeps the **same exact b=3 BnB mathematics, logical order,
numerical guard, and v0.1 compiled path profile**.  It adds only the next two
TN-engine optimizations:

- **P3 — GPU-resident operands**
  - The base tensor network is uploaded to GPU **once** as shared CuPy arrays.
  - All four b=3 `Network` objects reference the same base GPU buffers.
  - A new syndrome updates only the syndrome tensors in-place.
  - A new BnB node updates only the small logical boundary projectors in-place.
  - Cache hits therefore do **not** call `reset_operands(...)`.

- **P2 — cuTensorNet constant-intermediate caching**
  - Static code/logical/noise operands are marked `isConstant`.
  - Syndrome operands and BnB boundary projectors remain variable.
  - cuTensorNet can cache contraction intermediates that depend only on static
    operands and reuse them across BnB branches and across syndrome changes.

P0/P1 from v0.1 remain active:

- compiled path+slicing profile is loaded from `bb72_b3_paths.pkl`;
- prepared Network topologies survive syndrome changes.

## Important workspace policy in v0.2

P2 needs cuTensorNet's **CACHE** workspace to remain alive.  But we still do not
want all four large **SCRATCH** workspaces resident simultaneously.

Therefore v0.2 executes with:

```text
CACHE   : retained
SCRATCH : released after each contraction
```

The current cuQuantum high-level `Network` public flag `release_workspace=True`
releases both kinds.  v0.2 therefore uses the current Network implementation's
scratch-only internal release helper after each contraction.  The runner prints
whether this is supported:

```text
scratch_only_release_supported: True
p2_cache_retained: True
cache_workspace_live: True
scratch_workspace_live: False
```

This deliberately leaves **P4 (workspace retention / memory tuning)** for v0.3.

## Files

- `multi_logical_tensor_network_decoder.py` — v0.2 P3/P2 implementation.
- `exact_block_bnb.py` — same BnB logic; only adds CuPy -> NumPy result conversion.
- `run_bb72_exact_bnb_b3_two_syndromes.py` — easy + symmetry-degenerate hard case.
- `tune_bb72_b3_paths.py` — unchanged v0.1 P0 path compiler.
- `test_exact_block_bnb.py` — CPU BnB tests.

## Reuse the v0.1 path profile

You do **not** need to retune P0 just to try v0.2. Copy your existing
`bb72_b3_paths.pkl` into this directory and run:

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl
```

The full-4096 correctness reference uses P3 but deliberately does **not** retain
P2 cache workspace, so it cannot consume a fifth large constant-cache workspace
and interfere with the four BnB topologies.

## What to inspect in the output

### P3 should show one base upload

Look for:

```text
P3 GPU operand info = {
  'enabled': True,
  'base_uploads': 1,
  ...
}
```

`base_uploads` should remain `1` for the whole program.  When CASE B changes to
a new syndrome, `syndrome_updates` should increase, but the static base tensors
should not be uploaded again.

### P2 should be genuinely active

Each b=3 cache row should show:

```text
uses_constant_qualifiers: True
gpu_resident_operands: True
p2_cache_requested: True
p2_cache_retained: True
cache_workspace_live: True
scratch_workspace_live: False
```

Also inspect `workspace_cache_size`.  A positive value means cuTensorNet has
allocated actual cache space for reusable constant intermediates.  The amount
can differ strongly by topology/path.

### Cross-syndrome reuse must still work

CASE B RUN 1 should immediately have positive cache-hit deltas for the same four
b=3 topologies; a syndrome change must not recreate them.

## Useful ablations

To measure **P3 only** (GPU resident operands, no constant-intermediate cache):

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl \
  --no-constant-cache
```

To approximately recover **v0.1 behavior** in the same code base:

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl \
  --no-gpu-resident-operands \
  --no-constant-cache
```

This gives a clean three-way comparison:

```text
v0.1-like        = P0 + P1
P3-only          = P0 + P1 + P3
v0.2 default     = P0 + P1 + P3 + P2
```

## Correctness / safety

- Constant qualifiers are **not** placed on syndrome tensors.
- Constant qualifiers are **not** placed on logical boundary projectors.
- Static operands keep stable GPU pointers for the lifetime of the decoder.
- Variable operands are overwritten in place.
- If cuTensorNet reports a memory-allocation failure, v0.2 has an emergency
  fallback that purges retained P2 workspaces and retries without changing the
  BnB mathematics.
- The full-4096 reference remains the end-to-end correctness check.

## Local checks performed

```text
python -m py_compile ...     PASS
pytest -q test_exact_block_bnb.py
2 passed
```

GPU/cuTensorNet behavior must be verified on Aspire because this environment
cannot execute the CUDA-specific path.
