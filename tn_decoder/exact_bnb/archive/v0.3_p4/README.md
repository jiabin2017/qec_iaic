# v0.3 = P4 (built on v0.2 = P3 + P2 and v0.1 = P0 + P1)

v0.3 keeps the **same exact b=3 BnB mathematics, logical order, numerical
protection, compiled path profile, GPU-resident operands, and constant-cache
logic**.  The only new performance feature is P4:

- **P4 — retain cuTensorNet SCRATCH workspaces across repeated contractions**
  - v0.2 released SCRATCH after every oracle call while keeping only P2 CACHE.
  - v0.3 keeps SCRATCH alive for a prepared topology whenever the retention
    budget allows it.
  - Repeated calls to the same topology can therefore reuse the already
    allocated multi-GB scratch workspace instead of allocating/freeing it again.
  - A global LRU budget prevents scratch retention from consuming the whole GPU.
  - Evicting SCRATCH does **not** discard the prepared `Network`, compiled path,
    GPU operands, or P2 constant CACHE.

The optimization stack is now:

```text
v0.1 : P0 compiled path + P1 cross-syndrome prepared-Network reuse
v0.2 : v0.1 + P3 GPU operands + P2 constant-intermediate cache
v0.3 : v0.2 + P4 retained SCRATCH workspace
```

## Default BB72 policy

The runner defaults to:

```text
memory_limit          = 50%   # per cuTensorNet Network, same path profile as v0.1/v0.2
scratch retain budget = 70%   # global P4 retained-SCRATCH budget
```

With the v0.1 profile used in the previous BB72 run, the four b=3 scratch
requirements were approximately:

```text
[0,1,2]    ~ 6.98 GB
[3,4,5]    ~ 0.44 GB
[6,7,8]    ~ 8.79 GB
[9,10,11]  ~ 6.48 GB
----------------------
total      ~22.7 GB
```

Thus all four should normally fit under a 70% budget on a 40-GB A100, while
leaving substantial headroom for CUDA/cuTensorNet overhead and the small P2
CACHE allocations.

## Memory-aware retention

Before allocating SCRATCH for a topology, v0.3 checks the sum of currently live
SCRATCH workspaces plus the requested workspace. If that would exceed the
budget, least-recently-used SCRATCH buffers are released until the request fits.

Accepted budget syntax:

```text
--scratch-retain-budget 70%
--scratch-retain-budget 28GB
--scratch-retain-budget 24GiB
```

A percentage is resolved against total device memory.

## Full-4096 validation is deliberately deferred

The full 12-open reference topology needs roughly 20+ GB of SCRATCH with the
current profile. Keeping that simultaneously with all four b=3 retained
workspaces would exceed a 40-GB GPU.

Therefore v0.3 first finishes **all online BnB timing measurements**:

1. easy RUN1
2. easy RUN2
3. hard RUN1
4. hard RUN2

Only after those measurements are frozen does the runner release the retained
b=3 SCRATCH buffers and execute the two full-4096 correctness references.
Prepared Networks, paths, P3 buffers, and P2 state survive this release.

This prevents the validation reference from contaminating the P4 benchmark.

## Run

Reuse the existing v0.1/v0.2 path profile; no retuning is required just to test
P4:

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl
```

For a pure performance run without full-4096 validation:

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl \
  --no-full-reference
```

## What should prove that P4 is active?

After the first easy run, each b=3 topology should normally show:

```text
p4_retain_scratch_requested: True
p4_scratch_retained: True
scratch_workspace_live: True
```

The aggregate line should show a non-zero live scratch total:

```text
P4 workspace info = {
    'planned_scratch_bytes': ...,
    'live_scratch_bytes': ...,
    'scratch_reuse_hits': ...,
    'scratch_evictions': ...,
    'last_budget_bytes': ...
}
```

On easy RUN2, `p4_scratch_reuse_hits` should increase for the four topologies.
On hard RUN1, the same retained workspaces should continue to be reused across
the syndrome change.

If the 70% budget can hold all four BB72 workspaces, `scratch_evictions` should
remain zero during the four BnB runs.

## A/B comparison against v0.2

Disable P4 while keeping P0/P1/P3/P2:

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl \
  --no-retain-scratch
```

That is the most important v0.2-vs-v0.3 ablation.

Other existing ablations remain available:

```bash
# P4 + P3 but no P2 constant cache
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl \
  --no-constant-cache

# approximately v0.1 engine behavior (also disable P4)
python run_bb72_exact_bnb_b3_two_syndromes.py \
  --path-profile bb72_b3_paths.pkl \
  --no-gpu-resident-operands \
  --no-constant-cache \
  --no-retain-scratch
```

## OOM safety

If a retained-workspace contraction still encounters a CUDA/cuTensorNet memory
allocation error, v0.3:

1. removes the failing topology instance,
2. releases retained P4 SCRATCH workspaces,
3. purges P2 CACHE as well when P2 is enabled,
4. recreates the topology with the compiled path,
5. retries once in the safe v0.2 released-SCRATCH mode.

The fallback is printed through `fast fallback` and does not alter the BnB
mathematics.

## Files

- `multi_logical_tensor_network_decoder.py` — P0/P1/P3/P2 plus new P4 policy.
- `exact_block_bnb.py` — unchanged exact b=3 BnB search.
- `run_bb72_exact_bnb_b3_two_syndromes.py` — performance + deferred validation.
- `tune_bb72_b3_paths.py` — unchanged P0 offline path compiler.
- `test_exact_block_bnb.py` — unchanged CPU BnB regression tests.

## Local checks performed

```text
python -m py_compile *.py    PASS
pytest -q test_exact_block_bnb.py
2 passed
```

This environment does not contain the full CUDA-QX/quimb/cuTensorNet GPU stack,
so actual retained-workspace behavior must be verified on Aspire.
