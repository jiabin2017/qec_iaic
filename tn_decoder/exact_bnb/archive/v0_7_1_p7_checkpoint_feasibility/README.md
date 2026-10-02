# v0.7.1b — P7 checkpoint/separator feasibility test (touch-tracking fix)

This is **not a new production decoder**.  Production remains the validated
v0.6 baseline with `parent_batch_size=1`.

The previous v0.7 pairwise-memoization attempt is intentionally not carried
forward.  That experiment was correct but produced zero cache hits and large
slowdowns.

## What this test checks

The intended P7 idea is now:

1. run the validated b9 root contraction with logicals `0..8` open;
2. rank the 512 prefix masses;
3. keep a contraction checkpoint/environment from the root calculation;
4. for only the surviving prefixes, fix the nine prefix output bits in that
   checkpoint and continue a much smaller residual network with logicals
   `9,10,11` open.

The critical issue is that the current b9 root marginalizes logicals 9,10,11
with `[1,1]^T`.  Once any such projector has been contracted, that tail bit has
been summed out and cannot be recovered.

Therefore the script traces the **actual validated P5 b9 contraction path** and
finds the latest point at which all three tail `[1,1]^T` projectors are still
untouched.

At that checkpoint it reports:

- how much of the root contraction FLOP work has already been completed;
- how many tensors remain;
- checkpoint memory per cuTensorNet slice;
- memory required if checkpoints for all 16 root slices are retained;
- residual memory after fixing one selected 9-bit prefix;
- largest residual tensor;
- support/overlap statistics of the three tail logical operators.

This directly tests whether the proposed

```text
big root contraction
        |
saved environment
        |
choose A1,A2,A3
        |
small residual TN with logicals 9,10,11 open
```

is structurally plausible with the **current** optimized b9 path.

## Run

```bash
qsub submit_p7_checkpoint_feasibility.pbs
```

This does not perform the expensive P7 pairwise contractions.  It is mainly a
symbolic contraction-tree/path analysis and should be much cheaper than the
previous v0.7 benchmark.

Outputs:

```text
p7_checkpoint_feasibility/
├── p7_checkpoint_feasibility.json
└── p7_safe_checkpoints.csv
```

plus the PBS log:

```text
output-p7-checkpoint-feasibility.txt
```

## How we will interpret it

A promising checkpoint has both:

1. a substantial `completed_flop_fraction`; and
2. manageable `retain_all_slices_live_bytes` / branch-sliced residual memory.

If the first tail `[1,1]` projector is consumed very early, the current P5
single-shot-optimal contraction tree is incompatible with checkpoint reuse.
Then P7 would require a **new path optimized with a separator/checkpoint
constraint**, rather than further caching tricks.

If a late, compact safe checkpoint already exists, the next step is to implement
a real checkpoint-resume prototype that keeps the saved environment and replaces
the three tail marginalizers by open tail legs only after the winning prefixes
are known.

## v0.7.1b fix

The first v0.7.1 analyzer incorrectly treated a standalone tail `[1,1]^T`
projector as already "touched" merely because its lineage existed in the live
tensor list. That made the safe-checkpoint list empty before any contraction
was performed and caused `safe[-1]` to raise `IndexError`.

v0.7.1b tracks **actual contraction participation** separately. A tail bit is
marked touched only when its `[1,1]^T` projector is selected by a contraction
step.
