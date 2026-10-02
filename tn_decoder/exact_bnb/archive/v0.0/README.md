# Step 2.3: tie-aware full4096 validation

The hard syndrome exposed three nearly degenerate leading logical sectors.
The previous validation incorrectly required the BnB lambda string to equal
`np.argmax(full4096)` exactly.  That is not a reliable criterion when separate
float64 TN contractions reorder candidates inside the observed cancellation
noise.

This version:
- evaluates the BnB-selected sector inside the SAME full4096 tensor,
- prints the top 8 full4096 sectors and their posterior gaps,
- keeps strict same-index PASS when possible,
- otherwise accepts a numerically ML-equivalent sector only when its gap from
  the full-tensor maximum is below the conservative numerical tolerance inferred
  from the BnB contraction,
- still raises an error for a genuine gap larger than that tolerance.

This does not prove an exact mathematical tie.  It diagnoses whether the ML
ordering is numerically unresolved at the current float64 TN precision.


# Step 2.2 adaptive numerical guard

Hard syndromes showed that cancellation noise varies strongly from one TN
contraction to another, so a single fixed negative threshold is brittle.

This revision uses an adaptive per-oracle guard:

- Physical subtree masses must be non-negative.
- If an oracle output contains a most-negative residual `-eta`, that directly
  exposes a cancellation/noise scale for that contraction.
- Negative entries are clipped to zero.
- Pruning for *all children from that contraction* uses
  `bound + max(base_guard, 16*eta) <= incumbent`.
- Extremely large negative residuals (default > 1e-6 of root mass) still abort.

This is conservative: a noisier contraction causes less pruning, not more.
The mathematical BnB bound is unchanged.  Since float64 TN contractions are not
rigorously interval-certified, full-4096 verification remains the correctness
check for these experiments.

# Step 2.1 numerical-guard hotfix

The harder weight-3 collision produced a tiny negative subtree mass from
floating-point cancellation: about -3.53e-4 raw, which is only about
-4.72e-12 on the root posterior scale.  Physical subtree masses are
non-negative.

For correctness-first BnB, the default numerical guard is therefore increased
from 1e-12 to 1e-10 of the root mass.  Pruning remains conservative:
`measured_bound + guard <= incumbent`.  Small negative values within the guard
are clipped to zero; larger negatives still abort.

The search now reports how many negative entries were clipped and the most
negative raw/posterior-scale value.  Full-4096 verification remains the final
correctness check.

# Step 2: easy + hard syndrome, each repeated twice

Run in the same directory as the V5.2 `multi_logical_tensor_network_decoder.py`:

```bash
python run_bb72_exact_bnb_b3_two_syndromes.py
```

The hard candidate is not chosen by arbitrarily raising the error rate. The
script deterministically enumerates weight-3 errors until it finds two errors
with the same BB72 syndrome but different logical sectors. Since both errors
have the same physical weight, they have the same microscopic prior under the
independent noise model. This guarantees a genuine cross-sector ambiguity at
minimum, although the total sector posterior can still be unequal because of
other errors in each coset.

For each syndrome the script:

1. clears the fast cache;
2. runs exact b=3 BnB once (cold topology cache);
3. immediately runs the same BnB again without clearing the cache;
4. prints per-topology cache-hit deltas and timings;
5. computes one full-4096 reference and checks the BnB ML sector.

The oracle uses `release_workspace=True`, so prepared Network/path metadata is
kept while large scratch workspaces are released after each contraction.
