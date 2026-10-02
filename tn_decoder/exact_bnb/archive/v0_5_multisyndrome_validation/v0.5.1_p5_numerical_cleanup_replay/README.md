# BB72 v0.5.1 — P5 numerical-cleanup / mismatch replay

This package is the next step after the v0.5 multi-syndrome validation.  It does **not** change the decoder algorithm.  It replays the 490 extracted b9-vs-b12 anomalous cases with one reconstructed BB72 basis and one shared multi-logical TN decoder, then compares the 9+3 BnB oracle against a common 12-logical full-tensor reference.

## Why the supplied bundle is sufficient

The input bundle contains all 490 replay inputs (`errors`, `syndromes`, `true_lambdas`), the exact b9/b12 layout tags and recorded results, the v0.5 decoder/BnB sources, and both compiled path profiles.  H_Z/L_Z/L_X were not saved originally, so this replay rebuilds them using the **existing v0.5 `build_bb72()` constructor**.  Before any replay is accepted, it checks **all 490 extracted cases** satisfy

```
H_Z @ error mod 2 == frozen syndrome
L_Z @ error mod 2 == frozen true_lambda
LX @ LZ.T mod 2 == I_12
```

If any check fails, the tool aborts.  This prevents silently replaying with a different qLDPC logical basis.

The b9 profile also contains a full 12-logical topology.  The script verifies that its full-topology `path+slicing` exactly matches the standalone b12 profile before running.

## What the replay actually measures

For every selected case it performs:

1. **Replay b9 exact BnB** (`[0..8] | [9,10,11]`) with a traced oracle.
2. **One full 12-logical contraction** on the same decoder/noise model/profile, producing the raw 4096-entry tensor.
3. Evaluate the original b9 candidate, original b12 candidate, replayed b9 candidate, and true logical sector **inside that same full tensor**.
4. Marginalize the same full tensor to a 9-bit prefix tensor and compare it directly against the b9 root oracle.
5. For every explored 3-bit tail, compare the b9 tail oracle against the corresponding slice of the full tensor.

This answers the immediate question: **where does b9-vs-b12 disagreement first appear?**  It distinguishes root-prefix disagreement / potential pruning from tail-contraction disagreement and simple near-ties.

Important limitation: the common 12-logical tensor is still a float64 cuTensorNet contraction.  Therefore this is a **numerical replay/audit**, not yet a mathematically rigorous high-precision certificate.  Cases where the full tensor itself shows strong cancellation are explicitly flagged `reference_stressed=1` and should be sent to a later independent/high-precision reference step.

## Run all 490 cases

```bash
conda activate cudaqx-qec
python replay_p5_numerical_cleanup.py \
  --input-bundle input_bundle \
  --output-dir p5_numerical_cleanup_replay \
  --selection all \
  --save-arrays all
```

The job is resumable.  Each finished case is committed immediately to:

```
p5_numerical_cleanup_replay/replay.sqlite
```

Rerunning the same command skips completed `case_id`s.

For a quick first pass on the original top-100 priority cases:

```bash
python replay_p5_numerical_cleanup.py \
  --input-bundle input_bundle \
  --output-dir p5_numerical_cleanup_replay_priority \
  --selection priority \
  --priority-limit 100 \
  --save-arrays all
```

Or replay one named case, e.g. the report's stress p=0.005 shot 827:

```bash
python replay_p5_numerical_cleanup.py \
  --input-bundle input_bundle \
  --output-dir replay_shot827 \
  --case-id stress_w2-3-4-5_p0p005__shot_000827 \
  --save-arrays all
```

## Main outputs

```
replay.sqlite
replay_results.csv
replay_summary.json
replay_priority_top100.csv
basis_verification.json
profile_verification.json
H_Z_rebuilt.npy
L_Z_rebuilt.npy
L_X_rebuilt.npy
case_arrays/*.npz
```

`case_arrays/*.npz` saves the raw 4096 full tensor, raw 512-entry b9 root tensor, and all explored 8-entry b9 tail tensors.  For 490 cases this remains modest in size and makes later diagnosis possible without repeating GPU contractions.

## Key classification labels

- `AGREE_REPLAY` — replayed b9 picks the common full-tensor argmax.
- `NEAR_TIE_COMMON_FULL` — different sector, but common-reference probability gap <= `--tie-tol-prob` (default `1e-10`).
- `REF_PREFIX_PRUNED__CHECK_ROOT_BOUND` — the common-reference best sector's 9-bit prefix was never explored by b9; inspect the root prefix bound vs the full-tensor marginalized prefix.
- `REF_PREFIX_EXPLORED__TAIL_NUMERICAL_DISAGREEMENT` — the correct prefix was explored, so disagreement is downstream in the 3-bit tail contraction/ranking.
- Prefix `FULL_REFERENCE_STRESSED__...` — the full 12-logical tensor itself has negative-cancellation scale/root above `--reference-stress-rtol`; do not treat it as a final ground truth.

## Recommended first action

Run the single-case shot 827 command first.  If it works and the basis/profile checks pass, run all 490 cases.  The 490-case run should be much shorter than the original 360k validation because it only replays the extracted anomalies.

## PBS

`submit_replay_single_gpu.pbs` is included.  It requests one GPU and 8 hours, which should be ample based on the v0.5 b9/full timings.  Edit the PBS project/queue if needed for your account.
