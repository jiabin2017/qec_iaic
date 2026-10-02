BB72 v0.5 P5 numerical-cleanup / mismatch replay input bundle
Extraction UTC: 2026-09-23T15:18:37.550447+00:00
Original root: C:\tsbyttn\projects\qldpc\tn_decoder\exact_bnb\v0_5_multisyndrome_validation

Scope
Read-only SQL selection, frozen-NPZ row extraction, byte-identical file copying and ZIP packaging.
No decoder import/execution, contraction, replay algorithm, numerical repair or source modification.
The bundle is replay INPUT material, not a runnable replay tool.

Confirmed layout tags
b9 identity = stageA-b9-identity
b12 full = stageA-b12-identity
Both logical_order = [0,1,2,3,4,5,6,7,8,9,10,11]. Blocks are 9+3 and 12, respectively.
Seven run manifests, all layout_meta records, p5_1_scan result entries, job log paths and
profile metadata/topology contents agree on these layout/profile associations.
manifests/input_locations.json records the exact local databases, frozen NPZs and manifests.
manifests/layout_meta_extracted.json preserves recorded remote runtime profile paths.

Selection and unique keys
INNER JOIN shots b9/b12 on (dataset_name,shot_id), exact layout tags above.
Include iff best_lambda differs OR abs(best_probability_b9-best_probability_b12) > 1e-10.
The comparison uses original SQLite double values, with no rounding before selection.
All 60,000 paired rows exist with full shot IDs 0..4999 in each condition.
Unique replay key is (dataset_name,shot_id); case_id is dataset_name + '__shot_' + six-digit ID.
Stress datasets at different priors remain DISTINCT replay conditions, even for reused physical errors.
No de-duplication across priors or physical error vectors has been performed.

Counts
natural: sector mismatches=103; prob_diff>1e-10=0; union=103.
stress all p: sector mismatches=271; prob_diff>1e-10=125; intersection=9; union=387.
union unique replay cases=490.
HTML report numbers match: True.
HTML comparison discrepancy: None.
Report totals 1,860/635 concern FIVE candidate layouts; this bundle selects only b9-vs-b12.

Per-dataset checks
natural_p0p005: paired=5000, sector=0, prob>1e-10=0, overlap=0, union=0
natural_p0p01: paired=5000, sector=0, prob>1e-10=0, overlap=0, union=0
natural_p0p02: paired=5000, sector=15, prob>1e-10=0, overlap=0, union=15
natural_p0p03: paired=5000, sector=20, prob>1e-10=0, overlap=0, union=20
natural_p0p04: paired=5000, sector=37, prob>1e-10=0, overlap=0, union=37
natural_p0p05: paired=5000, sector=31, prob>1e-10=0, overlap=0, union=31
stress_w2-3-4-5_p0p005: paired=5000, sector=77, prob>1e-10=114, overlap=9, union=182
stress_w2-3-4-5_p0p01: paired=5000, sector=53, prob>1e-10=11, overlap=0, union=64
stress_w2-3-4-5_p0p02: paired=5000, sector=38, prob>1e-10=0, overlap=0, union=38
stress_w2-3-4-5_p0p03: paired=5000, sector=37, prob>1e-10=0, overlap=0, union=37
stress_w2-3-4-5_p0p04: paired=5000, sector=34, prob>1e-10=0, overlap=0, union=34
stress_w2-3-4-5_p0p05: paired=5000, sector=32, prob>1e-10=0, overlap=0, union=32

Files and field conventions
numerical_cleanup_cases.csv: every selected pair, requested common fields plus ALL raw shots
columns for both layouts, prefixed b9_/b12_. Thus diagnostics are retained under their real names.
b9_pruned_sectors is an exact alias of recorded b9_pruned_leaves, not a recomputed count.
physical_error_weight aliases shots.error_weight; decoder_p aliases shots.p.
Preserved numeric diagnostics include expanded_internal_nodes, clipped_negative_entries,
most_negative_raw_mass, max_local_guard_abs, topology_call_counts, oracle_seconds and created_at.
neg_scale, local_guard, max_negative_mass, negative_mass_count and expanded_nodes are NOT
columns in these databases. No columns with those names were fabricated. In particular,
per-call guards/negative scales cannot be recovered from per-shot aggregate diagnostics.
topology_call_counts is the original JSON TEXT, not an estimated count.
root_mass is an unnormalized TN mass; do not reinterpret it as a normalized probability.
wall_seconds is recorded online BnB search time, not full job time.

numerical_cleanup_cases.npz: 490 selected rows only; array order matches cases.csv.
errors: (490,72), uint8; syndromes: (490,36), uint8;
true_lambdas: (490,12), uint8. These are direct rows of frozen NPZ arrays.
case_id/dataset_name/dataset_type/source_run are Unicode arrays, shot_id is integer,
decoder_p is float64. true_lambda is also stored as an exact 12-character bit string.
Nonzero indices are compact ragged arrays: error_indices and error_offsets, likewise
syndrome_indices and syndrome_offsets. Row i uses indices[offsets[i]:offsets[i+1]].
All physical positions, syndrome positions and shot IDs are ZERO-based.
All arrays support np.load(...,allow_pickle=False); there are no object arrays.
numerical_cleanup_cases_readable.csv shows nonzero index lists and exact logical strings.
Read *_lambda and case_id/dataset_name as STRINGS; automatic spreadsheet numeric conversion
can discard leading zeros. CSV floats use round-trip Python representations.

Priority list
numerical_cleanup_priority.csv contains the top 100 rows, with priority_rank.
Ordering is lexicographic: sector mismatch first; within sector mismatches decreasing
probability_abs_diff; then probability_mismatch; then stress prior 0.005; then decreasing
physical weight; then decreasing probability_abs_diff and deterministic dataset/shot tie breaks.
This keeps large-gap sector disagreements first, then probability-only disagreements.
stress p=0.005 shot_id=827 is included with priority_example=1,
at priority rank 1.

Runtime files and limitations
source/ contains exactly the five existing project Python modules resolved by the runner's
static import chain. Their bytes are unchanged. No new research/replay code is bundled.
profiles/ contains both self-contained compiled path/slicing profiles, including all their
existing entries, not rewritten subsets. Original scan JSON, best config, checkpoint, run/data
manifests and PBS submission files are copied unchanged into manifests/.
Only the two required profile files are included; scan JSON may also describe other layouts.
Historical remote paths and original profile_path strings are intentionally preserved. Local
bundle profile locations are listed separately in profile_inventory.json.
No canonical pool pickle is required by load_fast_path_profile() for these self-contained files.

IMPORTANT PROVENANCE LIMITS:
The run stored paths/layout settings but did not record source/profile SHA-256 hashes or an
environment lock. Included sources are the CURRENT local files associated with that run
directory; historical byte-for-byte identity cannot be independently certified.
Current b9 profile metadata has full_reference_topology_added_after_scan=True. Besides its
two runtime BnB topologies it also contains a full 12-bit reference topology. That extra
entry's path/slicing equal the standalone b12 profile: True. It is preserved as-is;
the marker says 'after scan', and does not establish whether addition preceded/followed each
v0.5 validation run. The original runtime profile bytes were not separately archived.
Torch/GPU are logged; cuQuantum 26.3.2 is profile/scan metadata. The CUDA-QX commit in the
decoder header is a stated target, not proof of the installed runtime revision. Actual qldpc,
quimb, NumPy and CuPy versions and external installed package source snapshots are unavailable.
The project source imports cudaq_qec, quimb, cuquantum, cupy, torch, qldpc, sympy and numpy.
Those external packages are not included or substituted with potentially different versions.

Matrices
H_Z/L_Z/L_X were not exported: no saved run-linked arrays were found. See matrices_info.txt.
未导出，需要运行现有 code constructor 重建。No matrices were inferred or reimplemented.

What is not available
The databases contain no full posterior arrays, runner-up probabilities, per-node/per-call
positive-mass error bounds, or search traces. This bundle preserves actual recorded values
without inferring unrecorded diagnostics or declaring any mismatch a tie/pruning failure.

Integrity
All seven SQLite schemas compare identical; sqlite_schema.txt contains the shared DDL and
the list of checked databases. Frozen error/syndrome weights and true_lambda are verified
against both database rows for every selected case. Both priority and readable files refer
to case IDs in the primary CSV; selected NPZ rows are independently round-trip verified.
manifests/input_inventory.json records original file SHA-256 values. Sources are rehashed
after extraction to confirm no original file changed. bundle_files.sha256 hashes all other
bundle files. No original results.sqlite, full frozen dataset NPZ, or unrelated profiles
are included in the ZIP.
