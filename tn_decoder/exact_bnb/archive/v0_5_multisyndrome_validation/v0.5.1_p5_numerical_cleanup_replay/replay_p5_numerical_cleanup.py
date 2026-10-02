from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def bits_to_str(bits: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in bits)


def str_to_bits(text: str) -> tuple[int, ...]:
    s = str(text).strip()
    if not s or any(ch not in "01" for ch in s):
        raise ValueError(f"invalid bit string {text!r}")
    return tuple(int(ch) for ch in s)


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        try:
            x = x.get()
        except Exception:
            pass
    return np.asarray(x, dtype=np.float64)


def sync_cuda() -> None:
    try:
        import cupy as cp
        cp.cuda.Device().synchronize()
        return
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def find_profile_entry(profile: Mapping[str, Any], key: tuple[int, ...]) -> Mapping[str, Any]:
    paths = profile.get("paths", profile)
    for raw_key, row in paths.items():
        if isinstance(raw_key, str):
            text = raw_key.strip().strip("()[]")
            parsed = tuple(int(x.strip()) for x in text.split(",") if x.strip())
        else:
            parsed = tuple(int(x) for x in raw_key)
        if parsed == key:
            return row
    raise KeyError(f"profile has no entry for open_logicals={key}")


def verify_profiles(b9_path: Path, b12_path: Path) -> dict[str, Any]:
    b9 = load_pickle(b9_path)
    b12 = load_pickle(b12_path)
    full_key = tuple(range(12))
    e9 = find_profile_entry(b9, full_key)
    e12 = find_profile_entry(b12, full_key)
    same_path = list(e9["path"]) == list(e12["path"])
    same_slices = list(e9.get("slices", ())) == list(e12.get("slices", ()))
    return {
        "b9_profile": str(b9_path),
        "b12_profile": str(b12_path),
        "full_path_equal": bool(same_path),
        "full_slices_equal": bool(same_slices),
        "b9_metadata": dict(b9.get("metadata", {})),
        "b12_metadata": dict(b12.get("metadata", {})),
    }


@dataclass
class TraceCall:
    fixed: dict[int, int]
    open_logicals: tuple[int, ...]
    raw: np.ndarray
    elapsed_seconds: float


class TracedOracle:
    def __init__(
        self,
        *,
        decoder: Any,
        syndrome: np.ndarray,
        memory_limit: str,
        constant_cache: bool,
        gpu_resident_operands: bool,
    ) -> None:
        self.decoder = decoder
        self.syndrome = syndrome.astype(float).tolist()
        self.memory_limit = memory_limit
        self.constant_cache = bool(constant_cache)
        self.gpu_resident_operands = bool(gpu_resident_operands)
        self.calls: list[TraceCall] = []

    def __call__(self, fixed_logicals: Mapping[int, int], open_logicals: Sequence[int]) -> Any:
        fixed = {int(k): int(v) for k, v in fixed_logicals.items()}
        opened = tuple(int(x) for x in open_logicals)
        sync_cuda()
        t0 = time.perf_counter()
        out = self.decoder.contract_logical_mass_fast(
            syndrome=self.syndrome,
            fixed_logicals=fixed,
            open_logicals=list(opened),
            autotune_iterations=0,
            memory_limit=self.memory_limit,
            release_workspace=True,
            use_constant_qualifiers=self.constant_cache,
            use_gpu_resident_operands=self.gpu_resident_operands,
            retain_constant_cache=self.constant_cache,
            retain_scratch_workspace=False,
        )
        sync_cuda()
        dt = time.perf_counter() - t0
        raw = to_numpy(out).copy()
        self.calls.append(TraceCall(fixed=fixed, open_logicals=opened, raw=raw, elapsed_seconds=dt))
        return out


def contract_full_reference(
    *, decoder: Any, syndrome: np.ndarray, memory_limit: str,
    constant_cache: bool, gpu_resident_operands: bool,
) -> tuple[np.ndarray, float]:
    sync_cuda()
    t0 = time.perf_counter()
    out = decoder.contract_logical_mass_fast(
        syndrome=syndrome.astype(float).tolist(),
        fixed_logicals={},
        open_logicals=list(range(12)),
        autotune_iterations=0,
        memory_limit=memory_limit,
        release_workspace=True,
        use_constant_qualifiers=constant_cache,
        use_gpu_resident_operands=gpu_resident_operands,
        retain_constant_cache=constant_cache,
        retain_scratch_workspace=False,
    )
    sync_cuda()
    return to_numpy(out).copy(), time.perf_counter() - t0


def numerical_stats(arr: np.ndarray, root_for_guard: float | None, *, guard_rtol: float, guard_atol: float,
                    negative_guard_factor: float) -> dict[str, float | int]:
    a = np.asarray(arr, dtype=np.float64)
    min_raw = float(a.min(initial=0.0))
    neg_scale = max(0.0, -min_raw)
    if root_for_guard is None:
        root_for_guard = float(a.sum())
    base_guard = max(float(guard_atol), float(guard_rtol) * abs(float(root_for_guard)))
    local_guard = max(base_guard, float(negative_guard_factor) * neg_scale)
    return {
        "min_raw": min_raw,
        "negative_count": int(np.count_nonzero(a < 0.0)),
        "negative_scale": neg_scale,
        "base_guard_abs": base_guard,
        "local_guard_abs": local_guard,
    }


def norm_gap(a: float, b: float, denom: float) -> float:
    d = abs(float(denom))
    return float(a - b) / d if d > 0 else float("nan")


def max_abs_normalized(a: np.ndarray, b: np.ndarray, denom: float) -> tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    ma = float(delta.max(initial=0.0))
    d = abs(float(denom))
    return ma, (ma / d if d > 0 else float("nan"))


def create_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS replay (
            case_id TEXT PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            decoder_p REAL NOT NULL,
            shot_id INTEGER NOT NULL,
            physical_error_weight INTEGER NOT NULL,
            syndrome_weight INTEGER NOT NULL,
            true_lambda TEXT NOT NULL,
            original_b9_lambda TEXT NOT NULL,
            original_b12_lambda TEXT NOT NULL,
            original_probability_abs_diff REAL NOT NULL,
            replay_b9_lambda TEXT NOT NULL,
            replay_b9_probability REAL NOT NULL,
            replay_b9_root_mass REAL NOT NULL,
            replay_b9_oracle_calls INTEGER NOT NULL,
            replay_b9_wall_seconds REAL NOT NULL,
            replay_b9_clipped_negative_entries INTEGER NOT NULL,
            replay_b9_most_negative_raw_mass REAL NOT NULL,
            replay_b9_max_local_guard_abs REAL NOT NULL,
            full_ref_lambda TEXT NOT NULL,
            full_ref_probability_rawnorm REAL NOT NULL,
            full_ref_probability_clipnorm REAL NOT NULL,
            full_ref_root_raw REAL NOT NULL,
            full_ref_root_clipped REAL NOT NULL,
            full_ref_contract_seconds REAL NOT NULL,
            full_ref_negative_count INTEGER NOT NULL,
            full_ref_min_raw REAL NOT NULL,
            full_ref_negative_scale REAL NOT NULL,
            full_ref_local_guard_abs REAL NOT NULL,
            original_b9_prob_common_ref REAL NOT NULL,
            original_b12_prob_common_ref REAL NOT NULL,
            replay_b9_prob_common_ref REAL NOT NULL,
            true_lambda_prob_common_ref REAL NOT NULL,
            gap_ref_vs_replay_b9_prob REAL NOT NULL,
            gap_ref_vs_original_b9_prob REAL NOT NULL,
            gap_ref_vs_original_b12_prob REAL NOT NULL,
            ref_prefix_explored INTEGER NOT NULL,
            b9_ref_prefix_bound_raw REAL NOT NULL,
            b9_ref_prefix_bound_prob REAL NOT NULL,
            full_ref_prefix_sum_raw REAL NOT NULL,
            full_ref_prefix_sum_prob REAL NOT NULL,
            ref_prefix_bound_delta_prob REAL NOT NULL,
            b9_root_vs_full_marginal_max_abs REAL NOT NULL,
            b9_root_vs_full_marginal_max_abs_norm REAL NOT NULL,
            b9_tail_vs_full_slice_max_abs REAL,
            b9_tail_vs_full_slice_max_abs_norm REAL,
            b9_ref_tail_mass_raw REAL,
            full_ref_leaf_mass_raw REAL NOT NULL,
            classification TEXT NOT NULL,
            reference_stressed INTEGER NOT NULL,
            matrix_syndrome_ok INTEGER NOT NULL,
            matrix_true_lambda_ok INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


REPLAY_COLUMNS = [
    "case_id", "dataset_name", "dataset_type", "decoder_p", "shot_id",
    "physical_error_weight", "syndrome_weight", "true_lambda",
    "original_b9_lambda", "original_b12_lambda", "original_probability_abs_diff",
    "replay_b9_lambda", "replay_b9_probability", "replay_b9_root_mass",
    "replay_b9_oracle_calls", "replay_b9_wall_seconds",
    "replay_b9_clipped_negative_entries", "replay_b9_most_negative_raw_mass",
    "replay_b9_max_local_guard_abs", "full_ref_lambda",
    "full_ref_probability_rawnorm", "full_ref_probability_clipnorm",
    "full_ref_root_raw", "full_ref_root_clipped", "full_ref_contract_seconds",
    "full_ref_negative_count", "full_ref_min_raw", "full_ref_negative_scale",
    "full_ref_local_guard_abs", "original_b9_prob_common_ref",
    "original_b12_prob_common_ref", "replay_b9_prob_common_ref",
    "true_lambda_prob_common_ref", "gap_ref_vs_replay_b9_prob",
    "gap_ref_vs_original_b9_prob", "gap_ref_vs_original_b12_prob",
    "ref_prefix_explored", "b9_ref_prefix_bound_raw", "b9_ref_prefix_bound_prob",
    "full_ref_prefix_sum_raw", "full_ref_prefix_sum_prob", "ref_prefix_bound_delta_prob",
    "b9_root_vs_full_marginal_max_abs", "b9_root_vs_full_marginal_max_abs_norm",
    "b9_tail_vs_full_slice_max_abs", "b9_tail_vs_full_slice_max_abs_norm",
    "b9_ref_tail_mass_raw", "full_ref_leaf_mass_raw", "classification",
    "reference_stressed", "matrix_syndrome_ok", "matrix_true_lambda_ok",
]


def insert_row(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    cols = REPLAY_COLUMNS
    vals = [row.get(c) for c in cols]
    q = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT OR REPLACE INTO replay ({','.join(cols)}) VALUES ({q})",
        vals,
    )
    conn.commit()


def write_csv_from_db(conn: sqlite3.Connection, path: Path) -> None:
    cur = conn.execute(
        "SELECT " + ",".join(REPLAY_COLUMNS) + " FROM replay ORDER BY decoder_p, dataset_name, shot_id"
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(REPLAY_COLUMNS)
        w.writerows(cur.fetchall())


def summarize(conn: sqlite3.Connection, outdir: Path) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT classification, COUNT(*), MAX(gap_ref_vs_replay_b9_prob), "
        "MAX(b9_root_vs_full_marginal_max_abs_norm), MAX(COALESCE(b9_tail_vs_full_slice_max_abs_norm,0)), "
        "SUM(reference_stressed) FROM replay GROUP BY classification ORDER BY COUNT(*) DESC"
    ).fetchall()
    by_class = [
        {
            "classification": r[0], "count": int(r[1]),
            "max_gap_ref_vs_replay_b9_prob": float(r[2]) if r[2] is not None else None,
            "max_root_marginal_delta_norm": float(r[3]) if r[3] is not None else None,
            "max_tail_delta_norm": float(r[4]) if r[4] is not None else None,
            "reference_stressed_count": int(r[5] or 0),
        }
        for r in rows
    ]
    total = int(conn.execute("SELECT COUNT(*) FROM replay").fetchone()[0])
    significant = int(conn.execute(
        "SELECT COUNT(*) FROM replay WHERE gap_ref_vs_replay_b9_prob > 1e-10"
    ).fetchone()[0])
    pruned = int(conn.execute(
        "SELECT COUNT(*) FROM replay WHERE classification LIKE 'REF_PREFIX_PRUNED%'"
    ).fetchone()[0])
    summary = {"total_replayed": total, "significant_common_reference_gap_gt_1e-10": significant,
               "reference_prefix_pruned_cases": pruned, "by_classification": by_class}
    (outdir / "replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Highest-risk rows for the next stage.
    cur = conn.execute(
        "SELECT " + ",".join(REPLAY_COLUMNS) + " FROM replay "
        "ORDER BY gap_ref_vs_replay_b9_prob DESC, reference_stressed DESC, decoder_p ASC LIMIT 100"
    )
    with (outdir / "replay_priority_top100.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(REPLAY_COLUMNS); w.writerows(cur.fetchall())
    return summary


def load_case_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def npz_case_map(npz_path: Path) -> dict[str, int]:
    with np.load(npz_path, allow_pickle=False) as z:
        return {str(cid): i for i, cid in enumerate(z["case_id"].tolist())}


def load_case_arrays(npz_path: Path, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        return z["errors"][idx].astype(np.uint8), z["syndromes"][idx].astype(np.uint8), z["true_lambdas"][idx].astype(np.uint8)


def save_case_arrays(path: Path, *, full_raw: np.ndarray, root_raw: np.ndarray, trace_calls: list[TraceCall],
                     case_meta: Mapping[str, Any]) -> None:
    prefixes = []
    tails = []
    tail_open = []
    for call in trace_calls[1:]:
        if call.open_logicals == (9, 10, 11) and len(call.fixed) == 9:
            prefixes.append([int(call.fixed[i]) for i in range(9)])
            tails.append(call.raw.astype(np.float64))
            tail_open.append(list(call.open_logicals))
    if prefixes:
        pfx = np.asarray(prefixes, dtype=np.uint8)
        tail_arr = np.asarray(tails, dtype=np.float64)
    else:
        pfx = np.empty((0, 9), dtype=np.uint8)
        tail_arr = np.empty((0, 2, 2, 2), dtype=np.float64)
    np.savez_compressed(
        path,
        full_raw=np.asarray(full_raw, dtype=np.float64),
        b9_root_raw=np.asarray(root_raw, dtype=np.float64),
        tail_prefixes=pfx,
        tail_raws=tail_arr,
        case_meta_json=np.asarray(json.dumps(dict(case_meta), sort_keys=True)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay BB72 v0.5 b9-vs-b12 numerical mismatches on one common decoder/reference tensor")
    ap.add_argument("--input-bundle", default=str(ROOT / "input_bundle"))
    ap.add_argument("--output-dir", default="p5_numerical_cleanup_replay")
    ap.add_argument("--selection", choices=["all", "priority"], default="all")
    ap.add_argument("--priority-limit", type=int, default=100)
    ap.add_argument("--case-id", action="append", default=[], help="Optional exact case_id; may be repeated. Overrides --selection.")
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--guard-atol", type=float, default=0.0)
    ap.add_argument("--negative-guard-factor", type=float, default=16.0)
    ap.add_argument("--tie-tol-prob", type=float, default=1e-10)
    ap.add_argument("--reference-stress-rtol", type=float, default=1e-10,
                    help="Flag full reference if negative cancellation scale/root exceeds this value.")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--save-arrays", choices=["all", "priority", "none"], default="all")
    ap.add_argument("--no-constant-cache", dest="constant_cache", action="store_false")
    ap.add_argument("--no-gpu-resident-operands", dest="gpu_resident_operands", action="store_false")
    ap.set_defaults(constant_cache=True, gpu_resident_operands=True)
    args = ap.parse_args()

    input_dir = Path(args.input_bundle).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    arrays_dir = outdir / "case_arrays"
    arrays_dir.mkdir(exist_ok=True)

    cases_csv = input_dir / "numerical_cleanup_cases.csv"
    priority_csv = input_dir / "numerical_cleanup_priority.csv"
    cases_npz = input_dir / "numerical_cleanup_cases.npz"
    b9_profile = input_dir / "profiles" / "canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl"
    b12_profile = input_dir / "profiles" / "canonical_paths_b12_0-1-2-3-4-5-6-7-8-9-10-11.pkl"

    profile_check = verify_profiles(b9_profile, b12_profile)
    if not (profile_check["full_path_equal"] and profile_check["full_slices_equal"]):
        raise RuntimeError("b9 embedded full topology path/slicing does not match standalone b12 profile")
    (outdir / "profile_verification.json").write_text(json.dumps(profile_check, indent=2, sort_keys=True) + "\n")

    all_rows = load_case_rows(cases_csv)
    by_id = {r["case_id"]: r for r in all_rows}
    priority_ids = [r["case_id"] for r in load_case_rows(priority_csv)]
    if args.case_id:
        missing = [x for x in args.case_id if x not in by_id]
        if missing:
            raise KeyError(f"unknown case_id(s): {missing}")
        selected = [by_id[x] for x in args.case_id]
    elif args.selection == "priority":
        selected = [by_id[x] for x in priority_ids[: max(0, args.priority_limit)]]
    else:
        selected = list(all_rows)

    # Deterministic order: by p, dataset, shot, so a decoder can be reused per p.
    selected.sort(key=lambda r: (float(r["decoder_p"]), r["dataset_name"], int(r["shot_id"])))
    npz_map = npz_case_map(cases_npz)

    # Heavy imports are delayed so the package can still be inspected/compiled without CUDA-QX.
    from exact_block_bnb import ExactBlockBnB
    from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder
    from run_bb72_multisyndrome_validation import build_bb72

    H, LZ, LX = build_bb72()
    if H.shape != (36, 72) or LZ.shape != (12, 72) or LX.shape != (12, 72):
        raise RuntimeError(f"unexpected BB72 matrix shapes H={H.shape}, LZ={LZ.shape}, LX={LX.shape}")
    if not np.array_equal((LX @ LZ.T) % 2, np.eye(12, dtype=np.uint8)):
        raise RuntimeError("reconstructed logical basis failed LX @ LZ.T == I")

    # Strong basis/provenance check before spending GPU time: all 490 extracted rows must match.
    basis_errors = []
    for r in all_rows:
        i = npz_map[r["case_id"]]
        err, syn, true_lam = load_case_arrays(cases_npz, i)
        syn_calc = (H @ err) % 2
        lam_calc = (LZ @ err) % 2
        if not np.array_equal(syn_calc.astype(np.uint8), syn):
            basis_errors.append((r["case_id"], "syndrome"))
        if not np.array_equal(lam_calc.astype(np.uint8), true_lam):
            basis_errors.append((r["case_id"], "true_lambda"))
    if basis_errors:
        sample = basis_errors[:10]
        raise RuntimeError(
            "Rebuilt qldpc BB72 basis does not reproduce frozen dataset. "
            f"First failures: {sample}. Abort rather than replay a different basis."
        )
    np.save(outdir / "H_Z_rebuilt.npy", H.astype(np.uint8))
    np.save(outdir / "L_Z_rebuilt.npy", LZ.astype(np.uint8))
    np.save(outdir / "L_X_rebuilt.npy", LX.astype(np.uint8))
    (outdir / "basis_verification.json").write_text(json.dumps({
        "all_extracted_cases_checked": len(all_rows),
        "all_syndromes_match": True,
        "all_true_lambdas_match": True,
        "H_Z_shape": list(H.shape), "L_Z_shape": list(LZ.shape), "L_X_shape": list(LX.shape),
        "LX_LZ_symplectic_identity": True,
    }, indent=2, sort_keys=True) + "\n")

    db = create_db(outdir / "replay.sqlite")
    completed = {r[0] for r in db.execute("SELECT case_id FROM replay").fetchall()}
    priority_set = set(priority_ids[: max(0, args.priority_limit)])

    current_p: float | None = None
    decoder = None
    new_count = 0
    t_all = time.perf_counter()

    for pos, r in enumerate(selected, 1):
        cid = r["case_id"]
        if cid in completed:
            continue
        p = float(r["decoder_p"])
        idx = npz_map[cid]
        err, syndrome, true_lam_arr = load_case_arrays(cases_npz, idx)
        true_lambda = bits_to_str(true_lam_arr)
        syn_ok = int(np.array_equal(((H @ err) % 2).astype(np.uint8), syndrome))
        lam_ok = int(np.array_equal(((LZ @ err) % 2).astype(np.uint8), true_lam_arr))
        if not (syn_ok and lam_ok):
            raise RuntimeError(f"basis validation unexpectedly failed for {cid}")

        if current_p is None or abs(p - current_p) > 1e-15:
            if decoder is not None:
                try:
                    decoder.clear_fast_network_cache()
                except Exception:
                    pass
                del decoder
                cleanup_gpu()
            decoder = MultiLogicalTensorNetworkDecoder(
                H=H, logical_obs=LZ, noise_model=[p] * int(H.shape[1]), device="cuda"
            )
            decoder.set_fast_cache_max_entries(8)
            decoder.load_fast_path_profile(b9_profile)
            current_p = p
            print(f"\n=== decoder p={p:g} ===")

        traced = TracedOracle(
            decoder=decoder,
            syndrome=syndrome,
            memory_limit=args.memory_limit,
            constant_cache=args.constant_cache,
            gpu_resident_operands=args.gpu_resident_operands,
        )
        bnb = ExactBlockBnB(
            k=12, block_size=9, oracle=traced, logical_order=list(range(12)),
            guard_rtol=args.guard_rtol, guard_atol=args.guard_atol,
            negative_guard_factor=args.negative_guard_factor, print_tree=False,
        )
        b9_result = bnb.search()

        full_raw, full_seconds = contract_full_reference(
            decoder=decoder, syndrome=syndrome, memory_limit=args.memory_limit,
            constant_cache=args.constant_cache,
            gpu_resident_operands=args.gpu_resident_operands,
        )
        if full_raw.shape != (2,) * 12:
            raise RuntimeError(f"full reference shape {full_raw.shape}, expected {(2,) * 12}")

        full_root_raw = float(full_raw.sum())
        if not np.isfinite(full_root_raw) or full_root_raw <= 0.0:
            raise FloatingPointError(f"{cid}: invalid full root mass {full_root_raw}")
        full_stats = numerical_stats(
            full_raw, full_root_raw, guard_rtol=args.guard_rtol, guard_atol=args.guard_atol,
            negative_guard_factor=args.negative_guard_factor,
        )
        full_clip = np.maximum(full_raw, 0.0)
        full_root_clip = float(full_clip.sum())
        ref_idx = tuple(int(x) for x in np.unravel_index(int(np.argmax(full_clip)), full_clip.shape))
        ref_lambda = bits_to_str(ref_idx)
        ref_best_mass = float(full_clip[ref_idx])
        ref_prob_raw = ref_best_mass / full_root_raw
        ref_prob_clip = ref_best_mass / full_root_clip if full_root_clip > 0 else float("nan")

        original_b9 = str_to_bits(r["b9_best_lambda"])
        original_b12 = str_to_bits(r["b12_best_lambda"])
        replay_b9 = tuple(int(x) for x in b9_result.best_lambda)
        true_idx = tuple(int(x) for x in true_lam_arr.tolist())

        def full_prob(bits: tuple[int, ...]) -> float:
            return float(full_clip[bits]) / full_root_raw

        p_orig_b9 = full_prob(original_b9)
        p_orig_b12 = full_prob(original_b12)
        p_replay_b9 = full_prob(replay_b9)
        p_true = full_prob(true_idx)
        gap_replay = ref_prob_raw - p_replay_b9
        gap_orig_b9 = ref_prob_raw - p_orig_b9
        gap_orig_b12 = ref_prob_raw - p_orig_b12

        if not traced.calls or traced.calls[0].open_logicals != tuple(range(9)) or traced.calls[0].fixed:
            raise RuntimeError(f"{cid}: unexpected first b9 oracle call")
        root_raw = traced.calls[0].raw
        if root_raw.shape != (2,) * 9:
            raise RuntimeError(f"{cid}: b9 root shape {root_raw.shape}")
        b9_root_mass = float(root_raw.sum())
        root_stats = numerical_stats(
            root_raw, b9_root_mass, guard_rtol=args.guard_rtol, guard_atol=args.guard_atol,
            negative_guard_factor=args.negative_guard_factor,
        )

        # The same full tensor, marginalized over the last 3 logical axes, is a common-reference
        # expectation for the 9-bit root oracle. This pinpoints cross-topology cancellation error.
        full_marg9_raw = full_raw.sum(axis=(9, 10, 11))
        root_delta_abs, root_delta_norm = max_abs_normalized(root_raw, full_marg9_raw, full_root_raw)

        ref_prefix = ref_idx[:9]
        ref_tail = ref_idx[9:]
        b9_ref_prefix_raw = float(root_raw[ref_prefix])
        b9_ref_prefix_prob = b9_ref_prefix_raw / b9_root_mass
        full_ref_prefix_raw = float(full_marg9_raw[ref_prefix])
        full_ref_prefix_prob = full_ref_prefix_raw / full_root_raw
        prefix_delta_prob = b9_ref_prefix_prob - full_ref_prefix_prob

        ref_tail_call: TraceCall | None = None
        tail_delta_abs = None
        tail_delta_norm = None
        b9_ref_tail_mass_raw = None
        for call in traced.calls[1:]:
            if call.open_logicals != (9, 10, 11) or len(call.fixed) != 9:
                continue
            prefix = tuple(int(call.fixed[i]) for i in range(9))
            full_slice = full_raw[prefix]
            da, dn = max_abs_normalized(call.raw, full_slice, full_root_raw)
            if prefix == ref_prefix:
                ref_tail_call = call
                tail_delta_abs, tail_delta_norm = da, dn
                b9_ref_tail_mass_raw = float(call.raw[ref_tail])

        ref_prefix_explored = int(ref_tail_call is not None)
        ref_stressed = int(
            float(full_stats["negative_scale"]) / abs(full_root_raw) > float(args.reference_stress_rtol)
        )

        if replay_b9 == ref_idx:
            classification = "AGREE_REPLAY"
        elif gap_replay <= float(args.tie_tol_prob):
            classification = "NEAR_TIE_COMMON_FULL"
        elif not ref_prefix_explored:
            classification = "REF_PREFIX_PRUNED__CHECK_ROOT_BOUND"
        else:
            classification = "REF_PREFIX_EXPLORED__TAIL_NUMERICAL_DISAGREEMENT"
        if ref_stressed and classification != "AGREE_REPLAY":
            classification = "FULL_REFERENCE_STRESSED__" + classification

        row = {
            "case_id": cid,
            "dataset_name": r["dataset_name"],
            "dataset_type": r["dataset_type"],
            "decoder_p": p,
            "shot_id": int(r["shot_id"]),
            "physical_error_weight": int(r["physical_error_weight"]),
            "syndrome_weight": int(r["syndrome_weight"]),
            "true_lambda": true_lambda,
            "original_b9_lambda": r["b9_best_lambda"],
            "original_b12_lambda": r["b12_best_lambda"],
            "original_probability_abs_diff": float(r["probability_abs_diff"]),
            "replay_b9_lambda": bits_to_str(replay_b9),
            "replay_b9_probability": float(b9_result.best_probability),
            "replay_b9_root_mass": float(b9_result.root_mass),
            "replay_b9_oracle_calls": int(b9_result.stats.oracle_calls),
            "replay_b9_wall_seconds": float(b9_result.stats.wall_seconds),
            "replay_b9_clipped_negative_entries": int(b9_result.stats.clipped_negative_entries),
            "replay_b9_most_negative_raw_mass": float(b9_result.stats.most_negative_raw_mass),
            "replay_b9_max_local_guard_abs": float(b9_result.stats.max_local_guard_abs),
            "full_ref_lambda": ref_lambda,
            "full_ref_probability_rawnorm": float(ref_prob_raw),
            "full_ref_probability_clipnorm": float(ref_prob_clip),
            "full_ref_root_raw": full_root_raw,
            "full_ref_root_clipped": full_root_clip,
            "full_ref_contract_seconds": float(full_seconds),
            "full_ref_negative_count": int(full_stats["negative_count"]),
            "full_ref_min_raw": float(full_stats["min_raw"]),
            "full_ref_negative_scale": float(full_stats["negative_scale"]),
            "full_ref_local_guard_abs": float(full_stats["local_guard_abs"]),
            "original_b9_prob_common_ref": float(p_orig_b9),
            "original_b12_prob_common_ref": float(p_orig_b12),
            "replay_b9_prob_common_ref": float(p_replay_b9),
            "true_lambda_prob_common_ref": float(p_true),
            "gap_ref_vs_replay_b9_prob": float(gap_replay),
            "gap_ref_vs_original_b9_prob": float(gap_orig_b9),
            "gap_ref_vs_original_b12_prob": float(gap_orig_b12),
            "ref_prefix_explored": ref_prefix_explored,
            "b9_ref_prefix_bound_raw": b9_ref_prefix_raw,
            "b9_ref_prefix_bound_prob": b9_ref_prefix_prob,
            "full_ref_prefix_sum_raw": full_ref_prefix_raw,
            "full_ref_prefix_sum_prob": full_ref_prefix_prob,
            "ref_prefix_bound_delta_prob": float(prefix_delta_prob),
            "b9_root_vs_full_marginal_max_abs": float(root_delta_abs),
            "b9_root_vs_full_marginal_max_abs_norm": float(root_delta_norm),
            "b9_tail_vs_full_slice_max_abs": None if tail_delta_abs is None else float(tail_delta_abs),
            "b9_tail_vs_full_slice_max_abs_norm": None if tail_delta_norm is None else float(tail_delta_norm),
            "b9_ref_tail_mass_raw": None if b9_ref_tail_mass_raw is None else float(b9_ref_tail_mass_raw),
            "full_ref_leaf_mass_raw": float(full_clip[ref_idx]),
            "classification": classification,
            "reference_stressed": ref_stressed,
            "matrix_syndrome_ok": syn_ok,
            "matrix_true_lambda_ok": lam_ok,
        }
        insert_row(db, row)
        new_count += 1

        save_arrays = args.save_arrays == "all" or (args.save_arrays == "priority" and cid in priority_set)
        if save_arrays:
            save_case_arrays(
                arrays_dir / f"{cid}.npz", full_raw=full_raw, root_raw=root_raw,
                trace_calls=traced.calls,
                case_meta={"case_id": cid, "decoder_p": p, "classification": classification,
                           "full_ref_lambda": ref_lambda, "replay_b9_lambda": bits_to_str(replay_b9)},
            )

        if new_count % max(1, args.progress_every) == 0 or pos == len(selected):
            elapsed = time.perf_counter() - t_all
            print(
                f"[{pos}/{len(selected)}] new={new_count} p={p:g} {cid} "
                f"class={classification} gap={gap_replay:.3e} calls={b9_result.stats.oracle_calls} "
                f"elapsed={elapsed/60:.1f} min"
            )

    write_csv_from_db(db, outdir / "replay_results.csv")
    summary = summarize(db, outdir)
    print("\nReplay complete/resumed.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("results:", outdir / "replay_results.csv")
    print("summary:", outdir / "replay_summary.json")


if __name__ == "__main__":
    main()
