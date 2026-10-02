from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TN_DECODER_ROOT = SCRIPT_DIR.parent
if str(TN_DECODER_ROOT) not in sys.path:
    sys.path.insert(0, str(TN_DECODER_ROOT))

from common.bb72_tn_common import (
    build_cache_meta,
    enumerate_masks,
    fwht_inplace,
    load_path_cache,
    mask_to_bitstring,
    probe_torch_cuda,
    query_flip_probability_with_fallback,
    restore_cached_path,
    sector_index_from_bits,
    suppress_output,
)


DEFAULT_P_GRID = "0.005,0.01,0.02,0.03,0.04,0.05"
DEFAULT_PATH_CACHE = (
    TN_DECODER_ROOT
    / "tn_reconstruct"
    / "path_cache"
    / "bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_p_grid(value: str) -> list[float]:
    p_values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not p_values:
        raise ValueError("--p-grid must contain at least one physical error rate")
    return p_values


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = work.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if work[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(rows):
            if row != rank and work[row, col]:
                work[row] ^= work[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def import_cudaqx_helpers() -> dict[str, Any]:
    try:
        from common.bb72_baseline_common import build_problem
        from examples.bb72_cudaqx_nv_tn_flip_prob_example import (
            build_decoder,
            build_logical_observable,
            build_syndrome,
            logical_signature_in_basis,
            pick_optimize_arg,
            resolve_logical_basis,
        )
    except SystemExit as exc:
        message = str(exc) or repr(exc)
        raise RuntimeError(
            "Failed to import CUDA-QX example helpers. Run this debug script in the "
            "same Linux/container environment used for the TN LER sweep. "
            f"Original error: {message}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to import BB72/CUDA-QX dependencies. Check that cudaq_qec and qldpc "
            f"are installed in this Python environment. Original error: {exc!r}"
        ) from exc

    return {
        "build_problem": build_problem,
        "build_decoder": build_decoder,
        "build_logical_observable": build_logical_observable,
        "build_syndrome": build_syndrome,
        "logical_signature_in_basis": logical_signature_in_basis,
        "pick_optimize_arg": pick_optimize_arg,
        "resolve_logical_basis": resolve_logical_basis,
    }


def raw_wht_probabilities(flip_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moments = 1.0 - 2.0 * flip_probabilities.astype(np.float64, copy=False)
    probabilities = moments.copy()
    fwht_inplace(probabilities)
    probabilities /= probabilities.size
    return moments, probabilities


def task_parameters(
    *,
    global_task_id: int,
    p_values: list[float],
    shots_per_p: int,
    base_seed: int,
) -> dict[str, Any]:
    total_tasks = len(p_values) * shots_per_p
    if global_task_id < 0 or global_task_id >= total_tasks:
        raise ValueError(
            f"--global-task-id must be in [0, {total_tasks - 1}], got {global_task_id}"
        )
    p_index = global_task_id // shots_per_p
    shot_id_for_p = global_task_id % shots_per_p
    return {
        "global_task_id": global_task_id,
        "p_index": p_index,
        "p": p_values[p_index],
        "shot_id_for_p": shot_id_for_p,
        "seed": base_seed + p_index * 1_000_000 + shot_id_for_p,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise RuntimeError(f"Cannot infer CSV header for empty rows: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def support_string(vector: np.ndarray) -> str:
    return str(np.flatnonzero(vector).tolist())


def build_runtime_context(
    *,
    helpers: dict[str, Any],
    error_type: str,
    p: float,
    device: str,
    logical_basis_source_arg: str,
    suppress_cudaqx_logs: bool,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    problem = helpers["build_problem"](error_type)
    h_matrix = np.asarray(problem.check_matrix, dtype=np.uint8)
    logical_basis, logical_basis_source = helpers["resolve_logical_basis"](
        problem,
        error_type,
        logical_basis_source_arg,
    )
    logical_basis = np.asarray(logical_basis, dtype=np.uint8)
    noise_model = np.full(problem.n, float(p), dtype=np.float64)
    masks = enumerate_masks(problem.k)

    probe_logical_obs = helpers["build_logical_observable"](logical_basis, masks[1])
    with suppress_output(suppress_cudaqx_logs):
        probe_decoder = helpers["build_decoder"](h_matrix, probe_logical_obs, noise_model, device)

    runtime_info = {
        "actual_device": str(probe_decoder.contractor_config.device),
        "contractor": str(probe_decoder.contractor_config.contractor_name),
        "optimize_arg": helpers["pick_optimize_arg"](probe_decoder),
        "logical_basis_source": logical_basis_source,
        "logical_basis_rank_over_physical_bits": gf2_rank(logical_basis),
        "logical_basis_row_weights": np.count_nonzero(logical_basis, axis=1).astype(int).tolist(),
    }
    return problem, h_matrix, logical_basis, noise_model, masks, runtime_info


def query_all_flip_probabilities(
    *,
    helpers: dict[str, Any],
    problem: object,
    h_matrix: np.ndarray,
    logical_basis: np.ndarray,
    noise_model: np.ndarray,
    masks: np.ndarray,
    path_entries: dict[str, Any],
    device: str,
    syndrome: np.ndarray,
    suppress_cudaqx_logs: bool,
    progress_every: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float, int, dict[str, Any]]:
    flip_probabilities = np.zeros(1 << problem.k, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    decode_seconds = 0.0
    cache_hits = 0
    stats: dict[str, Any] = {
        "public_failure_count": 0,
        "raw_fallback_count": 0,
        "raw_clip_count": 0,
        "raw_fallback_seconds": 0.0,
        "max_raw_relative_negative_mass": 0.0,
    }

    rows.append(
        {
            "mask_index": 0,
            "mask_bits": mask_to_bitstring(masks[0]),
            "logical_obs_weight": 0,
            "flip_prob_p1": "0.000000000000000000e+00",
            "moment": "1.000000000000000000e+00",
            "decode_seconds": "0.000000",
            "path_cache_hit": False,
            "flip_prob_source": "implicit-zero",
            "public_valid": True,
            "public_converged": True,
            "public_p1": "0.000000000000000000e+00",
            "fallback_reason": "",
            "raw_z0": "",
            "raw_z1": "",
            "raw_p1": "",
            "raw_clipped_mass_count": 0,
            "raw_relative_negative_mass": "0.000000000000000000e+00",
            "raw_fallback_seconds": "0.000000",
            "logical_obs_support": "[]",
        }
    )

    for mask_index, logical_mask in enumerate(masks):
        if mask_index == 0:
            continue

        mask_bits = mask_to_bitstring(logical_mask)
        cache_entry = path_entries.get(mask_bits)
        if cache_entry is None:
            raise RuntimeError(f"Missing cached path for logical mask {mask_bits}")

        logical_obs = helpers["build_logical_observable"](logical_basis, logical_mask)
        with suppress_output(suppress_cudaqx_logs):
            decoder = helpers["build_decoder"](h_matrix, logical_obs, noise_model, device)
        restore_cached_path(decoder, cache_entry)

        query = query_flip_probability_with_fallback(
            decoder,
            syndrome,
            optimize_arg=None,
        )
        elapsed = float(query["query_decode_seconds"])
        decode_seconds += elapsed
        p1 = float(query["p1"])
        flip_probabilities[mask_index] = p1
        cache_hits += 1
        if not query["public_valid"]:
            stats["public_failure_count"] += 1
        if query["raw_used"]:
            stats["raw_fallback_count"] += 1
        stats["raw_clip_count"] += int(query["raw_clipped_mass_count"])
        stats["raw_fallback_seconds"] += float(query["raw_fallback_seconds"])
        stats["max_raw_relative_negative_mass"] = max(
            float(stats["max_raw_relative_negative_mass"]),
            float(query["raw_relative_negative_mass"]),
        )

        rows.append(
            {
                "mask_index": mask_index,
                "mask_bits": mask_bits,
                "logical_obs_weight": int(np.count_nonzero(logical_obs)),
                "flip_prob_p1": f"{p1:.18e}",
                "moment": f"{1.0 - 2.0 * p1:.18e}",
                "decode_seconds": f"{elapsed:.6f}",
                "path_cache_hit": True,
                "flip_prob_source": query["source"],
                "public_valid": query["public_valid"],
                "public_converged": query["public_converged"],
                "public_p1": (
                    "" if query["public_p1"] is None else f"{float(query['public_p1']):.18e}"
                ),
                "fallback_reason": query["fallback_reason"],
                "raw_z0": "" if query["raw_z0"] is None else f"{float(query['raw_z0']):.18e}",
                "raw_z1": "" if query["raw_z1"] is None else f"{float(query['raw_z1']):.18e}",
                "raw_p1": "" if query["raw_p1"] is None else f"{float(query['raw_p1']):.18e}",
                "raw_clipped_mass_count": query["raw_clipped_mass_count"],
                "raw_relative_negative_mass": f"{float(query['raw_relative_negative_mass']):.18e}",
                "raw_fallback_seconds": f"{float(query['raw_fallback_seconds']):.6f}",
                "logical_obs_support": str(np.flatnonzero(logical_obs.reshape(-1)).tolist()),
            }
        )

        if progress_every > 0 and mask_index % progress_every == 0:
            print(
                f"queried {mask_index}/{len(masks) - 1} masks "
                f"latest_mask={mask_bits} p1={p1:.12e} source={query['source']}",
                flush=True,
            )

    return flip_probabilities, rows, decode_seconds, cache_hits, stats


def build_sector_rows(
    *,
    masks: np.ndarray,
    probabilities: np.ndarray,
    true_sector_index: int,
    limit: int,
    descending: bool,
) -> list[dict[str, Any]]:
    order = np.argsort(probabilities)
    if descending:
        order = order[::-1]
    if limit > 0:
        order = order[:limit]

    rows: list[dict[str, Any]] = []
    for rank, sector_index in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "sector_index": int(sector_index),
                "sector_bits": mask_to_bitstring(masks[int(sector_index)]),
                "probability": f"{float(probabilities[int(sector_index)]):.18e}",
                "is_true_sector": int(sector_index) == true_sector_index,
            }
        )
    return rows


def build_negative_sector_rows(
    *,
    masks: np.ndarray,
    probabilities: np.ndarray,
    true_sector_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    negative_indices = np.flatnonzero(probabilities < 0.0)
    order = negative_indices[np.argsort(probabilities[negative_indices])]
    if limit > 0:
        order = order[:limit]

    rows: list[dict[str, Any]] = []
    for rank, sector_index in enumerate(order, start=1):
        rows.append(
            {
                "negative_rank": rank,
                "sector_index": int(sector_index),
                "sector_bits": mask_to_bitstring(masks[int(sector_index)]),
                "probability": f"{float(probabilities[int(sector_index)]):.18e}",
                "minus_one_over_num_sectors_error": f"{float(probabilities[int(sector_index)] + 1.0 / probabilities.size):.18e}",
                "is_true_sector": int(sector_index) == true_sector_index,
            }
        )
    return rows


def build_delta_deviation_rows(
    *,
    masks: np.ndarray,
    moments: np.ndarray,
    flip_probabilities: np.ndarray,
    true_signature: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    expected_parities = np.mod(masks @ true_signature, 2).astype(np.uint8)
    expected_delta_moments = np.where(expected_parities == 0, 1.0, -1.0)
    deviations = np.abs(moments - expected_delta_moments)
    order = np.argsort(deviations)[::-1]
    if limit > 0:
        order = order[:limit]

    rows: list[dict[str, Any]] = []
    for rank, mask_index in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "mask_index": int(mask_index),
                "mask_bits": mask_to_bitstring(masks[int(mask_index)]),
                "expected_parity_from_sampled_sector": int(expected_parities[int(mask_index)]),
                "flip_prob_p1": f"{float(flip_probabilities[int(mask_index)]):.18e}",
                "moment": f"{float(moments[int(mask_index)]):.18e}",
                "expected_delta_moment": f"{float(expected_delta_moments[int(mask_index)]):.18e}",
                "abs_deviation_from_delta_sector": f"{float(deviations[int(mask_index)]):.18e}",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug one BB72 CUDA-QX TN LER shot by saving all flip probabilities and "
            "the raw Walsh-Hadamard reconstructed sector probabilities."
        )
    )
    parser.add_argument("--global-task-id", type=int, default=174)
    parser.add_argument("--p-grid", default=DEFAULT_P_GRID)
    parser.add_argument("--shots-per-p", type=int, default=5000)
    parser.add_argument("--base-seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--error-type", choices=["X", "Z"], default="Z")
    parser.add_argument(
        "--logical-basis-source",
        default="auto",
        choices=["auto", "qldpc-dual", "weight6-x"],
    )
    parser.add_argument("--path-cache", type=Path, default=DEFAULT_PATH_CACHE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/debug_wht_task_<global-task-id>",
    )
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--top-sector-limit", type=int, default=32)
    parser.add_argument(
        "--negative-sector-limit",
        type=int,
        default=256,
        help="Use 0 to write all negative sectors.",
    )
    parser.add_argument("--delta-deviation-limit", type=int, default=256)
    parser.add_argument(
        "--suppress-cudaqx-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()
    p_values = parse_p_grid(args.p_grid)
    params = task_parameters(
        global_task_id=args.global_task_id,
        p_values=p_values,
        shots_per_p=args.shots_per_p,
        base_seed=args.base_seed,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = SCRIPT_DIR / "results" / f"debug_wht_task_{args.global_task_id:06d}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    helpers = import_cudaqx_helpers()
    (
        problem,
        h_matrix,
        logical_basis,
        noise_model,
        masks,
        runtime_info,
    ) = build_runtime_context(
        helpers=helpers,
        error_type=args.error_type,
        p=params["p"],
        device=args.device,
        logical_basis_source_arg=args.logical_basis_source,
        suppress_cudaqx_logs=args.suppress_cudaqx_logs,
    )

    expected_cache_meta = build_cache_meta(
        problem,
        h_matrix,
        logical_basis,
        error_type=args.error_type,
        logical_basis_source=runtime_info["logical_basis_source"],
        contractor_name=runtime_info["contractor"],
    )
    path_cache, cache_status = load_path_cache(args.path_cache.resolve(), expected_cache_meta)
    path_entries = path_cache["paths"]
    expected_entries = (1 << problem.k) - 1
    if cache_status["state"] != "loaded" or len(path_entries) != expected_entries:
        raise RuntimeError(
            "This debug script expects a complete path cache matching the active contractor. "
            f"state={cache_status['state']} entries={len(path_entries)} expected={expected_entries} "
            f"path={args.path_cache.resolve()} message={cache_status['message']}"
        )

    syndrome, sampled_error = helpers["build_syndrome"](
        problem,
        "sample",
        None,
        params["p"],
        params["seed"],
    )
    if sampled_error is None:
        raise RuntimeError("sample mode should return the sampled error")

    true_signature = helpers["logical_signature_in_basis"](sampled_error, logical_basis)
    true_sector_index = sector_index_from_bits(true_signature)

    print(
        f"debug task={args.global_task_id} p={params['p']} seed={params['seed']} "
        f"syndrome_weight={int(np.count_nonzero(syndrome))} "
        f"error_weight={int(np.count_nonzero(sampled_error))} "
        f"true_sector={true_sector_index}",
        flush=True,
    )

    query_start = time.perf_counter()
    (
        flip_probabilities,
        flip_rows,
        decode_seconds,
        cache_hits,
        query_stats,
    ) = query_all_flip_probabilities(
        helpers=helpers,
        problem=problem,
        h_matrix=h_matrix,
        logical_basis=logical_basis,
        noise_model=noise_model,
        masks=masks,
        path_entries=path_entries,
        device=args.device,
        syndrome=syndrome,
        suppress_cudaqx_logs=args.suppress_cudaqx_logs,
        progress_every=args.progress_every,
    )
    query_seconds = time.perf_counter() - query_start

    wht_start = time.perf_counter()
    moments, sector_probabilities = raw_wht_probabilities(flip_probabilities)
    wht_seconds = time.perf_counter() - wht_start

    sector_order = np.argsort(sector_probabilities)[::-1]
    true_rank = int(np.flatnonzero(sector_order == true_sector_index)[0]) + 1
    map_sector_index = int(sector_order[0])
    negative_indices = np.flatnonzero(sector_probabilities < 0.0)
    large_negative_indices = np.flatnonzero(sector_probabilities < -1e-10)

    invalid_flip_low = np.flatnonzero(flip_probabilities < -1e-12)
    invalid_flip_high = np.flatnonzero(flip_probabilities > 1.0 + 1e-12)
    near_half_indices = np.flatnonzero(np.abs(flip_probabilities - 0.5) < 1e-6)

    flip_csv = output_dir / "flip_probabilities.csv"
    sector_csv = output_dir / "sector_probabilities_raw_wht.csv"
    top_csv = output_dir / "top_sectors_raw_wht.csv"
    negative_csv = output_dir / "negative_sectors_raw_wht.csv"
    deviation_csv = output_dir / "delta_sector_moment_deviations.csv"
    arrays_npz = output_dir / "debug_arrays.npz"
    summary_json = output_dir / "debug_summary.json"

    write_csv(flip_csv, flip_rows)
    sector_rows = [
        {
            "sector_index": index,
            "sector_bits": mask_to_bitstring(masks[index]),
            "probability": f"{float(sector_probabilities[index]):.18e}",
            "is_true_sector": index == true_sector_index,
            "is_map_sector": index == map_sector_index,
        }
        for index in range(sector_probabilities.size)
    ]
    write_csv(sector_csv, sector_rows)
    write_csv(
        top_csv,
        build_sector_rows(
            masks=masks,
            probabilities=sector_probabilities,
            true_sector_index=true_sector_index,
            limit=args.top_sector_limit,
            descending=True,
        ),
    )
    write_csv(
        negative_csv,
        build_negative_sector_rows(
            masks=masks,
            probabilities=sector_probabilities,
            true_sector_index=true_sector_index,
            limit=args.negative_sector_limit,
        ),
    )
    write_csv(
        deviation_csv,
        build_delta_deviation_rows(
            masks=masks,
            moments=moments,
            flip_probabilities=flip_probabilities,
            true_signature=true_signature,
            limit=args.delta_deviation_limit,
        ),
    )
    np.savez_compressed(
        arrays_npz,
        flip_probabilities=flip_probabilities,
        moments=moments,
        sector_probabilities_raw_wht=sector_probabilities,
        syndrome=syndrome,
        sampled_error=sampled_error,
        true_signature=true_signature,
        logical_basis=logical_basis,
        masks=masks,
    )

    summary = {
        "created_at_utc": utc_now(),
        "task": params,
        "environment": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "torch": probe_torch_cuda(),
        },
        "runtime_info": runtime_info,
        "code": {
            "name": "[[72,12,6]]",
            "n": int(problem.n),
            "k": int(problem.k),
            "num_checks": int(problem.num_checks),
            "error_type": args.error_type,
            "logical_basis_source_arg": args.logical_basis_source,
        },
        "syndrome": {
            "weight": int(np.count_nonzero(syndrome)),
            "support": np.flatnonzero(syndrome).tolist(),
            "bits": mask_to_bitstring(syndrome),
        },
        "sampled_error": {
            "weight": int(np.count_nonzero(sampled_error)),
            "support": np.flatnonzero(sampled_error).tolist(),
        },
        "true_sector": {
            "index": true_sector_index,
            "bits": mask_to_bitstring(true_signature),
            "raw_wht_probability": float(sector_probabilities[true_sector_index]),
            "rank": true_rank,
        },
        "map_sector": {
            "index": map_sector_index,
            "bits": mask_to_bitstring(masks[map_sector_index]),
            "raw_wht_probability": float(sector_probabilities[map_sector_index]),
            "success": map_sector_index == true_sector_index,
        },
        "path_cache": {
            "file": args.path_cache.resolve(),
            "state": cache_status["state"],
            "message": cache_status["message"],
            "entries_loaded": len(path_entries),
            "expected_entries": expected_entries,
            "cache_hits": cache_hits,
        },
        "flip_probabilities": {
            "count": int(flip_probabilities.size),
            "min": float(np.min(flip_probabilities)),
            "max": float(np.max(flip_probabilities)),
            "invalid_low_count": int(invalid_flip_low.size),
            "invalid_high_count": int(invalid_flip_high.size),
            "near_half_count_abs_tol_1e_minus_6": int(near_half_indices.size),
        },
        "query_health": query_stats,
        "moments": {
            "min": float(np.min(moments)),
            "max": float(np.max(moments)),
        },
        "raw_wht": {
            "sum": float(np.sum(sector_probabilities)),
            "min_probability": float(np.min(sector_probabilities)),
            "max_probability": float(np.max(sector_probabilities)),
            "negative_count_lt_0": int(negative_indices.size),
            "large_negative_count_lt_minus_1e_minus_10": int(large_negative_indices.size),
            "negative_mass": float(-np.sum(sector_probabilities[negative_indices]))
            if negative_indices.size
            else 0.0,
            "one_over_num_sectors": float(1.0 / sector_probabilities.size),
            "most_negative_sector_index": int(np.argmin(sector_probabilities)),
            "most_negative_sector_bits": mask_to_bitstring(masks[int(np.argmin(sector_probabilities))]),
        },
        "timing_seconds": {
            "query_total": query_seconds,
            "decode_only_sum": decode_seconds,
            "walsh_hadamard": wht_seconds,
            "script_total": time.perf_counter() - total_start,
        },
        "outputs": {
            "flip_probabilities_csv": flip_csv,
            "sector_probabilities_raw_wht_csv": sector_csv,
            "top_sectors_raw_wht_csv": top_csv,
            "negative_sectors_raw_wht_csv": negative_csv,
            "delta_sector_moment_deviations_csv": deviation_csv,
            "arrays_npz": arrays_npz,
            "summary_json": summary_json,
        },
    }
    write_json(summary_json, summary)

    print(f"wrote_summary={summary_json}", flush=True)
    print(
        f"raw_wht_min={summary['raw_wht']['min_probability']:.18e} "
        f"negative_count={summary['raw_wht']['negative_count_lt_0']} "
        f"large_negative_count={summary['raw_wht']['large_negative_count_lt_minus_1e_minus_10']} "
        f"map_sector={map_sector_index} true_sector={true_sector_index} "
        f"success={map_sector_index == true_sector_index}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
