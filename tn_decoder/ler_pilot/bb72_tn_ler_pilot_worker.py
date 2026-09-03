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

from common.bb72_baseline_common import build_problem
from common.bb72_tn_common import (
    build_cache_meta,
    enumerate_masks,
    load_path_cache,
    mask_to_bitstring,
    probe_torch_cuda,
    reconstruct_coset_probs,
    restore_cached_path,
    sector_index_from_bits,
    suppress_output,
)
from examples.bb72_cudaqx_nv_tn_flip_prob_example import (
    build_decoder,
    build_logical_observable,
    build_syndrome,
    logical_signature_in_basis,
    pick_optimize_arg,
    resolve_logical_basis,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_expected_cache(
    *,
    error_type: str,
    p: float,
    device: str,
    logical_basis_source_arg: str,
    suppress_cudaqx_logs: bool,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    problem = build_problem(error_type)
    h_matrix = np.asarray(problem.check_matrix, dtype=np.uint8)
    logical_basis, logical_basis_source = resolve_logical_basis(
        problem,
        error_type,
        logical_basis_source_arg,
    )
    noise_model = np.full(problem.n, float(p), dtype=np.float64)

    masks = enumerate_masks(problem.k)
    probe_logical_obs = build_logical_observable(logical_basis, masks[1])
    with suppress_output(suppress_cudaqx_logs):
        probe_decoder = build_decoder(h_matrix, probe_logical_obs, noise_model, device)

    meta = build_cache_meta(
        problem,
        h_matrix,
        logical_basis,
        error_type=error_type,
        logical_basis_source=logical_basis_source,
        contractor_name=str(probe_decoder.contractor_config.contractor_name),
    )
    runtime_info = {
        "actual_device": str(probe_decoder.contractor_config.device),
        "contractor": str(probe_decoder.contractor_config.contractor_name),
        "optimize_arg": pick_optimize_arg(probe_decoder),
        "logical_basis_source": logical_basis_source,
    }
    return problem, h_matrix, logical_basis, noise_model, masks, meta, runtime_info


def reconstruct_one_shot(
    *,
    problem: object,
    h_matrix: np.ndarray,
    logical_basis: np.ndarray,
    noise_model: np.ndarray,
    masks: np.ndarray,
    path_entries: dict[str, Any],
    device: str,
    p: float,
    seed: int,
    suppress_cudaqx_logs: bool,
) -> dict[str, Any]:
    shot_start = time.perf_counter()
    syndrome, sampled_error = build_syndrome(
        problem,
        "sample",
        None,
        p,
        seed,
    )
    if sampled_error is None:
        raise RuntimeError("sample mode should return the sampled error")

    flip_probabilities = np.zeros(1 << problem.k, dtype=np.float64)
    decode_seconds = 0.0
    cache_hits = 0

    for mask_index, logical_mask in enumerate(masks):
        if mask_index == 0:
            continue

        mask_bits = mask_to_bitstring(logical_mask)
        cache_entry = path_entries.get(mask_bits)
        if cache_entry is None:
            raise RuntimeError(f"Missing cached path for logical mask {mask_bits}")

        logical_obs = build_logical_observable(logical_basis, logical_mask)
        with suppress_output(suppress_cudaqx_logs):
            decoder = build_decoder(h_matrix, logical_obs, noise_model, device)
        restore_cached_path(decoder, cache_entry)

        decode_start = time.perf_counter()
        result = decoder.decode([float(x) for x in syndrome])
        decode_seconds += time.perf_counter() - decode_start

        flip_probabilities[mask_index] = float(
            np.asarray(result.result, dtype=np.float64).reshape(-1)[0]
        )
        cache_hits += 1

    reconstruct_start = time.perf_counter()
    coset_probabilities = reconstruct_coset_probs(flip_probabilities)
    reconstruct_seconds = time.perf_counter() - reconstruct_start

    map_sector_index = int(np.argmax(coset_probabilities))
    true_signature = logical_signature_in_basis(sampled_error, logical_basis)
    true_sector_index = sector_index_from_bits(true_signature)
    success = map_sector_index == true_sector_index
    total_seconds = time.perf_counter() - shot_start

    return {
        "seed": seed,
        "p": p,
        "syndrome_weight": int(np.count_nonzero(syndrome)),
        "error_weight": int(np.count_nonzero(sampled_error)),
        "true_error_support": str(np.flatnonzero(sampled_error).tolist()),
        "true_sector_index": true_sector_index,
        "true_sector_bits": mask_to_bitstring(masks[true_sector_index]),
        "map_sector_index": map_sector_index,
        "map_sector_bits": mask_to_bitstring(masks[map_sector_index]),
        "map_probability": float(coset_probabilities[map_sector_index]),
        "success": success,
        "failure": not success,
        "cache_hits": cache_hits,
        "decode_seconds": decode_seconds,
        "walsh_hadamard_seconds": reconstruct_seconds,
        "shot_total_seconds": total_seconds,
    }


def write_worker_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write an empty worker CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_worker_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One CUDA-QX BB72 TN LER pilot worker.")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--shots-per-worker", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.02)
    parser.add_argument("--base-seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--error-type", choices=["X", "Z"], default="Z")
    parser.add_argument(
        "--logical-basis-source",
        default="auto",
        choices=["auto", "qldpc-dual", "weight6-x"],
    )
    parser.add_argument("--path-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suppress-cudaqx-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress repeated CUDA-QX contractor setup logs during decoder construction.",
    )
    return parser.parse_args()


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id must be in [0, {args.num_workers - 1}]")
    if args.shots_per_worker <= 0:
        raise ValueError("--shots-per-worker must be positive")

    torch_info = probe_torch_cuda()
    (
        problem,
        h_matrix,
        logical_basis,
        noise_model,
        masks,
        expected_cache_meta,
        runtime_info,
    ) = build_expected_cache(
        error_type=args.error_type,
        p=args.p,
        device=args.device,
        logical_basis_source_arg=args.logical_basis_source,
        suppress_cudaqx_logs=args.suppress_cudaqx_logs,
    )

    path_cache, cache_status = load_path_cache(args.path_cache, expected_cache_meta)
    path_entries = path_cache["paths"]
    expected_entries = (1 << problem.k) - 1
    if cache_status["state"] != "loaded" or len(path_entries) != expected_entries:
        raise RuntimeError(
            "This pilot requires a complete existing path cache. "
            f"state={cache_status['state']} entries={len(path_entries)} "
            f"expected={expected_entries} path={args.path_cache}"
        )

    rows: list[dict[str, Any]] = []
    for local_shot_id in range(args.shots_per_worker):
        global_shot_id = args.worker_id * args.shots_per_worker + local_shot_id
        seed = args.base_seed + global_shot_id
        print(
            f"worker={args.worker_id} local_shot={local_shot_id} "
            f"global_shot={global_shot_id} seed={seed} start",
            flush=True,
        )
        shot_record = reconstruct_one_shot(
            problem=problem,
            h_matrix=h_matrix,
            logical_basis=logical_basis,
            noise_model=noise_model,
            masks=masks,
            path_entries=path_entries,
            device=args.device,
            p=args.p,
            seed=seed,
            suppress_cudaqx_logs=args.suppress_cudaqx_logs,
        )
        shot_record.update(
            {
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "local_shot_id": local_shot_id,
                "global_shot_id": global_shot_id,
                "hostname": platform.node(),
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                "actual_device": runtime_info["actual_device"],
                "contractor": runtime_info["contractor"],
                "logical_basis_source": runtime_info["logical_basis_source"],
                "path_cache": str(args.path_cache),
                "path_cache_entries_loaded": len(path_entries),
                "suppress_cudaqx_logs": args.suppress_cudaqx_logs,
                "started_at_utc": utc_now(),
            }
        )
        rows.append(shot_record)
        print(
            f"worker={args.worker_id} global_shot={global_shot_id} "
            f"success={shot_record['success']} true_sector={shot_record['true_sector_index']} "
            f"map_sector={shot_record['map_sector_index']} "
            f"seconds={shot_record['shot_total_seconds']:.3f}",
            flush=True,
        )

    output_dir = args.output_dir
    worker_csv = output_dir / f"worker_{args.worker_id:03d}.csv"
    worker_summary_json = output_dir / f"worker_{args.worker_id:03d}_summary.json"
    write_worker_csv(worker_csv, rows)

    failures = sum(1 for row in rows if bool(row["failure"]))
    summary: dict[str, Any] = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "shots": len(rows),
        "failures": failures,
        "logical_error_rate": failures / len(rows),
        "runtime_seconds": time.perf_counter() - total_start,
        "worker_csv": str(worker_csv),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "actual_device": runtime_info["actual_device"],
        "contractor": runtime_info["contractor"],
        "path_cache": str(args.path_cache),
        "path_cache_load_state": cache_status["state"],
        "path_cache_entries_loaded": len(path_entries),
        "suppress_cudaqx_logs": args.suppress_cudaqx_logs,
        "torch": torch_info,
    }
    write_worker_summary(worker_summary_json, summary)
    print(
        f"worker={args.worker_id} wrote_csv={worker_csv} "
        f"failures={failures}/{len(rows)} runtime_seconds={summary['runtime_seconds']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
