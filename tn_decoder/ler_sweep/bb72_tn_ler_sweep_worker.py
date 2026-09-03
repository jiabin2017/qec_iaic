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


def parse_p_grid(value: str) -> list[float]:
    p_values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not p_values:
        raise ValueError("--p-grid must contain at least one physical error rate")
    return p_values


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


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
    syndrome, sampled_error = build_syndrome(problem, "sample", None, p, seed)
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
    true_sector_probability = float(coset_probabilities[true_sector_index])
    sorted_indices = np.argsort(coset_probabilities)[::-1]
    true_rank = int(np.flatnonzero(sorted_indices == true_sector_index)[0]) + 1
    top1_probability = float(coset_probabilities[sorted_indices[0]])
    top2_probability = float(coset_probabilities[sorted_indices[1]])
    success = map_sector_index == true_sector_index

    return {
        "seed": seed,
        "p": p,
        "syndrome_weight": int(np.count_nonzero(syndrome)),
        "error_weight": int(np.count_nonzero(sampled_error)),
        "true_error_support": str(np.flatnonzero(sampled_error).tolist()),
        "true_sector_index": true_sector_index,
        "true_sector_bits": mask_to_bitstring(masks[true_sector_index]),
        "true_sector_probability": true_sector_probability,
        "true_sector_rank": true_rank,
        "map_sector_index": map_sector_index,
        "map_sector_bits": mask_to_bitstring(masks[map_sector_index]),
        "map_probability": top1_probability,
        "top2_probability": top2_probability,
        "top1_top2_gap": top1_probability - top2_probability,
        "success": success,
        "failure": not success,
        "cache_hits": cache_hits,
        "decode_seconds": decode_seconds,
        "walsh_hadamard_seconds": reconstruct_seconds,
        "shot_total_seconds": time.perf_counter() - shot_start,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_checkpoint_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_empty = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_empty:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One CUDA-QX BB72 TN LER sweep worker.")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--p-grid", required=True)
    parser.add_argument("--shots-per-p", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=20260902)
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
    if args.shots_per_p <= 0:
        raise ValueError("--shots-per-p must be positive")

    p_values = parse_p_grid(args.p_grid)
    torch_info = probe_torch_cuda()
    total_tasks = len(p_values) * args.shots_per_p
    assigned_tasks = [
        task_id for task_id in range(total_tasks) if task_id % args.num_workers == args.worker_id
    ]
    assigned_task_indices = {task_id: index for index, task_id in enumerate(assigned_tasks)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_csv = args.output_dir / f"worker_{args.worker_id:03d}.csv"
    checkpoint_csv = args.output_dir / f"worker_{args.worker_id:03d}_checkpoint.csv"
    rows = load_checkpoint(checkpoint_csv)
    completed_task_ids = {int(row["global_task_id"]) for row in rows}
    assigned_task_set = set(assigned_tasks)
    unexpected_task_ids = completed_task_ids - assigned_task_set
    if unexpected_task_ids:
        raise RuntimeError(
            f"Checkpoint {checkpoint_csv} contains tasks not assigned to worker "
            f"{args.worker_id}: {sorted(unexpected_task_ids)[:10]}"
        )
    remaining_tasks = [task_id for task_id in assigned_tasks if task_id not in completed_task_ids]
    print(
        f"worker={args.worker_id} checkpoint={checkpoint_csv} "
        f"completed={len(completed_task_ids)} remaining={len(remaining_tasks)}",
        flush=True,
    )

    cache_context_by_p: dict[float, tuple[Any, ...]] = {}
    for remaining_index, task_id in enumerate(remaining_tasks):
        p_index = task_id // args.shots_per_p
        shot_id_for_p = task_id % args.shots_per_p
        p = p_values[p_index]
        seed = args.base_seed + p_index * 1_000_000 + shot_id_for_p

        if p not in cache_context_by_p:
            context = build_expected_cache(
                error_type=args.error_type,
                p=p,
                device=args.device,
                logical_basis_source_arg=args.logical_basis_source,
                suppress_cudaqx_logs=args.suppress_cudaqx_logs,
            )
            problem = context[0]
            expected_cache_meta = context[5]
            path_cache, cache_status = load_path_cache(args.path_cache, expected_cache_meta)
            path_entries = path_cache["paths"]
            expected_entries = (1 << problem.k) - 1
            if cache_status["state"] != "loaded" or len(path_entries) != expected_entries:
                raise RuntimeError(
                    "LER sweep requires a complete existing path cache. "
                    f"state={cache_status['state']} entries={len(path_entries)} "
                    f"expected={expected_entries} path={args.path_cache}"
                )
            cache_context_by_p[p] = (*context, path_entries, cache_status)

        (
            problem,
            h_matrix,
            logical_basis,
            noise_model,
            masks,
            _expected_cache_meta,
            runtime_info,
            path_entries,
            cache_status,
        ) = cache_context_by_p[p]

        print(
            f"worker={args.worker_id} task={remaining_index + 1}/{len(remaining_tasks)} "
            f"global_task={task_id} p={p} shot_id_for_p={shot_id_for_p} seed={seed} start",
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
            p=p,
            seed=seed,
            suppress_cudaqx_logs=args.suppress_cudaqx_logs,
        )
        shot_record.update(
            {
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "global_task_id": task_id,
                "p_index": p_index,
                "shot_id_for_p": shot_id_for_p,
                "assigned_task_index": assigned_task_indices[task_id],
                "hostname": platform.node(),
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                "actual_device": runtime_info["actual_device"],
                "contractor": runtime_info["contractor"],
                "logical_basis_source": runtime_info["logical_basis_source"],
                "path_cache": str(args.path_cache),
                "path_cache_load_state": cache_status["state"],
                "path_cache_entries_loaded": len(path_entries),
                "suppress_cudaqx_logs": args.suppress_cudaqx_logs,
                "started_at_utc": utc_now(),
            }
        )
        append_checkpoint_row(checkpoint_csv, shot_record)
        rows.append(shot_record)
        print(
            f"worker={args.worker_id} global_task={task_id} p={p} "
            f"success={shot_record['success']} true_rank={shot_record['true_sector_rank']} "
            f"seconds={shot_record['shot_total_seconds']:.3f}",
            flush=True,
        )

    if not rows:
        raise RuntimeError(f"worker {args.worker_id} received no tasks")

    worker_summary_json = args.output_dir / f"worker_{args.worker_id:03d}_summary.json"
    rows.sort(key=lambda row: int(row["global_task_id"]))
    write_csv(worker_csv, rows)

    failures = sum(1 for row in rows if parse_bool(row["failure"]))
    summary: dict[str, Any] = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "num_p_values": len(p_values),
        "shots_per_p": args.shots_per_p,
        "assigned_tasks": len(rows),
        "failures": failures,
        "logical_error_rate_over_assigned_tasks": failures / len(rows),
        "runtime_seconds": time.perf_counter() - total_start,
        "worker_csv": str(worker_csv),
        "checkpoint_csv": str(checkpoint_csv),
        "resumed_rows": len(completed_task_ids),
        "new_rows": len(rows) - len(completed_task_ids),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "path_cache": str(args.path_cache),
        "p_grid": p_values,
        "torch": torch_info,
    }
    write_json(worker_summary_json, summary)
    print(
        f"worker={args.worker_id} wrote_csv={worker_csv} "
        f"failures={failures}/{len(rows)} runtime_seconds={summary['runtime_seconds']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
