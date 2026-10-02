from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write empty CSV")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def optional_int_sum(rows: list[dict[str, str]], key: str) -> int | None:
    if not any(key in row for row in rows):
        return None
    return sum(int(float(row.get(key) or 0)) for row in rows)


def optional_float_sum(rows: list[dict[str, str]], key: str) -> float | None:
    if not any(key in row for row in rows):
        return None
    return sum(float(row.get(key) or 0.0) for row in rows)


def optional_float_max(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return max(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all 7-GPU H100 LER sweep batches.")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--p-grid", required=True)
    parser.add_argument("--shots-per-p", type=int, required=True)
    parser.add_argument("--expected-batches", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p_values = [float(item.strip()) for item in args.p_grid.split(",") if item.strip()]
    if not p_values:
        raise ValueError("--p-grid must contain at least one value")

    expected_total_shots = len(p_values) * args.shots_per_p
    batch_dirs = sorted(path for path in (args.sweep_dir / "batches").glob("batch_*") if path.is_dir())
    if not batch_dirs:
        raise RuntimeError(f"No batch directories found under {args.sweep_dir / 'batches'}")

    rows: list[dict[str, str]] = []
    missing_batch_outputs: list[str] = []
    for batch_dir in batch_dirs:
        all_shots = batch_dir / "all_shots.csv"
        if not all_shots.exists():
            missing_batch_outputs.append(str(all_shots))
            continue
        for row in read_csv(all_shots):
            row["batch_dir"] = batch_dir.name
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No shot rows found under {args.sweep_dir}")

    rows.sort(key=lambda row: int(row["global_task_id"]))
    global_task_ids = [int(row["global_task_id"]) for row in rows]
    id_counts = Counter(global_task_ids)
    duplicate_global_task_ids = [
        task_id for task_id, count in sorted(id_counts.items()) if count > 1
    ]
    missing_global_task_ids = [
        task_id for task_id in range(expected_total_shots) if task_id not in id_counts
    ]
    unexpected_global_task_ids = [
        task_id for task_id in sorted(id_counts) if task_id < 0 or task_id >= expected_total_shots
    ]

    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["p"])].append(row)

    ler_rows: list[dict[str, Any]] = []
    for p in p_values:
        p_rows = grouped.get(p, [])
        shots = len(p_rows)
        failures = sum(1 for row in p_rows if parse_bool(row["failure"]))
        ler = failures / shots if shots else math.nan
        stderr = math.sqrt(ler * (1.0 - ler) / shots) if shots else math.nan
        ler_rows.append(
            {
                "p": p,
                "shots": shots,
                "expected_shots": args.shots_per_p,
                "failures": failures,
                "logical_error_rate": ler,
                "binomial_stderr": stderr,
                "successes": shots - failures,
                "avg_shot_total_seconds": (
                    sum(float(row["shot_total_seconds"]) for row in p_rows) / shots
                    if shots
                    else math.nan
                ),
                "avg_decode_seconds": (
                    sum(float(row["decode_seconds"]) for row in p_rows) / shots
                    if shots
                    else math.nan
                ),
                "avg_walsh_hadamard_seconds": (
                    sum(float(row["walsh_hadamard_seconds"]) for row in p_rows) / shots
                    if shots
                    else math.nan
                ),
                "public_failure_count": optional_int_sum(p_rows, "public_failure_count"),
                "raw_fallback_count": optional_int_sum(p_rows, "raw_fallback_count"),
                "raw_clip_count": optional_int_sum(p_rows, "raw_clip_count"),
                "raw_fallback_seconds": optional_float_sum(p_rows, "raw_fallback_seconds"),
                "max_raw_relative_negative_mass": optional_float_max(
                    p_rows, "max_raw_relative_negative_mass"
                ),
            }
        )

    all_rows_path = args.sweep_dir / "all_shots.csv"
    ler_csv_path = args.sweep_dir / "ler_summary.csv"
    write_csv(all_rows_path, rows)
    write_csv(ler_csv_path, ler_rows)

    expected_batch_count_ok = (
        args.expected_batches is None or len(batch_dirs) == args.expected_batches
    )
    success = (
        len(rows) == expected_total_shots
        and expected_batch_count_ok
        and not duplicate_global_task_ids
        and not missing_global_task_ids
        and not unexpected_global_task_ids
        and not missing_batch_outputs
        and all(int(row["shots"]) == args.shots_per_p for row in ler_rows)
    )

    summary: dict[str, Any] = {
        "success": success,
        "total_shots": len(rows),
        "expected_total_shots": expected_total_shots,
        "completed_batch_dirs": [path.name for path in batch_dirs],
        "completed_batch_count": len(batch_dirs),
        "expected_batches": args.expected_batches,
        "p_grid": p_values,
        "shots_per_p": args.shots_per_p,
        "duplicate_global_task_ids": str(duplicate_global_task_ids),
        "missing_global_task_ids": str(missing_global_task_ids),
        "unexpected_global_task_ids": str(unexpected_global_task_ids),
        "missing_batch_outputs": missing_batch_outputs,
        "actual_device_counts": str(Counter(row["actual_device"] for row in rows)),
        "contractor_counts": str(Counter(row["contractor"] for row in rows)),
        "path_cache_entries_loaded_values": str(
            sorted(set(int(row["path_cache_entries_loaded"]) for row in rows))
        ),
        "public_failure_count": optional_int_sum(rows, "public_failure_count"),
        "raw_fallback_count": optional_int_sum(rows, "raw_fallback_count"),
        "raw_clip_count": optional_int_sum(rows, "raw_clip_count"),
        "raw_fallback_seconds": optional_float_sum(rows, "raw_fallback_seconds"),
        "max_raw_relative_negative_mass": optional_float_max(
            rows, "max_raw_relative_negative_mass"
        ),
        "all_shots_csv": str(all_rows_path),
        "ler_summary_csv": str(ler_csv_path),
    }
    write_json(args.sweep_dir / "sweep_summary.json", summary)
    write_csv(args.sweep_dir / "sweep_summary.csv", [summary])

    print(
        f"success={success} total_shots={len(rows)} expected={expected_total_shots} "
        f"batches={len(batch_dirs)} ler_summary_csv={ler_csv_path}",
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
