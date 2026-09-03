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


def load_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("worker_[0-9][0-9][0-9].csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge BB72 TN LER sweep worker outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--p-grid", required=True)
    parser.add_argument("--shots-per-p", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.output_dir)
    if not rows:
        raise RuntimeError(f"No worker CSV files found in {args.output_dir}")

    p_values = [float(item.strip()) for item in args.p_grid.split(",") if item.strip()]
    expected_total_shots = len(p_values) * args.shots_per_p
    worker_ids = [int(row["worker_id"]) for row in rows]
    global_task_ids = [int(row["global_task_id"]) for row in rows]
    duplicate_global_task_ids = [
        task_id for task_id, count in sorted(Counter(global_task_ids).items()) if count > 1
    ]
    missing_global_task_ids = [
        task_id for task_id in range(expected_total_shots) if task_id not in set(global_task_ids)
    ]

    all_rows_path = args.output_dir / "all_shots.csv"
    write_csv(all_rows_path, rows)

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
            }
        )

    ler_csv_path = args.output_dir / "ler_summary.csv"
    write_csv(ler_csv_path, ler_rows)

    success = (
        len(rows) == expected_total_shots
        and len(set(worker_ids)) == args.expected_workers
        and not duplicate_global_task_ids
        and not missing_global_task_ids
        and all(int(row["shots"]) == args.shots_per_p for row in ler_rows)
    )
    summary: dict[str, Any] = {
        "success": success,
        "total_shots": len(rows),
        "expected_total_shots": expected_total_shots,
        "expected_workers": args.expected_workers,
        "worker_ids": str(sorted(set(worker_ids))),
        "p_grid": p_values,
        "shots_per_p": args.shots_per_p,
        "duplicate_global_task_ids": str(duplicate_global_task_ids),
        "missing_global_task_ids": str(missing_global_task_ids),
        "actual_device_counts": str(Counter(row["actual_device"] for row in rows)),
        "contractor_counts": str(Counter(row["contractor"] for row in rows)),
        "path_cache_entries_loaded_values": str(
            sorted(set(int(row["path_cache_entries_loaded"]) for row in rows))
        ),
        "all_shots_csv": str(all_rows_path),
        "ler_summary_csv": str(ler_csv_path),
    }
    write_json(args.output_dir / "sweep_summary.json", summary)
    write_csv(args.output_dir / "sweep_summary.csv", [summary])

    print(
        f"success={success} total_shots={len(rows)} expected={expected_total_shots} "
        f"ler_summary_csv={ler_csv_path}",
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
