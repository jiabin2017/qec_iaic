from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    if not rows:
        raise RuntimeError("Cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge BB72 TN LER pilot worker outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, default=4)
    parser.add_argument("--shots-per-worker", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.output_dir)
    if not rows:
        raise RuntimeError(f"No worker CSV files found in {args.output_dir}")

    expected_total_shots = args.expected_workers * args.shots_per_worker
    worker_ids = [int(row["worker_id"]) for row in rows]
    global_shot_ids = [int(row["global_shot_id"]) for row in rows]
    duplicate_global_shot_ids = [
        shot_id for shot_id, count in sorted(Counter(global_shot_ids).items()) if count > 1
    ]
    missing_global_shot_ids = [
        shot_id for shot_id in range(expected_total_shots) if shot_id not in set(global_shot_ids)
    ]
    failures = sum(1 for row in rows if parse_bool(row["failure"]))
    successes = sum(1 for row in rows if parse_bool(row["success"]))
    total_shots = len(rows)

    all_rows_path = args.output_dir / "all_shots.csv"
    write_csv(all_rows_path, rows)

    summary: dict[str, Any] = {
        "success": (
            total_shots == expected_total_shots
            and len(set(worker_ids)) == args.expected_workers
            and not duplicate_global_shot_ids
            and not missing_global_shot_ids
        ),
        "total_shots": total_shots,
        "expected_total_shots": expected_total_shots,
        "successes": successes,
        "failures": failures,
        "logical_error_rate": failures / total_shots,
        "worker_ids": str(sorted(set(worker_ids))),
        "expected_workers": args.expected_workers,
        "shots_per_worker": args.shots_per_worker,
        "duplicate_global_shot_ids": str(duplicate_global_shot_ids),
        "missing_global_shot_ids": str(missing_global_shot_ids),
        "cuda_visible_devices": str(
            {int(row["worker_id"]): row["cuda_visible_devices"] for row in rows}
        ),
        "actual_device_counts": str(Counter(row["actual_device"] for row in rows)),
        "contractor_counts": str(Counter(row["contractor"] for row in rows)),
        "path_cache_entries_loaded_values": str(
            sorted(set(int(row["path_cache_entries_loaded"]) for row in rows))
        ),
        "sum_shot_total_seconds": sum(float(row["shot_total_seconds"]) for row in rows),
        "avg_shot_total_seconds": sum(float(row["shot_total_seconds"]) for row in rows)
        / total_shots,
        "max_shot_total_seconds": max(float(row["shot_total_seconds"]) for row in rows),
        "all_shots_csv": str(all_rows_path),
    }

    summary_json = args.output_dir / "pilot_summary.json"
    summary_csv = args.output_dir / "pilot_summary.csv"
    write_json(summary_json, summary)
    write_csv(summary_csv, [summary])

    print(
        f"success={summary['success']} total_shots={total_shots} "
        f"failures={failures} logical_error_rate={summary['logical_error_rate']:.6g} "
        f"avg_shot_total_seconds={summary['avg_shot_total_seconds']:.3f} "
        f"summary_json={summary_json}",
        flush=True,
    )
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
