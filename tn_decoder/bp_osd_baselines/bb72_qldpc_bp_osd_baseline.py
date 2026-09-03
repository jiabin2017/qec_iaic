from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import ClassVar

TN_DECODER_ROOT = Path(__file__).resolve().parents[1]
if str(TN_DECODER_ROOT) not in sys.path:
    sys.path.insert(0, str(TN_DECODER_ROOT))

from common.bb72_baseline_common import (
    add_common_bb72_args,
    build_noise,
    build_problem,
    default_results_dir,
    parse_p_grid,
)
from common.local_code_capacity_runtime import BPOSDDecoderFactory, CodeCapacityRunner


@dataclass(frozen=True)
class QldpcBPOSDDecoderFactory(BPOSDDecoderFactory):
    """Thin wrapper so CSV metadata distinguishes this baseline."""

    name: ClassVar[str] = "qldpc_bp_osd"


def parse_osd_orders(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("osd-orders must not be empty")
    if any(order < 0 for order in values):
        raise ValueError(f"osd-orders must be non-negative, got {values}")
    return values


def make_csv_path(base: Path | None, error_type: str, osd_order: int) -> Path:
    if base is None:
        return default_results_dir() / f"bb72_qldpc_bp_osd_order{osd_order}_{error_type}.csv"
    return base.with_name(f"{base.stem}_order{osd_order}{base.suffix}")


def make_runtime_summary_path(base: Path | None, error_type: str) -> Path:
    if base is None:
        return default_results_dir() / f"bb72_qldpc_bp_osd_runtime_summary_{error_type}.csv"
    return base.with_name(f"{base.stem}_runtime_summary{base.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BB72 BP+OSD baseline using the qldpc package decoder stack."
    )
    add_common_bb72_args(parser, default_seed=20260826)
    parser.add_argument("--bp-max-iter", type=int, default=50)
    parser.add_argument(
        "--bp-method",
        default="minimum_sum",
        choices=["minimum_sum", "product_sum", "product_sum_notanh"],
    )
    parser.add_argument("--schedule", default="parallel", choices=["parallel", "serial"])
    parser.add_argument(
        "--osd-method",
        default="OSD_CS",
        choices=["OSD_0", "OSD_E", "OSD_CS"],
    )
    parser.add_argument(
        "--osd-orders",
        default="0,2,4,6,8,10",
        help="Comma-separated OSD orders to sweep. Default: 0,2,4,6,8,10.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV stem. Multi-order runs append _order{n}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_values = parse_p_grid(args.p_grid)
    osd_orders = parse_osd_orders(args.osd_orders)
    if args.osd_method == "OSD_0" and any(order != 0 for order in osd_orders):
        raise ValueError("OSD_0 only supports order 0")

    problem = build_problem(args.error_type)
    noise = build_noise(args.error_type)

    runtime_rows: list[dict[str, object]] = []
    written_paths: list[Path] = []
    for index, osd_order in enumerate(osd_orders):
        decoder_factory = QldpcBPOSDDecoderFactory(
            max_iter=args.bp_max_iter,
            bp_method=args.bp_method,
            schedule=args.schedule,
            osd_method=args.osd_method,
            osd_order=osd_order,
        )
        runner = CodeCapacityRunner(
            problem,
            noise,
            decoder_factory,
            seed=args.seed,
            code_name="bb72",
        )

        start = time.perf_counter()
        result = runner.sweep(p_values, shots=args.shots)
        elapsed_seconds = time.perf_counter() - start

        csv_path = make_csv_path(args.csv, args.error_type, osd_order)
        written = result.write_csv(csv_path)
        written_paths.append(written)
        runtime_rows.append(
            {
                "osd_method": args.osd_method,
                "osd_order": osd_order,
                "error_type": args.error_type,
                "shots_per_point": args.shots,
                "num_p_points": len(p_values),
                "total_decodes": args.shots * len(p_values),
                "runtime_seconds": elapsed_seconds,
                "seconds_per_decode": elapsed_seconds / (args.shots * len(p_values)),
                "csv_path": str(written),
            }
        )
        print(
            f"osd_order={osd_order}: runtime_seconds={elapsed_seconds:.6f}, csv={written}"
        )

    runtime_summary_path = make_runtime_summary_path(args.csv, args.error_type)
    runtime_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "osd_method",
                "osd_order",
                "error_type",
                "shots_per_point",
                "num_p_points",
                "total_decodes",
                "runtime_seconds",
                "seconds_per_decode",
                "csv_path",
            ],
        )
        writer.writeheader()
        writer.writerows(runtime_rows)

    print("completed qldpc BB72 BP+OSD baseline")
    print("code=[[72,12,6]]")
    print(f"error_type={args.error_type}")
    print(f"shots_per_point={args.shots}")
    print(f"p_grid={p_values}")
    print(f"osd_orders={osd_orders}")
    for path in written_paths:
        print(f"csv={path}")
    print(f"runtime_summary_csv={runtime_summary_path}")


if __name__ == "__main__":
    main()
