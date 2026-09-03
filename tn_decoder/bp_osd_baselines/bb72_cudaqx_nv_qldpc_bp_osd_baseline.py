from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, ClassVar

import numpy as np

try:
    from scipy import sparse
except ImportError:  # pragma: no cover - scipy is optional in CUDA-QX docs
    sparse = None

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
from common.local_code_capacity_runtime import CodeCapacityRunner, DecoderFactory, SyndromeDecoder


def _load_cudaq_qec() -> Any:
    try:
        import cudaq_qec as qec
    except ImportError as exc:  # pragma: no cover - expected on this Windows host
        raise ImportError(
            "Failed to import cudaq_qec. Run this script on the Linux machine "
            "where cudaq-qec is installed with GPU support."
        ) from exc
    return qec

class NVQldpcDecoderAdapter:
    """Normalize CUDA-QX decode results to the runner's decoder interface."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.converge: bool | None = None
        self.iter: int | None = None

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        outcome = self._inner.decode(np.asarray(syndrome, dtype=np.uint8))
        result = getattr(outcome, "result", outcome)
        converged = getattr(outcome, "converged", getattr(outcome, "converge", None))
        iterations = None
        for name in ("iter", "iterations", "num_iterations"):
            if hasattr(outcome, name):
                iterations = int(getattr(outcome, name))
                break

        self.converge = None if converged is None else bool(converged)
        self.iter = iterations
        return np.asarray(result, dtype=np.uint8)


def parse_osd_orders(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("osd-orders must not be empty")
    if any(order < 0 for order in values):
        raise ValueError(f"osd-orders must be non-negative, got {values}")
    return values


def make_csv_path(base: Path | None, error_type: str, osd_order: int) -> Path:
    if base is None:
        return (
            default_results_dir()
            / f"bb72_cudaqx_nv_qldpc_bp_osd_order{osd_order}_{error_type}.csv"
        )
    return base.with_name(f"{base.stem}_order{osd_order}{base.suffix}")


def make_runtime_summary_path(base: Path | None, error_type: str) -> Path:
    if base is None:
        return (
            default_results_dir()
            / f"bb72_cudaqx_nv_qldpc_bp_osd_runtime_summary_{error_type}.csv"
        )
    return base.with_name(f"{base.stem}_runtime_summary{base.suffix}")


def query_visible_gpus() -> list[tuple[int, str]] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--query-gpu=index,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    gpus: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append((index, parts[1]))
    return gpus or None


def summarize_visible_gpus(gpus: list[tuple[int, str]] | None) -> str:
    if not gpus:
        return "<unavailable>"
    return "; ".join(f"{index}:{name}" for index, name in gpus)


def infer_device_label(
    requested_cuda_device_id: int | None,
    gpus: list[tuple[int, str]] | None,
) -> str:
    if requested_cuda_device_id is None:
        if gpus:
            for index, name in gpus:
                if index == 0:
                    return f"gpu:0 ({name}) [inferred default]"
            return f"gpu:0 [inferred default]"
        return "gpu:default [inferred from CUDA-QX docs]"

    if gpus:
        for index, name in gpus:
            if index == requested_cuda_device_id:
                return f"gpu:{index} ({name}) [requested]"
    return f"gpu:{requested_cuda_device_id} [requested]"


def report_decoder_device(
    decoder: Any,
    requested_cuda_device_id: int | None,
) -> dict[str, str]:
    reported_attrs: list[str] = []
    for attr in ("device", "device_id", "cuda_device_id", "backend"):
        if hasattr(decoder, attr):
            value = getattr(decoder, attr)
            reported_attrs.append(f"{attr}={value!r}")

    gpus = query_visible_gpus()
    return {
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "visible_gpus": summarize_visible_gpus(gpus),
        "decoder_reported_device": ", ".join(reported_attrs) if reported_attrs else "<not exposed>",
        "inferred_execution_device": infer_device_label(requested_cuda_device_id, gpus),
    }


@dataclass(frozen=True)
class NVQldpcDecoderFactory(DecoderFactory):
    """BB72 BP+OSD baseline backed by CUDA-QX nv-qldpc-decoder."""

    max_iterations: int = 50
    bp_method: int = 1
    scale_factor: float = 1.0
    use_sparsity: bool = True
    use_osd: bool = True
    osd_method: int = 3
    osd_order: int = 2
    cuda_device_id: int | None = None
    name: ClassVar[str] = "nv_qldpc_bp_osd"

    def build(self, problem, p: float) -> SyndromeDecoder:
        qec = _load_cudaq_qec()
        parity = np.asarray(problem.check_matrix, dtype=np.uint8)
        if self.use_sparsity and sparse is not None:
            parity_input: Any = sparse.csr_matrix(parity, dtype=np.uint8)
        else:
            parity_input = parity

        opts: dict[str, Any] = {
            "error_rate_vec": np.full(problem.n, float(p), dtype=np.float64),
            "max_iterations": self.max_iterations,
            "bp_method": self.bp_method,
            "use_sparsity": self.use_sparsity,
            "use_osd": self.use_osd,
        }
        if self.bp_method == 1:
            opts["scale_factor"] = self.scale_factor
        if self.use_osd:
            opts["osd_method"] = self.osd_method
            opts["osd_order"] = self.osd_order
        if self.cuda_device_id is not None:
            opts["cuda_device_id"] = self.cuda_device_id

        decoder = qec.get_decoder("nv-qldpc-decoder", parity_input, **opts)
        return NVQldpcDecoderAdapter(decoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BB72 BP+OSD baseline using CUDA-QX nv-qldpc-decoder. "
            "This requires a CUDA-Q compatible GPU."
        )
    )
    add_common_bb72_args(parser, default_seed=20260826)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument(
        "--bp-method",
        type=int,
        default=1,
        choices=[0, 1],
        help="0=sum-product, 1=min-sum. Default 1 to align with the qldpc baseline.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--dense", action="store_true", help="Disable sparse CSR input.")
    parser.add_argument("--no-osd", action="store_true")
    parser.add_argument(
        "--osd-method",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="1=OSD-0, 2=Exhaustive, 3=Combination Sweep.",
    )
    parser.add_argument(
        "--osd-orders",
        default="0,2,4,6,8,10",
        help="Comma-separated OSD orders to sweep. Default: 0,2,4,6,8,10.",
    )
    parser.add_argument("--cuda-device-id", type=int, default=None)
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
    if args.osd_method == 1 and any(order != 0 for order in osd_orders):
        raise ValueError("CUDA-QX osd_method=1 (OSD-0) only supports order 0")

    problem = build_problem(args.error_type)
    noise = build_noise(args.error_type)

    runtime_rows: list[dict[str, object]] = []
    written_paths: list[Path] = []
    effective_orders = [0] if args.no_osd else osd_orders
    for osd_order in effective_orders:
        decoder_factory = NVQldpcDecoderFactory(
            max_iterations=args.max_iterations,
            bp_method=args.bp_method,
            scale_factor=args.scale_factor,
            use_sparsity=not args.dense,
            use_osd=not args.no_osd,
            osd_method=args.osd_method,
            osd_order=osd_order,
            cuda_device_id=args.cuda_device_id,
        )
        runner = CodeCapacityRunner(
            problem,
            noise,
            decoder_factory,
            seed=args.seed,
            code_name="bb72",
        )
        device_report = report_decoder_device(
            decoder_factory.build(problem, p_values[0]),
            args.cuda_device_id,
        )

        start = time.perf_counter()
        result = runner.sweep(p_values, shots=args.shots)
        elapsed_seconds = time.perf_counter() - start

        csv_path = make_csv_path(args.csv, args.error_type, osd_order)
        written = result.write_csv(csv_path)
        written_paths.append(written)
        runtime_rows.append(
            {
                "bp_method": args.bp_method,
                "use_osd": not args.no_osd,
                "osd_method": args.osd_method,
                "osd_order": osd_order,
                "error_type": args.error_type,
                "shots_per_point": args.shots,
                "num_p_points": len(p_values),
                "total_decodes": args.shots * len(p_values),
                "runtime_seconds": elapsed_seconds,
                "seconds_per_decode": elapsed_seconds / (args.shots * len(p_values)),
                "decoder_reported_device": device_report["decoder_reported_device"],
                "inferred_execution_device": device_report["inferred_execution_device"],
                "visible_gpus": device_report["visible_gpus"],
                "csv_path": str(written),
            }
        )
        print(
            f"osd_order={osd_order}: runtime_seconds={elapsed_seconds:.6f}, csv={written}"
        )
        print(
            "  device_report: "
            f"decoder_reported_device={device_report['decoder_reported_device']}, "
            f"inferred_execution_device={device_report['inferred_execution_device']}, "
            f"CUDA_VISIBLE_DEVICES={device_report['cuda_visible_devices_env']}, "
            f"visible_gpus={device_report['visible_gpus']}"
        )

    runtime_summary_path = make_runtime_summary_path(args.csv, args.error_type)
    runtime_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bp_method",
                "use_osd",
                "osd_method",
                "osd_order",
                "error_type",
                "shots_per_point",
                "num_p_points",
                "total_decodes",
                "runtime_seconds",
                "seconds_per_decode",
                "decoder_reported_device",
                "inferred_execution_device",
                "visible_gpus",
                "csv_path",
            ],
        )
        writer.writeheader()
        writer.writerows(runtime_rows)

    print("completed CUDA-QX BB72 nv-qldpc-decoder baseline")
    print("code=[[72,12,6]]")
    print(f"error_type={args.error_type}")
    print(f"shots_per_point={args.shots}")
    print(f"p_grid={p_values}")
    print(f"osd_orders={effective_orders}")
    for path in written_paths:
        print(f"csv={path}")
    print(f"runtime_summary_csv={runtime_summary_path}")


if __name__ == "__main__":
    main()
