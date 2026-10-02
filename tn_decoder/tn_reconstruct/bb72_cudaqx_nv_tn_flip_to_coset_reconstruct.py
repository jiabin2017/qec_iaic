from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

TN_DECODER_ROOT = Path(__file__).resolve().parents[1]
if str(TN_DECODER_ROOT) not in sys.path:
    sys.path.insert(0, str(TN_DECODER_ROOT))

from common.bb72_baseline_common import build_problem, default_results_dir
from common.bb72_tn_common import (
    enumerate_masks,
    format_syndrome_labels,
    mask_to_bitstring,
    query_flip_probability_with_fallback,
    reconstruct_coset_probs,
)
from examples.bb72_cudaqx_nv_tn_flip_prob_example import (
    build_decoder,
    build_logical_observable,
    build_syndrome,
    format_support,
    format_support_labels,
    logical_signature_in_basis,
    pick_optimize_arg,
    resolve_logical_basis,
)


def make_output_stem(
    output_stem: Path | None,
    error_type: str,
    basis_source: str,
    syndrome_mode: str,
) -> Path:
    if output_stem is not None:
        return output_stem
    return default_results_dir() / f"bb72_cudaqx_nv_tn_reconstruct_{error_type}_{basis_source}_{syndrome_mode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the full 4096-sector BB72 logical posterior from all "
            "CUDA-QX tensor-network logical-parity flip probabilities using a "
            "Walsh-Hadamard transform."
        )
    )
    parser.add_argument("--error-type", choices=["X", "Z"], default="Z")
    parser.add_argument("--p", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--device",
        default="cuda",
        help='Decoder device: "cuda", "cuda:0", or "cpu". Default: cuda.',
    )
    parser.add_argument(
        "--logical-basis-source",
        default="auto",
        choices=["auto", "qldpc-dual", "weight6-x"],
        help=(
            "Logical-basis source. auto picks weight6-x for Z-error decoding "
            "and qldpc-dual for X-error decoding."
        ),
    )
    parser.add_argument(
        "--syndrome-mode",
        default="sample",
        choices=["sample", "zero", "manual"],
        help="How to obtain the input syndrome. Default: sample.",
    )
    parser.add_argument(
        "--syndrome-bits",
        default=None,
        help="Manual syndrome bits as comma-separated 0/1 values.",
    )
    parser.add_argument(
        "--skip-path-opt",
        action="store_true",
        help="Skip explicit optimize_path() and let decode() optimize lazily.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="How many highest-probability sectors to include in the summary. Default: 16.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help=(
            "Optional output stem. The script writes <stem>_flip_probs.csv, "
            "<stem>_coset_probs.csv, and <stem>_summary.txt."
        ),
    )
    return parser.parse_args()


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()

    problem = build_problem(args.error_type)
    h_matrix = np.asarray(problem.check_matrix, dtype=np.uint8)
    logical_basis, logical_basis_source = resolve_logical_basis(
        problem,
        args.error_type,
        args.logical_basis_source,
    )
    noise_model = np.full(problem.n, float(args.p), dtype=np.float64)
    syndrome, sampled_error = build_syndrome(
        problem,
        args.syndrome_mode,
        args.syndrome_bits,
        args.p,
        args.seed,
    )
    output_stem = make_output_stem(
        args.output_stem,
        args.error_type,
        logical_basis_source,
        args.syndrome_mode,
    )

    num_logicals = problem.k
    num_sectors = 1 << num_logicals
    masks = enumerate_masks(num_logicals)
    flip_probabilities = np.zeros(num_sectors, dtype=np.float64)

    flip_rows: list[dict[str, object]] = []
    total_path_opt_seconds = 0.0
    total_decode_seconds = 0.0
    total_query_seconds = 0.0
    explicit_path_opt_count = 0
    actual_device = "<unset>"
    contractor_name = "<unset>"
    public_failure_count = 0
    raw_fallback_count = 0
    raw_clip_count = 0
    raw_fallback_seconds = 0.0
    max_raw_relative_negative_mass = 0.0

    for mask_index, logical_mask in enumerate(masks):
        mask_bits = mask_to_bitstring(logical_mask)
        if mask_index == 0:
            flip_rows.append(
                {
                    "mask_index": 0,
                    "logical_mask_a": mask_bits,
                    "logical_obs_weight": 0,
                    "flip_prob_p1": 0.0,
                    "flip_prob_p0": 1.0,
                    "path_opt_seconds": 0.0,
                    "decode_seconds": 0.0,
                    "raw_fallback_seconds": 0.0,
                    "query_total_seconds": 0.0,
                    "queried_decoder": False,
                    "flip_prob_source": "implicit-zero",
                    "public_valid": True,
                    "public_converged": True,
                    "public_p1": 0.0,
                    "fallback_reason": "",
                    "raw_z0": "",
                    "raw_z1": "",
                    "raw_p1": "",
                    "raw_clipped_mass_count": 0,
                    "raw_relative_negative_mass": 0.0,
                }
            )
            continue

        query_start = time.perf_counter()
        logical_obs = build_logical_observable(logical_basis, logical_mask)
        decoder = build_decoder(h_matrix, logical_obs, noise_model, args.device)
        optimize_arg = pick_optimize_arg(decoder)
        actual_device = str(decoder.contractor_config.device)
        contractor_name = str(decoder.contractor_config.contractor_name)

        path_seconds = 0.0
        if not args.skip_path_opt:
            path_start = time.perf_counter()
            decoder.optimize_path(optimize=optimize_arg)
            path_seconds = time.perf_counter() - path_start
            total_path_opt_seconds += path_seconds
            explicit_path_opt_count += 1

        query = query_flip_probability_with_fallback(
            decoder,
            syndrome,
            optimize_arg=optimize_arg,
        )
        decode_seconds = float(query["query_decode_seconds"])
        query_total_seconds = time.perf_counter() - query_start

        p1 = float(query["p1"])
        flip_probabilities[mask_index] = p1
        total_decode_seconds += decode_seconds
        raw_fallback_seconds += float(query["raw_fallback_seconds"])
        total_query_seconds += query_total_seconds
        if not query["public_valid"]:
            public_failure_count += 1
        if query["raw_used"]:
            raw_fallback_count += 1
        raw_clip_count += int(query["raw_clipped_mass_count"])
        max_raw_relative_negative_mass = max(
            max_raw_relative_negative_mass,
            float(query["raw_relative_negative_mass"]),
        )

        flip_rows.append(
            {
                "mask_index": mask_index,
                "logical_mask_a": mask_bits,
                "logical_obs_weight": int(np.count_nonzero(logical_obs)),
                "flip_prob_p1": p1,
                "flip_prob_p0": 1.0 - p1,
                "path_opt_seconds": path_seconds,
                "decode_seconds": decode_seconds,
                "raw_fallback_seconds": query["raw_fallback_seconds"],
                "query_total_seconds": query_total_seconds,
                "queried_decoder": True,
                "flip_prob_source": query["source"],
                "public_valid": query["public_valid"],
                "public_converged": query["public_converged"],
                "public_p1": query["public_p1"],
                "fallback_reason": query["fallback_reason"],
                "raw_z0": "" if query["raw_z0"] is None else query["raw_z0"],
                "raw_z1": "" if query["raw_z1"] is None else query["raw_z1"],
                "raw_p1": "" if query["raw_p1"] is None else query["raw_p1"],
                "raw_clipped_mass_count": query["raw_clipped_mass_count"],
                "raw_relative_negative_mass": query["raw_relative_negative_mass"],
            }
        )

    reconstruct_start = time.perf_counter()
    coset_probabilities = reconstruct_coset_probs(flip_probabilities)
    reconstruct_seconds = time.perf_counter() - reconstruct_start
    total_runtime_seconds = time.perf_counter() - total_start

    sector_order = np.argsort(-coset_probabilities)
    rank = np.empty(num_sectors, dtype=np.int32)
    rank[sector_order] = np.arange(1, num_sectors + 1, dtype=np.int32)

    true_signature = None
    true_sector_index = None
    true_sector_probability = None
    true_sector_rank = None
    if sampled_error is not None:
        true_signature = logical_signature_in_basis(sampled_error, logical_basis)
        true_sector_index = int(sum(int(bit) << i for i, bit in enumerate(true_signature.tolist())))
        true_sector_probability = float(coset_probabilities[true_sector_index])
        true_sector_rank = int(rank[true_sector_index])

    flip_csv_path = output_stem.with_name(f"{output_stem.name}_flip_probs.csv")
    coset_csv_path = output_stem.with_name(f"{output_stem.name}_coset_probs.csv")
    summary_txt_path = output_stem.with_name(f"{output_stem.name}_summary.txt")

    flip_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with flip_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mask_index",
                "logical_mask_a",
                "logical_obs_weight",
                "flip_prob_p1",
                "flip_prob_p0",
                "path_opt_seconds",
                "decode_seconds",
                "raw_fallback_seconds",
                "query_total_seconds",
                "queried_decoder",
                "flip_prob_source",
                "public_valid",
                "public_converged",
                "public_p1",
                "fallback_reason",
                "raw_z0",
                "raw_z1",
                "raw_p1",
                "raw_clipped_mass_count",
                "raw_relative_negative_mass",
            ],
        )
        writer.writeheader()
        writer.writerows(flip_rows)

    with coset_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sector_index",
                "sector_bits",
                "posterior_probability",
                "rank",
                "is_map",
                "is_true_sector",
            ],
        )
        writer.writeheader()
        for sector_index in range(num_sectors):
            sector_bits = mask_to_bitstring(masks[sector_index])
            writer.writerow(
                {
                    "sector_index": sector_index,
                    "sector_bits": sector_bits,
                    "posterior_probability": f"{coset_probabilities[sector_index]:.18e}",
                    "rank": int(rank[sector_index]),
                    "is_map": sector_index == int(sector_order[0]),
                    "is_true_sector": sampled_error is not None and sector_index == true_sector_index,
                }
            )

    logical_basis_type = "Z" if args.error_type == "X" else "X"
    top_rows: list[str] = []
    for order_index in range(min(args.top_k, num_sectors)):
        sector_index = int(sector_order[order_index])
        sector_bits = mask_to_bitstring(masks[sector_index])
        top_rows.append(
            f"  rank={order_index + 1:4d}  sector_index={sector_index:4d}  "
            f"sector_bits={sector_bits}  posterior={coset_probabilities[sector_index]:.12e}"
        )

    lines = [
        "code=[[72,12,6]]",
        f"error_type={args.error_type}",
        f"requested_device={args.device}",
        f"actual_device={actual_device}",
        f"contractor={contractor_name}",
        f"num_checks={problem.num_checks}",
        f"num_qubits={problem.n}",
        f"num_logicals={problem.k}",
        f"num_sectors={num_sectors}",
        f"p={args.p}",
        f"syndrome_mode={args.syndrome_mode}",
        f"logical_basis_type={logical_basis_type}",
        f"logical_basis_source={logical_basis_source}",
        f"syndrome_weight={int(np.count_nonzero(syndrome))}",
        f"syndrome={syndrome.tolist()}",
        f"syndrome_labels={format_syndrome_labels(syndrome)}",
        f"num_nontrivial_flip_queries={num_sectors - 1}",
        f"explicit_path_opt_count={explicit_path_opt_count}",
        f"implicit_zero_mask_queries=1",
        f"public_failure_count={public_failure_count}",
        f"raw_fallback_count={raw_fallback_count}",
        f"raw_clip_count={raw_clip_count}",
        f"max_raw_relative_negative_mass={max_raw_relative_negative_mass:.12e}",
        f"total_path_opt_seconds={total_path_opt_seconds:.6f}",
        f"total_decode_seconds={total_decode_seconds:.6f}",
        f"raw_fallback_seconds={raw_fallback_seconds:.6f}",
        f"total_query_seconds={total_query_seconds:.6f}",
        f"walsh_hadamard_seconds={reconstruct_seconds:.6f}",
        f"total_runtime_seconds={total_runtime_seconds:.6f}",
        f"flip_probs_csv={flip_csv_path}",
        f"coset_probs_csv={coset_csv_path}",
        "",
        "Interpretation:",
        "  For each nonzero parity mask a in F_2^12, the script queries",
        "  P(a·lambda = 1 | syndrome) with the CUDA-QX tensor-network decoder.",
        "  It then reconstructs the full 4096-sector posterior P(lambda | syndrome)",
        "  by inverse Walsh-Hadamard transform.",
        "",
        "Top sectors:",
        *top_rows,
    ]

    if sampled_error is not None:
        lines.extend(
            [
                "",
                "Sampled underlying error:",
                f"  true_error_weight={int(np.count_nonzero(sampled_error))}",
                f"  true_error_support={format_support(sampled_error)}",
                f"  true_error_support_labels={format_support_labels(sampled_error)}",
                f"  true_logical_signature_lambda={true_signature.tolist()}",
                f"  true_sector_index={true_sector_index}",
                f"  true_sector_probability={true_sector_probability:.12e}",
                f"  true_sector_rank={true_sector_rank}",
            ]
        )

    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_flip_probs_csv={flip_csv_path}")
    print(f"wrote_coset_probs_csv={coset_csv_path}")
    print(f"wrote_summary_txt={summary_txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
