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
    atomic_pickle_dump,
    build_cache_meta,
    enumerate_masks,
    format_syndrome_labels,
    initialize_path_cache,
    load_path_cache,
    make_cache_entry,
    mask_to_bitstring,
    optimize_decoder_path,
    query_flip_probability_with_fallback,
    reconstruct_coset_probs,
    restore_cached_path,
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


DEFAULT_CACHE_SAVE_EVERY = 32
DEFAULT_PROGRESS_EVERY = 128
DEFAULT_PATH_CACHE_DIR = Path(__file__).resolve().parent / "path_cache"


def make_output_stem(
    output_stem: Path | None,
    error_type: str,
    basis_source: str,
    syndrome_mode: str,
) -> Path:
    if output_stem is not None:
        return output_stem
    return default_results_dir() / (
        f"bb72_cudaqx_nv_tn_reconstruct_cached_{error_type}_{basis_source}_{syndrome_mode}"
    )


def make_default_path_cache_path(
    error_type: str,
    logical_basis_source: str,
    contractor_name: str,
) -> Path:
    stem = f"bb72_cudaqx_nv_tn_path_cache_{error_type}_{logical_basis_source}_{contractor_name}.pkl"
    return DEFAULT_PATH_CACHE_DIR / stem


def make_path_cache_path(
    explicit_path: Path | None,
    output_stem: Path,
    *,
    error_type: str,
    logical_basis_source: str,
    contractor_name: str,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    _ = output_stem
    return make_default_path_cache_path(error_type, logical_basis_source, contractor_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the full 4096-sector BB72 logical posterior from all "
            "CUDA-QX tensor-network logical-parity flip probabilities using a "
            "Walsh-Hadamard transform, while caching one optimized path per nonzero "
            "logical mask in a single total cache file."
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
        help=(
            "Skip explicit optimize_path() calls. Cached paths are still reused, "
            "but cache misses rely on the decoder's lazy optimization path."
        ),
    )
    parser.add_argument(
        "--disable-path-cache",
        action="store_true",
        help="Disable loading and saving the total path cache file.",
    )
    parser.add_argument(
        "--path-cache",
        type=Path,
        default=None,
        help=(
            "Optional explicit cache file path. Default: a shared tn_reconstruct/path_cache/*.pkl "
            "file keyed by error type, logical basis, and contractor."
        ),
    )
    parser.add_argument(
        "--cache-save-every",
        type=int,
        default=DEFAULT_CACHE_SAVE_EVERY,
        help=(
            "When the cache gains new entries, checkpoint the total cache file every N "
            "new paths. Default: 32."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress line every N queried masks. Set <= 0 to disable. Default: 128.",
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

    if args.cache_save_every <= 0:
        raise ValueError("--cache-save-every must be positive")

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

    probe_mask = masks[1]
    probe_logical_obs = build_logical_observable(logical_basis, probe_mask)
    probe_decoder = build_decoder(h_matrix, probe_logical_obs, noise_model, args.device)
    optimize_arg = pick_optimize_arg(probe_decoder)
    actual_device = str(probe_decoder.contractor_config.device)
    contractor_name = str(probe_decoder.contractor_config.contractor_name)

    expected_cache_meta = build_cache_meta(
        problem,
        h_matrix,
        logical_basis,
        error_type=args.error_type,
        logical_basis_source=logical_basis_source,
        contractor_name=contractor_name,
    )
    path_cache_path = make_path_cache_path(
        args.path_cache,
        output_stem,
        error_type=args.error_type,
        logical_basis_source=logical_basis_source,
        contractor_name=contractor_name,
    )

    if args.disable_path_cache:
        path_cache = initialize_path_cache(expected_cache_meta)
        path_cache_status = {
            "state": "disabled",
            "entries_loaded": 0,
            "message": "path cache disabled by --disable-path-cache",
        }
    else:
        path_cache, path_cache_status = load_path_cache(path_cache_path, expected_cache_meta)

    flip_rows: list[dict[str, object]] = []
    total_path_opt_seconds = 0.0
    total_decode_seconds = 0.0
    total_query_seconds = 0.0
    explicit_path_opt_count = 0
    explicit_path_cache_hit_count = 0
    cache_miss_count = 0
    cache_invalid_rebuild_count = 0
    lazy_path_capture_count = 0
    cache_checkpoint_writes = 0
    cache_entries_written = 0
    pending_new_cache_entries = 0
    queried_masks = 0
    public_failure_count = 0
    raw_fallback_count = 0
    raw_clip_count = 0
    raw_fallback_seconds = 0.0
    max_raw_relative_negative_mass = 0.0

    def maybe_checkpoint_cache(force: bool = False) -> None:
        nonlocal cache_checkpoint_writes
        nonlocal pending_new_cache_entries

        if args.disable_path_cache or pending_new_cache_entries == 0:
            return
        if not force and pending_new_cache_entries < args.cache_save_every:
            return
        atomic_pickle_dump(path_cache, path_cache_path)
        cache_checkpoint_writes += 1
        pending_new_cache_entries = 0

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
                    "path_source": "implicit-zero",
                    "path_cache_hit": False,
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

        logical_obs = build_logical_observable(logical_basis, logical_mask)
        decoder = probe_decoder if mask_index == 1 else build_decoder(
            h_matrix,
            logical_obs,
            noise_model,
            args.device,
        )

        query_start = time.perf_counter()
        path_seconds = 0.0
        decode_seconds = 0.0
        cache_hit = False
        path_source = "optimized"
        refresh_cache_after_decode = False

        cache_entry = None if args.disable_path_cache else path_cache["paths"].get(mask_bits)
        if cache_entry is not None:
            restore_cached_path(decoder, cache_entry)
            cache_hit = True
            path_source = "cache"
            explicit_path_cache_hit_count += 1
        else:
            cache_miss_count += 1
            if not args.skip_path_opt:
                path_info, path_seconds = optimize_decoder_path(decoder, optimize_arg)
                total_path_opt_seconds += path_seconds
                explicit_path_opt_count += 1
                if not args.disable_path_cache:
                    path_cache["paths"][mask_bits] = make_cache_entry(
                        decoder,
                        logical_obs,
                        str(path_info),
                    )
                    cache_entries_written += 1
                    pending_new_cache_entries += 1
                    maybe_checkpoint_cache(force=False)
            else:
                path_source = "lazy"

        try:
            query = query_flip_probability_with_fallback(
                decoder,
                syndrome,
                optimize_arg=optimize_arg,
            )
        except Exception:
            if not cache_hit:
                raise

            cache_invalid_rebuild_count += 1
            decoder.path_single = None
            decoder.slicing_single = None

            if args.skip_path_opt:
                path_source = "cache-invalid-lazy-rebuilt"
                refresh_cache_after_decode = True
            else:
                path_info, repaired_path_seconds = optimize_decoder_path(decoder, optimize_arg)
                path_seconds += repaired_path_seconds
                total_path_opt_seconds += repaired_path_seconds
                explicit_path_opt_count += 1
                path_source = "cache-invalid-reoptimized"
                if not args.disable_path_cache:
                    path_cache["paths"][mask_bits] = make_cache_entry(
                        decoder,
                        logical_obs,
                        str(path_info),
                    )
                    cache_entries_written += 1
                    pending_new_cache_entries += 1
                    maybe_checkpoint_cache(force=False)
            query = query_flip_probability_with_fallback(
                decoder,
                syndrome,
                optimize_arg=optimize_arg,
            )

        decode_seconds = float(query["query_decode_seconds"])
        query_total_seconds = time.perf_counter() - query_start

        if (
            not args.disable_path_cache
            and (cache_entry is None or refresh_cache_after_decode)
            and getattr(decoder, "path_single", None) is not None
            and (refresh_cache_after_decode or mask_bits not in path_cache["paths"])
        ):
            path_cache["paths"][mask_bits] = make_cache_entry(decoder, logical_obs, "lazy decode path")
            cache_entries_written += 1
            lazy_path_capture_count += 1
            pending_new_cache_entries += 1
            maybe_checkpoint_cache(force=False)

        p1 = float(query["p1"])
        flip_probabilities[mask_index] = p1
        total_decode_seconds += decode_seconds
        raw_fallback_seconds += float(query["raw_fallback_seconds"])
        total_query_seconds += query_total_seconds
        queried_masks += 1
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
                "path_source": path_source,
                "path_cache_hit": cache_hit,
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

        if args.progress_every > 0 and queried_masks % args.progress_every == 0:
            elapsed = time.perf_counter() - total_start
            print(
                "progress: "
                f"queried_masks={queried_masks}/{num_sectors - 1} "
                f"cache_hits={explicit_path_cache_hit_count} "
                f"cache_misses={cache_miss_count} "
                f"elapsed_seconds={elapsed:.3f}"
            )

    maybe_checkpoint_cache(force=True)

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
                "path_source",
                "path_cache_hit",
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
        f"cached_path_hit_count={explicit_path_cache_hit_count}",
        f"cache_miss_count={cache_miss_count}",
        f"cache_invalid_rebuild_count={cache_invalid_rebuild_count}",
        f"lazy_path_capture_count={lazy_path_capture_count}",
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
        f"path_cache_enabled={not args.disable_path_cache}",
        f"path_cache_file={path_cache_path}",
        f"path_cache_load_state={path_cache_status['state']}",
        f"path_cache_load_message={path_cache_status['message']}",
        f"path_cache_entries_loaded={path_cache_status['entries_loaded']}",
        f"path_cache_entries_final={len(path_cache['paths'])}",
        f"path_cache_entries_written_this_run={cache_entries_written}",
        f"path_cache_checkpoint_writes={cache_checkpoint_writes}",
        f"flip_probs_csv={flip_csv_path}",
        f"coset_probs_csv={coset_csv_path}",
        "",
        "Interpretation:",
        "  For each nonzero parity mask a in F_2^12, the script queries",
        "  P(a·lambda = 1 | syndrome) with the CUDA-QX tensor-network decoder.",
        "  It then reconstructs the full 4096-sector posterior P(lambda | syndrome)",
        "  by inverse Walsh-Hadamard transform.",
        "  This cached variant stores one optimized contraction path per nonzero mask",
        "  in a single total cache file, so later runs can skip most path optimization.",
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
    if not args.disable_path_cache:
        print(f"path_cache_file={path_cache_path}")
        print(f"path_cache_entries_final={len(path_cache['paths'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
