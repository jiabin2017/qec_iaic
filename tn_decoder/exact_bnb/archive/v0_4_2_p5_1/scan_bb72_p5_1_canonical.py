from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sympy.abc import x, y
from qldpc import codes
from qldpc.objects import Pauli

from exact_block_bnb import ExactBlockBnB
from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder
from p5_utils import (
    block_topologies,
    canonical_block_signature,
    canonical_topology,
    named_logical_order,
    order_label,
    parse_block_sizes,
)
from tune_bb72_p5_paths import (
    aggressive_gpu_cleanup,
    build_network_spec,
    cuquantum_version,
    gpu_memory_info,
    gpu_name,
    make_network,
    sync_cuda,
    tune_topology,
)
from run_bb72_exact_bnb_p5 import find_weight3_logical_collision


def build_code():
    code = codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    )
    H = np.asarray(code.matrix_z, dtype=np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    return H, LZ


def fixed_syndromes(H: np.ndarray, LZ: np.ndarray):
    easy_error = np.zeros(H.shape[1], dtype=np.uint8)
    easy_error[28] = 1
    easy_syndrome = ((H @ easy_error) % 2).astype(float)
    _, _, hard_syn_u8, _, _ = find_weight3_logical_collision(H, LZ)
    return {
        "easy": easy_syndrome,
        "hard": hard_syn_u8.astype(float),
    }


def _path_fingerprint(entry: dict[str, Any]) -> tuple:
    """Hashable identity for de-duplicating imported optimizer plans."""
    path = tuple(tuple(int(v) for v in pair) for pair in entry.get("path", ()))
    slices = tuple((str(a), int(b)) for a, b in entry.get("slices", ()))
    return path, slices


def load_imported_candidates(profile_dir: Path | None) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    """Load best plans from previous P5 profile files into a canonical pool.

    A previous profile may store the same logical set in several within-block
    orders.  We keep every distinct path plan as a candidate under the sorted
    canonical logical-set key.  They are re-benchmarked on the canonical output
    order before selection, so stale/random old timing does not decide the pool.
    """
    out: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    if profile_dir is None or not profile_dir.exists():
        return out

    seen: dict[tuple[int, ...], set[tuple]] = defaultdict(set)
    for path in sorted(profile_dir.glob("paths_b*.pkl")):
        try:
            with path.open("rb") as fh:
                profile = pickle.load(fh)
        except Exception as exc:
            print("IMPORT skip unreadable", path, exc)
            continue
        if not isinstance(profile, dict):
            continue
        for raw_key, raw_entry in profile.get("paths", {}).items():
            try:
                ordered = tuple(int(v) for v in raw_key)
                canonical = canonical_topology(ordered)
                entry = dict(raw_entry)
                fp = _path_fingerprint(entry)
            except Exception:
                continue
            if fp in seen[canonical]:
                continue
            seen[canonical].add(fp)
            entry["_import_source_profile"] = str(path)
            entry["_import_source_open_logicals"] = list(ordered)
            entry["_import_recorded_benchmark_seconds"] = entry.get(
                "benchmark_median_seconds"
            )
            out[canonical].append(entry)
    return out


def benchmark_fixed_plan(
    *,
    decoder,
    syndrome: np.ndarray,
    canonical: tuple[int, ...],
    candidate: dict[str, Any],
    memory_limit: str,
    repeats: int,
) -> dict[str, Any] | None:
    """Validate and re-benchmark an imported path on the canonical equation."""
    equation, operands, _ = build_network_spec(decoder, syndrome, canonical)
    aggressive_gpu_cleanup()
    network = make_network(equation, operands, decoder, memory_limit)
    try:
        optimize = {
            "path": [tuple(int(v) for v in pair) for pair in candidate["path"]],
            "slicing": [tuple(item) for item in candidate.get("slices", ())],
        }
        # This both validates that the plan is legal for the canonical equation
        # and installs it into the Network.
        _, info = network.contract_path(optimize=optimize)
        times: list[float] = []
        for _ in range(max(1, int(repeats))):
            sync_cuda()
            t0 = time.perf_counter()
            _ = network.contract(release_workspace=True)
            sync_cuda()
            times.append(time.perf_counter() - t0)

        row = dict(candidate)
        row["benchmark_times"] = [float(v) for v in times]
        row["benchmark_median_seconds"] = float(statistics.median(times))
        row["opt_cost"] = float(getattr(info, "opt_cost", row.get("opt_cost", float("nan"))))
        row["largest_intermediate"] = float(
            getattr(info, "largest_intermediate", row.get("largest_intermediate", float("nan")))
        )
        row["num_slices"] = int(getattr(info, "num_slices", row.get("num_slices", 1)))
        row["_candidate_kind"] = "imported_rebenchmarked"
        row["_canonical_open_logicals"] = list(canonical)
        return row
    except Exception as exc:
        print(
            "IMPORT candidate rejected for canonical",
            canonical,
            "source=",
            candidate.get("_import_source_open_logicals"),
            "error=",
            exc,
        )
        return None
    finally:
        try:
            network.free()
        except Exception:
            pass
        try:
            del network
        except Exception:
            pass
        aggressive_gpu_cleanup()


def benchmark_profile(
    *,
    H: np.ndarray,
    LZ: np.ndarray,
    syndromes: dict[str, np.ndarray],
    p: float,
    block_size: int,
    logical_order: tuple[int, ...],
    profile: dict[str, Any],
    memory_limit: str,
    guard_rtol: float,
    repeats: int,
    gpu_resident_operands: bool,
    constant_cache: bool,
    retain_scratch: bool,
    scratch_retain_budget: str,
) -> dict[str, Any]:
    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[p] * H.shape[1],
        device="cuda",
    )
    topologies = block_topologies(logical_order, block_size)
    decoder.set_fast_cache_max_entries(max(8, len(topologies) + 1))
    decoder.set_fast_path_profile(profile)

    def run_one(syndrome: np.ndarray):
        syndrome_list = syndrome.astype(float).tolist()

        def oracle(fixed_logicals, open_logicals):
            sync_cuda()
            out = decoder.contract_logical_mass_fast(
                syndrome=syndrome_list,
                fixed_logicals=dict(fixed_logicals),
                open_logicals=list(open_logicals),
                memory_limit=memory_limit,
                autotune_iterations=0,
                release_workspace=not retain_scratch,
                use_constant_qualifiers=constant_cache,
                use_gpu_resident_operands=gpu_resident_operands,
                retain_constant_cache=constant_cache,
                retain_scratch_workspace=retain_scratch,
                scratch_retention_budget=scratch_retain_budget,
            )
            sync_cuda()
            return out

        return ExactBlockBnB(
            k=LZ.shape[0],
            block_size=block_size,
            oracle=oracle,
            logical_order=logical_order,
            guard_rtol=guard_rtol,
            print_tree=False,
        ).search()

    cold_easy = run_one(syndromes["easy"])
    easy_runs = [run_one(syndromes["easy"]) for _ in range(max(1, repeats))]
    hard_runs = [run_one(syndromes["hard"]) for _ in range(max(1, repeats))]

    easy_times = [float(r.stats.wall_seconds) for r in easy_runs]
    hard_times = [float(r.stats.wall_seconds) for r in hard_runs]
    easy_oracles = [int(r.stats.oracle_calls) for r in easy_runs]
    hard_oracles = [int(r.stats.oracle_calls) for r in hard_runs]
    easy_leaves = [int(r.stats.leaves_evaluated) for r in easy_runs]
    hard_leaves = [int(r.stats.leaves_evaluated) for r in hard_runs]

    row = {
        "block_size": int(block_size),
        "logical_order": list(logical_order),
        "topologies": [list(t) for t in topologies],
        "canonical_topologies": [list(canonical_topology(t)) for t in topologies],
        "canonical_block_signature": [
            list(t) for t in canonical_block_signature(logical_order, block_size)
        ],
        "cold_easy_seconds": float(cold_easy.stats.wall_seconds),
        "easy_times": easy_times,
        "hard_times": hard_times,
        "easy_median_seconds": float(statistics.median(easy_times)),
        "hard_median_seconds": float(statistics.median(hard_times)),
        "mean_easy_hard_seconds": float(
            0.5 * (statistics.median(easy_times) + statistics.median(hard_times))
        ),
        "easy_oracle_calls": int(round(statistics.median(easy_oracles))),
        "hard_oracle_calls": int(round(statistics.median(hard_oracles))),
        "easy_leaves": int(round(statistics.median(easy_leaves))),
        "hard_leaves": int(round(statistics.median(hard_leaves))),
        "easy_best_probability": float(easy_runs[-1].best_probability),
        "hard_best_probability": float(hard_runs[-1].best_probability),
        "easy_best_lambda": list(easy_runs[-1].best_lambda),
        "hard_best_lambda": list(hard_runs[-1].best_lambda),
        "workspace": decoder.fast_workspace_info(),
    }

    decoder.release_fast_scratch_workspaces()
    decoder.clear_fast_network_cache()
    try:
        del decoder
    except Exception:
        pass
    aggressive_gpu_cleanup()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "P5.1 canonical-topology scan. Within-block logical permutations "
            "share one canonical path pool, while block order remains a real "
            "BnB search variable. Default Stage A scans b=5..12."
        )
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--block-sizes", default="5,6,7,8,9,10,11,12")
    ap.add_argument(
        "--order-strategies",
        default="identity,reverse,weight-asc,weight-desc,overlap-max,overlap-min",
    )
    ap.add_argument("--random-orders", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--path-trials", type=int, default=6)
    ap.add_argument("--path-samples", type=int, default=64)
    ap.add_argument("--path-benchmark-repeats", type=int, default=2)
    ap.add_argument("--decoder-repeats", type=int, default=2)
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--output-dir", default="p5_1_scan")
    ap.add_argument(
        "--import-profile-dir",
        default=None,
        help=(
            "Optional previous p5_scan directory. All paths_b*.pkl entries are "
            "canonicalized, re-benchmarked, and pooled. This lets the old "
            "identity/reverse searches compete fairly for the same logical set."
        ),
    )
    ap.add_argument(
        "--no-fresh-retune",
        dest="fresh_retune",
        action="store_false",
        help="Use only imported candidates when available; tune only unseen canonical topologies.",
    )
    ap.add_argument("--retain-scratch", action="store_true")
    ap.add_argument("--scratch-retain-budget", default="70%")
    ap.add_argument("--no-constant-cache", dest="constant_cache", action="store_false")
    ap.add_argument("--no-gpu-resident-operands", dest="gpu_resident_operands", action="store_false")
    ap.set_defaults(constant_cache=True, gpu_resident_operands=True, fresh_retune=True)
    args = ap.parse_args()

    H, LZ = build_code()
    k = int(LZ.shape[0])
    block_sizes = parse_block_sizes(args.block_sizes, k=k)
    syndromes = fixed_syndromes(H, LZ)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    import_dir = Path(args.import_profile_dir) if args.import_profile_dir else None
    imported = load_imported_candidates(import_dir)
    if imported:
        print("Imported canonical candidate sets:")
        for key in sorted(imported):
            print(" ", key, "candidates=", len(imported[key]))

    tuning_decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[args.p] * H.shape[1],
        device="cuda",
    )

    # canonical key -> selected best path row
    pool: dict[tuple[int, ...], dict[str, Any]] = {}
    # audit log including every imported/fresh candidate that competed
    pool_audit: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)

    def deterministic_seed(canonical: tuple[int, ...]) -> int:
        # Stable across scan order and Python processes.
        return int(args.seed + 100000 * len(canonical) + sum((i + 1) * (q + 1) * 137 for i, q in enumerate(canonical)))

    def ensure_canonical(canonical: tuple[int, ...]) -> dict[str, Any]:
        canonical = canonical_topology(canonical)
        if canonical in pool:
            return pool[canonical]

        print("\n" + "=" * 80)
        print("P5.1 CANONICAL TOPOLOGY", canonical)
        print("GPU memory before =", gpu_memory_info())
        print("=" * 80)

        candidates: list[dict[str, Any]] = []

        # Previous ordered profiles are only candidate generators. Re-benchmark
        # every imported plan on the canonical sorted-output equation before
        # allowing its old timing to influence selection.
        for imported_entry in imported.get(canonical, []):
            row = benchmark_fixed_plan(
                decoder=tuning_decoder,
                syndrome=syndromes["easy"],
                canonical=canonical,
                candidate=imported_entry,
                memory_limit=args.memory_limit,
                repeats=args.path_benchmark_repeats,
            )
            if row is not None:
                candidates.append(row)
                print(
                    "IMPORT candidate",
                    imported_entry.get("_import_source_open_logicals"),
                    "canonical median=",
                    f"{row['benchmark_median_seconds']:.6f}s",
                    "old median=",
                    imported_entry.get("_import_recorded_benchmark_seconds"),
                )

        # Fresh tuning is run exactly once per canonical logical set, regardless
        # of how many identity/reverse/heuristic configurations refer to it.
        if args.fresh_retune or not candidates:
            fresh = tune_topology(
                decoder=tuning_decoder,
                syndrome=syndromes["easy"],
                open_logicals=canonical,
                memory_limit=args.memory_limit,
                trials=args.path_trials,
                samples=args.path_samples,
                benchmark_repeats=args.path_benchmark_repeats,
                seed0=deterministic_seed(canonical),
            )
            fresh = dict(fresh)
            fresh["_candidate_kind"] = "fresh_canonical_tune"
            fresh["_canonical_open_logicals"] = list(canonical)
            candidates.append(fresh)

        candidates.sort(
            key=lambda row: (
                float(row["benchmark_median_seconds"]),
                float(row.get("opt_cost", float("inf"))),
                int(row.get("num_slices", 1)),
            )
        )
        best = dict(candidates[0])
        best["_canonical_open_logicals"] = list(canonical)
        best["_pool_num_competing_plans"] = len(candidates)
        best["_pool_candidate_summary"] = [
            {
                "kind": row.get("_candidate_kind", "unknown"),
                "source_profile": row.get("_import_source_profile"),
                "source_open_logicals": row.get("_import_source_open_logicals"),
                "benchmark_median_seconds": row.get("benchmark_median_seconds"),
                "recorded_old_benchmark_seconds": row.get("_import_recorded_benchmark_seconds"),
                "seed": row.get("seed"),
                "opt_cost": row.get("opt_cost"),
                "num_slices": row.get("num_slices"),
            }
            for row in candidates
        ]
        pool[canonical] = best
        pool_audit[canonical].extend(best["_pool_candidate_summary"])

        print(
            "CANONICAL BEST",
            canonical,
            "median=",
            best["benchmark_median_seconds"],
            "kind=",
            best.get("_candidate_kind"),
            "source=",
            best.get("_import_source_open_logicals"),
        )
        return best

    def make_profile(block_size: int, order: tuple[int, ...]) -> dict[str, Any]:
        ordered_topos = block_topologies(order, block_size)
        canonical_topos = tuple(canonical_topology(t) for t in ordered_topos)
        paths: dict[tuple[int, ...], dict[str, Any]] = {}
        for topo in canonical_topos:
            paths[topo] = ensure_canonical(topo)
        return {
            "format_version": 2,
            "metadata": {
                "code": "BB72",
                "n": int(H.shape[1]),
                "k": int(LZ.shape[0]),
                "optimization": "v0.4.2/P5.1 canonical path pool",
                "block_size": int(block_size),
                "logical_order": list(order),
                "bnb_topologies": [list(t) for t in ordered_topos],
                "canonical_bnb_topologies": [list(t) for t in canonical_topos],
                "memory_limit": str(args.memory_limit),
                "path_trials": int(args.path_trials),
                "path_samples": int(args.path_samples),
                "path_benchmark_repeats": int(args.path_benchmark_repeats),
                "canonical_path_sharing": True,
                "gpu_name": gpu_name(),
                "cuquantum_version": cuquantum_version(),
            },
            "paths": paths,
        }

    all_rows: list[dict[str, Any]] = []
    config_cache: dict[tuple[int, tuple[tuple[int, ...], ...]], tuple[dict[str, Any], dict[str, Any], Path]] = {}

    def evaluate(block_size: int, order: tuple[int, ...], tag: str):
        signature = canonical_block_signature(order, block_size)
        key = (int(block_size), signature)
        if key in config_cache:
            profile, old_row, profile_path = config_cache[key]
            row = dict(old_row)
            row["tag"] = tag
            row["deduplicated_to"] = old_row.get("tag")
            all_rows.append(row)
            print("DEDUP CONFIG", tag, "->", old_row.get("tag"), "signature=", signature)
            return profile, row, profile_path

        print("\n" + "#" * 80)
        print("P5.1 EVALUATE", tag)
        print("block_size =", block_size)
        print("logical_order =", list(order))
        print("ordered topologies =", list(block_topologies(order, block_size)))
        print("canonical signature =", signature)
        print("#" * 80)
        aggressive_gpu_cleanup(verbose=True)

        t0 = time.perf_counter()
        profile = make_profile(block_size, order)
        profile_compile_seconds = time.perf_counter() - t0
        profile_path = outdir / f"canonical_paths_b{block_size}_{order_label(order)}.pkl"
        with profile_path.open("wb") as fh:
            pickle.dump(profile, fh, protocol=pickle.HIGHEST_PROTOCOL)

        row = benchmark_profile(
            H=H,
            LZ=LZ,
            syndromes=syndromes,
            p=args.p,
            block_size=block_size,
            logical_order=order,
            profile=profile,
            memory_limit=args.memory_limit,
            guard_rtol=args.guard_rtol,
            repeats=args.decoder_repeats,
            gpu_resident_operands=args.gpu_resident_operands,
            constant_cache=args.constant_cache,
            retain_scratch=args.retain_scratch,
            scratch_retain_budget=args.scratch_retain_budget,
        )
        row["tag"] = tag
        row["profile_path"] = str(profile_path)
        row["profile_compile_seconds"] = float(profile_compile_seconds)
        all_rows.append(row)
        config_cache[key] = (profile, row, profile_path)
        print(
            "RESULT",
            tag,
            "easy=", row["easy_median_seconds"],
            "hard=", row["hard_median_seconds"],
            "mean=", row["mean_easy_hard_seconds"],
            "oracle_calls=", (row["easy_oracle_calls"], row["hard_oracle_calls"]),
        )

        checkpoint = {
            "settings": vars(args),
            "completed_results": all_rows,
            "canonical_pool": {
                str(key): {
                    "benchmark_median_seconds": value.get("benchmark_median_seconds"),
                    "candidate_kind": value.get("_candidate_kind"),
                    "source_open_logicals": value.get("_import_source_open_logicals"),
                    "num_competing_plans": value.get("_pool_num_competing_plans"),
                }
                for key, value in pool.items()
            },
        }
        (outdir / "p5_1_scan_checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
        aggressive_gpu_cleanup(verbose=True)
        return config_cache[key]

    # Stage A: continue the block-size curve beyond the previous b<=6 scan.
    identity = tuple(range(k))
    stage_a: list[dict[str, Any]] = []
    for b in block_sizes:
        _, row, _ = evaluate(b, identity, f"stageA-b{b}-identity")
        stage_a.append(row)
    best_a = min(stage_a, key=lambda r: r["mean_easy_hard_seconds"])
    best_b = int(best_a["block_size"])
    print("\nSTAGE A BEST block_size =", best_b, "mean =", best_a["mean_easy_hard_seconds"])

    # Stage B: compare groupings/search order with the same canonical path pool.
    # Thus identity vs reverse cannot win merely because one happened to get a
    # better random contraction path for the same logical set.
    strategy_names = [s.strip() for s in args.order_strategies.split(",") if s.strip()]
    orders: list[tuple[str, tuple[int, ...]]] = []
    seen_signatures: set[tuple[tuple[int, ...], ...]] = set()
    for i, name in enumerate(strategy_names):
        order = named_logical_order(
            name,
            k=k,
            logical_obs=LZ,
            block_size=best_b,
            seed=args.seed + i,
        )
        sig = canonical_block_signature(order, best_b)
        if sig not in seen_signatures:
            orders.append((name, order))
            seen_signatures.add(sig)
        else:
            print("SKIP equivalent within-block permutation:", name, "signature=", sig)
    for j in range(max(0, args.random_orders)):
        order = named_logical_order(
            "random",
            k=k,
            logical_obs=LZ,
            block_size=best_b,
            seed=args.seed + 50000 + j,
        )
        sig = canonical_block_signature(order, best_b)
        if sig not in seen_signatures:
            orders.append((f"random-{j}", order))
            seen_signatures.add(sig)

    stage_b: list[dict[str, Any]] = []
    for name, order in orders:
        _, row, _ = evaluate(best_b, order, f"stageB-b{best_b}-{name}")
        stage_b.append(row)

    unique_rows = [r for r in all_rows if "deduplicated_to" not in r]
    best = min(unique_rows, key=lambda r: r["mean_easy_hard_seconds"])

    # Ensure the full 12-logical reference plan exists in the canonical pool and
    # attach it to the winning profile for the existing correctness runner.
    full_topology = tuple(range(k))
    ensure_canonical(full_topology)
    best_key = (
        int(best["block_size"]),
        tuple(tuple(int(v) for v in t) for t in best["canonical_block_signature"]),
    )
    best_profile, _, best_profile_path = config_cache[best_key]
    best_profile["paths"][full_topology] = pool[full_topology]
    best_profile["metadata"]["full_reference_topology_added_after_scan"] = True
    with best_profile_path.open("wb") as fh:
        pickle.dump(best_profile, fh, protocol=pickle.HIGHEST_PROTOCOL)

    pool_payload = {
        "format_version": 1,
        "metadata": {
            "code": "BB72",
            "purpose": "P5.1 canonical path pool",
            "canonical_key": "sorted open-logical set",
            "memory_limit": str(args.memory_limit),
            "gpu_name": gpu_name(),
            "cuquantum_version": cuquantum_version(),
            "import_profile_dir": str(import_dir) if import_dir else None,
        },
        "paths": pool,
    }
    pool_path = outdir / "canonical_path_pool.pkl"
    with pool_path.open("wb") as fh:
        pickle.dump(pool_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "format_version": 2,
        "gpu_name": gpu_name(),
        "cuquantum_version": cuquantum_version(),
        "settings": vars(args),
        "stage_a_best_block_size": best_b,
        "best": best,
        "canonical_pool_path": str(pool_path),
        "canonical_pool_summary": {
            str(key): {
                "benchmark_median_seconds": value.get("benchmark_median_seconds"),
                "candidate_kind": value.get("_candidate_kind"),
                "source_profile": value.get("_import_source_profile"),
                "source_open_logicals": value.get("_import_source_open_logicals"),
                "num_competing_plans": value.get("_pool_num_competing_plans"),
                "candidate_summary": value.get("_pool_candidate_summary"),
            }
            for key, value in sorted(pool.items())
        },
        "results": sorted(unique_rows, key=lambda r: r["mean_easy_hard_seconds"]),
    }
    summary_path = outdir / "p5_1_scan_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    best_config = {
        "block_size": int(best["block_size"]),
        "logical_order": list(best["logical_order"]),
        "canonical_block_signature": best["canonical_block_signature"],
        "profile_path": str(best_profile_path),
        "easy_median_seconds": best["easy_median_seconds"],
        "hard_median_seconds": best["hard_median_seconds"],
        "mean_easy_hard_seconds": best["mean_easy_hard_seconds"],
    }
    best_config_path = outdir / "p5_1_best_config.json"
    best_config_path.write_text(json.dumps(best_config, indent=2))

    try:
        tuning_decoder.clear_fast_network_cache()
        del tuning_decoder
    except Exception:
        pass
    aggressive_gpu_cleanup(verbose=True)

    print("\n" + "=" * 80)
    print("P5.1 BEST CONFIG")
    print(json.dumps(best_config, indent=2))
    print("results =", summary_path)
    print("canonical pool =", pool_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
