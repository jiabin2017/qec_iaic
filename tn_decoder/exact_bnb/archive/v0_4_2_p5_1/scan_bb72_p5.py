from __future__ import annotations

import argparse
import gc
import json
import pickle
import statistics
import time
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
    named_logical_order,
    order_label,
    parse_block_sizes,
)
from tune_bb72_p5_paths import (
    tune_topology,
    gpu_name,
    cuquantum_version,
    aggressive_gpu_cleanup,
    gpu_memory_info,
)
from run_bb72_exact_bnb_p5 import find_weight3_logical_collision, sync_cuda


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
    # Keep the same deterministic easy case used in the original BB72 work.
    easy_error = np.zeros(H.shape[1], dtype=np.uint8)
    easy_error[28] = 1
    easy_syndrome = ((H @ easy_error) % 2).astype(float)
    hard_error, hard_competitor, hard_syn_u8, hard_lambda, competitor_lambda = \
        find_weight3_logical_collision(H, LZ)
    return {
        "easy": easy_syndrome,
        "hard": hard_syn_u8.astype(float),
    }


def compile_profile(
    *,
    H: np.ndarray,
    LZ: np.ndarray,
    syndrome: np.ndarray,
    p: float,
    block_size: int,
    logical_order: tuple[int, ...],
    memory_limit: str,
    trials: int,
    samples: int,
    benchmark_repeats: int,
    seed: int,
) -> dict[str, Any]:
    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[p] * H.shape[1],
        device="cuda",
    )
    entries: dict[tuple[int, ...], dict[str, Any]] = {}
    compile_start = time.perf_counter()
    topologies = block_topologies(logical_order, block_size)
    for topo_i, open_logicals in enumerate(topologies):
        entries[open_logicals] = tune_topology(
            decoder=decoder,
            syndrome=syndrome,
            open_logicals=open_logicals,
            memory_limit=memory_limit,
            trials=trials,
            samples=samples,
            benchmark_repeats=benchmark_repeats,
            seed0=seed + 1000 * topo_i,
        )
    decoder.clear_fast_network_cache()
    try:
        del decoder
    except Exception:
        pass
    aggressive_gpu_cleanup()
    return {
        "format_version": 2,
        "metadata": {
            "code": "BB72",
            "n": int(H.shape[1]),
            "k": int(LZ.shape[0]),
            "optimization": "v0.4/P5 scan",
            "block_size": int(block_size),
            "logical_order": list(logical_order),
            "bnb_topologies": [list(t) for t in topologies],
            "memory_limit": str(memory_limit),
            "trials": int(trials),
            "samples": int(samples),
            "benchmark_repeats": int(benchmark_repeats),
            "compile_wall_seconds": float(time.perf_counter() - compile_start),
            "gpu_name": gpu_name(),
            "cuquantum_version": cuquantum_version(),
        },
        "paths": entries,
    }


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

        result = ExactBlockBnB(
            k=LZ.shape[0],
            block_size=block_size,
            oracle=oracle,
            logical_order=logical_order,
            guard_rtol=guard_rtol,
            print_tree=False,
        ).search()
        return result

    # One cold easy pass prepares every topology actually visited by the easy
    # search.  Then measure steady-state easy.  The first hard pass is already a
    # cross-syndrome online measurement for all shared prepared topologies.
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
            "v0.4/P5 two-stage autotuner. Stage A scans block size with the "
            "identity logical order. Stage B scans logical group/order "
            "heuristics at the best block size."
        )
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--block-sizes", default="1,2,3,4,6")
    ap.add_argument(
        "--order-strategies",
        default="identity,reverse,weight-asc,weight-desc,overlap-max,overlap-min",
    )
    ap.add_argument("--random-orders", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--path-trials", type=int, default=3)
    ap.add_argument("--path-samples", type=int, default=16)
    ap.add_argument("--path-benchmark-repeats", type=int, default=1)
    ap.add_argument("--decoder-repeats", type=int, default=2)
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--output-dir", default="p5_scan")
    ap.add_argument(
        "--no-reuse-existing-profiles",
        dest="reuse_existing_profiles",
        action="store_false",
        help="Ignore compatible paths_b*.pkl files already present in output-dir.",
    )
    ap.add_argument("--retain-scratch", action="store_true")
    ap.add_argument("--scratch-retain-budget", default="70%")
    ap.add_argument("--no-constant-cache", dest="constant_cache", action="store_false")
    ap.add_argument("--no-gpu-resident-operands", dest="gpu_resident_operands", action="store_false")
    ap.set_defaults(
        constant_cache=True,
        gpu_resident_operands=True,
        reuse_existing_profiles=True,
    )
    args = ap.parse_args()

    H, LZ = build_code()
    k = LZ.shape[0]
    block_sizes = parse_block_sizes(args.block_sizes, k=k)
    syndromes = fixed_syndromes(H, LZ)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[int, tuple[int, ...]], tuple[dict[str, Any], dict[str, Any], Path]] = {}
    all_rows: list[dict[str, Any]] = []

    def evaluate(block_size: int, order: tuple[int, ...], tag: str):
        key = (int(block_size), tuple(order))
        if key in cache:
            return cache[key]
        print("\n" + "#" * 80)
        print("P5 EVALUATE", tag)
        print("block_size =", block_size)
        print("logical_order =", list(order))
        print("topologies =", list(block_topologies(order, block_size)))
        print("#" * 80)
        # Long multi-topology scans can leave large freed blocks cached in the
        # CUDA allocator. Clean before every independent configuration.
        aggressive_gpu_cleanup(verbose=True)

        profile_path = outdir / f"paths_b{block_size}_{order_label(order)}.pkl"
        profile = None
        if args.reuse_existing_profiles and profile_path.exists():
            try:
                with profile_path.open("rb") as fh:
                    candidate = pickle.load(fh)
                meta = candidate.get("metadata", {})
                compatible = (
                    int(meta.get("block_size", -1)) == int(block_size)
                    and tuple(int(v) for v in meta.get("logical_order", [])) == tuple(order)
                    and str(meta.get("memory_limit", "")) == str(args.memory_limit)
                )
                expected_topologies = set(block_topologies(order, block_size))
                have_topologies = set(tuple(int(v) for v in key) for key in candidate.get("paths", {}))
                compatible = compatible and expected_topologies.issubset(have_topologies)
                if compatible:
                    profile = candidate
                    print("REUSE existing compatible path profile:", profile_path)
                else:
                    print("IGNORE incompatible existing profile:", profile_path)
            except Exception as exc:
                print("IGNORE unreadable existing profile:", profile_path, exc)

        if profile is None:
            profile = compile_profile(
                H=H,
                LZ=LZ,
                syndrome=syndromes["easy"],
                p=args.p,
                block_size=block_size,
                logical_order=order,
                memory_limit=args.memory_limit,
                trials=args.path_trials,
                samples=args.path_samples,
                benchmark_repeats=args.path_benchmark_repeats,
                seed=args.seed + 10000 * len(cache),
            )
            with profile_path.open("wb") as fh:
                pickle.dump(profile, fh, protocol=pickle.HIGHEST_PROTOCOL)

        aggressive_gpu_cleanup(verbose=True)
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
        row["profile_compile_seconds"] = float(profile["metadata"]["compile_wall_seconds"])
        all_rows.append(row)
        cache[key] = (profile, row, profile_path)
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
        }
        (outdir / "p5_scan_checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
        aggressive_gpu_cleanup(verbose=True)
        return cache[key]

    # Stage A: block-size scan under one fixed ordering.
    identity = tuple(range(k))
    stage_a = []
    for b in block_sizes:
        _, row, _ = evaluate(b, identity, f"stageA-b{b}-identity")
        stage_a.append(row)
    best_a = min(stage_a, key=lambda r: r["mean_easy_hard_seconds"])
    best_b = int(best_a["block_size"])
    print("\nSTAGE A BEST block_size =", best_b, "mean =", best_a["mean_easy_hard_seconds"])

    # Stage B: grouping/order scan at the best block size.  Because each oracle
    # emits an entire block at once, within-block bit order matters much less than
    # which logicals are grouped together and which block is visited first.
    strategy_names = [s.strip() for s in args.order_strategies.split(",") if s.strip()]
    orders: list[tuple[str, tuple[int, ...]]] = []
    seen_orders: set[tuple[int, ...]] = set()
    for i, name in enumerate(strategy_names):
        order = named_logical_order(
            name,
            k=k,
            logical_obs=LZ,
            block_size=best_b,
            seed=args.seed + i,
        )
        if order not in seen_orders:
            orders.append((name, order))
            seen_orders.add(order)
    for j in range(max(0, args.random_orders)):
        order = named_logical_order(
            "random",
            k=k,
            logical_obs=LZ,
            block_size=best_b,
            seed=args.seed + 50000 + j,
        )
        if order not in seen_orders:
            orders.append((f"random-{j}", order))
            seen_orders.add(order)

    stage_b = []
    for name, order in orders:
        _, row, _ = evaluate(best_b, order, f"stageB-b{best_b}-{name}")
        stage_b.append(row)
    best = min(stage_b or stage_a, key=lambda r: r["mean_easy_hard_seconds"])

    # Add the full-4096 reference topology only to the winning profile so the
    # final correctness run can reuse a compiled path without paying this
    # offline cost for every P5 candidate.
    best_key = (int(best["block_size"]), tuple(int(v) for v in best["logical_order"]))
    best_profile, _, best_profile_path = cache[best_key]
    full_topology = tuple(range(k))
    if full_topology not in best_profile["paths"]:
        ref_decoder = MultiLogicalTensorNetworkDecoder(
            H=H, logical_obs=LZ, noise_model=[args.p] * H.shape[1], device="cuda"
        )
        best_profile["paths"][full_topology] = tune_topology(
            decoder=ref_decoder,
            syndrome=syndromes["easy"],
            open_logicals=full_topology,
            memory_limit=args.memory_limit,
            trials=args.path_trials,
            samples=args.path_samples,
            benchmark_repeats=args.path_benchmark_repeats,
            seed0=args.seed + 900000,
        )
        best_profile["metadata"]["full_reference_topology_added_after_scan"] = True
        with best_profile_path.open("wb") as fh:
            pickle.dump(best_profile, fh, protocol=pickle.HIGHEST_PROTOCOL)
        ref_decoder.clear_fast_network_cache()
        try:
            del ref_decoder
        except Exception:
            pass
        aggressive_gpu_cleanup(verbose=True)

    summary = {
        "format_version": 1,
        "gpu_name": gpu_name(),
        "cuquantum_version": cuquantum_version(),
        "settings": vars(args),
        "stage_a_best_block_size": best_b,
        "best": best,
        "results": sorted(all_rows, key=lambda r: r["mean_easy_hard_seconds"]),
    }
    summary_path = outdir / "p5_scan_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    best_config = {
        "block_size": int(best["block_size"]),
        "logical_order": list(best["logical_order"]),
        "profile_path": best["profile_path"],
        "easy_median_seconds": best["easy_median_seconds"],
        "hard_median_seconds": best["hard_median_seconds"],
        "mean_easy_hard_seconds": best["mean_easy_hard_seconds"],
    }
    (outdir / "p5_best_config.json").write_text(json.dumps(best_config, indent=2))

    print("\n" + "=" * 80)
    print("P5 BEST CONFIG")
    print(json.dumps(best_config, indent=2))
    print("results =", summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
