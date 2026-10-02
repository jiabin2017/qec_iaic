from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from exact_block_bnb import ExactBlockBnB
from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder


def sync_cuda() -> None:
    try:
        import cupy as cp
        cp.cuda.Device().synchronize()
        return
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_bb72():
    from sympy.abc import x, y
    from qldpc import codes
    from qldpc.objects import Pauli

    code = codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    )
    H = np.asarray(code.matrix_z, dtype=np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    LX = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    if not np.array_equal((LX @ LZ.T) % 2, np.eye(LZ.shape[0], dtype=np.uint8)):
        raise RuntimeError("LX @ LZ.T != I over GF(2)")
    return H, LZ, LX


def make_case(H: np.ndarray, LZ: np.ndarray, error_indices: list[int]) -> dict:
    e = np.zeros(H.shape[1], dtype=np.uint8)
    e[error_indices] = 1
    syndrome = (H @ e) % 2
    true_lambda = (LZ @ e) % 2
    return {
        "error_indices": list(error_indices),
        "error_weight": int(e.sum()),
        "syndrome": syndrome.astype(np.uint8),
        "syndrome_weight": int(syndrome.sum()),
        "true_lambda": tuple(int(x) for x in true_lambda.tolist()),
    }


def make_decoder(H, LZ, *, p: float, profile: Path, memory_limit: str):
    dec = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[float(p)] * int(H.shape[1]),
        device="cuda",
    )
    # root single, tail single, plus several P6 batch-size variants.
    dec.set_fast_cache_max_entries(16)
    info = dec.load_fast_path_profile(profile)
    meta = info.get("metadata", {})
    if meta.get("memory_limit") is not None and str(meta.get("memory_limit")) != str(memory_limit):
        print(
            f"WARNING: profile memory_limit={meta.get('memory_limit')}, "
            f"run memory_limit={memory_limit}"
        )
    return dec, info


def decode_once(
    *,
    decoder,
    syndrome: np.ndarray,
    parent_batch_size: int,
    memory_limit: str,
    guard_rtol: float,
    constant_cache: bool,
    gpu_resident_operands: bool,
    retain_scratch: bool,
    scratch_retain_budget: str,
):
    syndrome_list = syndrome.astype(float).tolist()
    calls: Counter[str] = Counter()

    def oracle(fixed_logicals, open_logicals):
        key = "single:" + "-".join(str(int(i)) for i in open_logicals)
        calls[key] += 1
        sync_cuda()
        out = decoder.contract_logical_mass_fast(
            syndrome=syndrome_list,
            fixed_logicals=dict(fixed_logicals),
            open_logicals=list(open_logicals),
            autotune_iterations=0,
            memory_limit=memory_limit,
            release_workspace=not retain_scratch,
            use_constant_qualifiers=constant_cache,
            use_gpu_resident_operands=gpu_resident_operands,
            retain_constant_cache=constant_cache,
            retain_scratch_workspace=retain_scratch,
            scratch_retention_budget=scratch_retain_budget,
        )
        sync_cuda()
        return out

    batch_oracle = None
    if int(parent_batch_size) > 1:
        def _batch_oracle(fixed_batch, open_logicals):
            key = (
                "batch" + str(len(fixed_batch)) + ":"
                + "-".join(str(int(i)) for i in open_logicals)
            )
            calls[key] += 1
            sync_cuda()
            out = decoder.contract_logical_mass_batch_fast(
                syndrome=syndrome_list,
                fixed_logicals_batch=[dict(x) for x in fixed_batch],
                open_logicals=list(open_logicals),
                autotune_iterations=0,
                memory_limit=memory_limit,
                release_workspace=not retain_scratch,
                use_constant_qualifiers=constant_cache,
                use_gpu_resident_operands=gpu_resident_operands,
                retain_constant_cache=constant_cache,
                retain_scratch_workspace=retain_scratch,
                scratch_retention_budget=scratch_retain_budget,
                fallback_to_sequential=True,
            )
            sync_cuda()
            return out
        batch_oracle = _batch_oracle

    searcher = ExactBlockBnB(
        k=12,
        block_size=9,
        oracle=oracle,
        batch_oracle=batch_oracle,
        parent_batch_size=int(parent_batch_size),
        logical_order=list(range(12)),
        guard_rtol=float(guard_rtol),
        print_tree=False,
    )
    result = searcher.search()
    return result, dict(calls)


def _mode_label(batch_size: int) -> str:
    return "sequential" if int(batch_size) <= 1 else f"batch{int(batch_size)}"


def _median(xs):
    return float(statistics.median(float(x) for x in xs))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "BB72 v0.6/P6 A/B benchmark: v0.5 sequential parents versus "
            "same-level final-parent micro-batching on the original easy/hard syndromes."
        )
    )
    ap.add_argument(
        "--profile",
        default="profiles/canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl",
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--parent-execution",
        choices=["compare", "sequential", "batched"],
        default="compare",
        help="compare = sequential plus every --batch-sizes entry",
    )
    ap.add_argument("--parent-batch-size", type=int, default=2)
    ap.add_argument("--batch-sizes", default="2,4")
    ap.add_argument("--output-dir", default="p6_easy_hard_results")
    ap.add_argument("--no-constant-cache", dest="constant_cache", action="store_false")
    ap.add_argument("--no-gpu-resident-operands", dest="gpu_resident_operands", action="store_false")
    ap.add_argument("--retain-scratch", action="store_true")
    ap.add_argument("--scratch-retain-budget", default="70%")
    ap.set_defaults(constant_cache=True, gpu_resident_operands=True)
    args = ap.parse_args()

    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be >=1 and warmup >=0")

    if args.parent_execution == "sequential":
        batch_sizes = [1]
    elif args.parent_execution == "batched":
        if args.parent_batch_size < 2:
            raise ValueError("--parent-batch-size must be >=2 in batched mode")
        batch_sizes = [int(args.parent_batch_size)]
    else:
        requested = [int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()]
        if any(x < 2 for x in requested):
            raise ValueError("--batch-sizes entries must be >=2")
        batch_sizes = [1] + list(dict.fromkeys(requested))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = Path(args.profile).resolve()
    if not profile.exists():
        raise FileNotFoundError(profile)

    H, LZ, LX = build_bb72()
    cases = {
        "easy": make_case(H, LZ, [28]),
        "hard": make_case(H, LZ, [0, 27, 30]),
    }

    print("===== BB72 v0.6 / P6 =====")
    print("profile =", profile)
    print("p =", args.p)
    print("batch sizes =", batch_sizes)
    print("repeats =", args.repeats, "warmup =", args.warmup)
    print("P2 constant cache =", args.constant_cache)
    print("P3 GPU resident =", args.gpu_resident_operands)
    print("P4 retain scratch =", args.retain_scratch)
    for name, c in cases.items():
        print(
            f"{name}: error={c['error_indices']} wt={c['error_weight']} "
            f"syndrome_wt={c['syndrome_weight']} "
            f"true_lambda={''.join(map(str, c['true_lambda']))}"
        )

    rows: list[dict] = []
    first_results: dict[tuple[str, str], object] = {}

    for batch_size in batch_sizes:
        mode = _mode_label(batch_size)
        cleanup_gpu()
        decoder, profile_info = make_decoder(
            H, LZ, p=args.p, profile=profile,
            memory_limit=args.memory_limit,
        )
        print("\n" + "=" * 72)
        print("MODE", mode, "parent_batch_size=", batch_size)
        print("profile topologies =", profile_info.get("open_logicals"))

        # Warm both cases. For P6 this is important: the hard warmup prepares
        # the new batch Network, so measured times are online steady-state and
        # exclude one-time contract_path/Network construction.
        for _ in range(args.warmup):
            for case_name in ("easy", "hard"):
                decode_once(
                    decoder=decoder,
                    syndrome=cases[case_name]["syndrome"],
                    parent_batch_size=batch_size,
                    memory_limit=args.memory_limit,
                    guard_rtol=args.guard_rtol,
                    constant_cache=args.constant_cache,
                    gpu_resident_operands=args.gpu_resident_operands,
                    retain_scratch=args.retain_scratch,
                    scratch_retain_budget=args.scratch_retain_budget,
                )

        for case_name in ("easy", "hard"):
            times = []
            oracle_calls = []
            batch_calls = []
            batched_parents = []
            wasted_parents = []
            leaves = []
            pruned = []
            call_counters = []
            results = []
            for rep in range(args.repeats):
                result, calls = decode_once(
                    decoder=decoder,
                    syndrome=cases[case_name]["syndrome"],
                    parent_batch_size=batch_size,
                    memory_limit=args.memory_limit,
                    guard_rtol=args.guard_rtol,
                    constant_cache=args.constant_cache,
                    gpu_resident_operands=args.gpu_resident_operands,
                    retain_scratch=args.retain_scratch,
                    scratch_retain_budget=args.scratch_retain_budget,
                )
                results.append(result)
                times.append(result.stats.wall_seconds)
                oracle_calls.append(result.stats.oracle_calls)
                batch_calls.append(result.stats.batch_oracle_calls)
                batched_parents.append(result.stats.batched_parent_queries)
                wasted_parents.append(result.stats.wasted_batched_parent_queries)
                leaves.append(result.stats.leaves_evaluated)
                pruned.append(result.stats.pruned_leaves)
                call_counters.append(calls)
                print(
                    f"{mode:10s} {case_name:4s} rep={rep+1}: "
                    f"wall={result.stats.wall_seconds:.6f}s "
                    f"physical_oracles={result.stats.oracle_calls} "
                    f"batch_calls={result.stats.batch_oracle_calls} "
                    f"batched_parents={result.stats.batched_parent_queries} "
                    f"wasted={result.stats.wasted_batched_parent_queries} "
                    f"leaves={result.stats.leaves_evaluated} "
                    f"P={result.best_probability:.15g} "
                    f"lambda={''.join(map(str, result.best_lambda))}"
                )

            first_results[(mode, case_name)] = results[0]
            row = {
                "mode": mode,
                "parent_batch_size": int(batch_size),
                "case": case_name,
                "wall_median_s": _median(times),
                "wall_min_s": float(min(times)),
                "wall_max_s": float(max(times)),
                "oracle_calls_median": _median(oracle_calls),
                "batch_calls_median": _median(batch_calls),
                "batched_parent_queries_median": _median(batched_parents),
                "wasted_batched_parents_median": _median(wasted_parents),
                "leaves_median": _median(leaves),
                "pruned_leaves_median": _median(pruned),
                "best_probability": float(results[0].best_probability),
                "best_lambda": "".join(map(str, results[0].best_lambda)),
                "true_lambda": "".join(map(str, cases[case_name]["true_lambda"])),
                "call_counter_example": json.dumps(call_counters[0], sort_keys=True),
            }
            rows.append(row)

        print("fast cache info after mode:")
        for info in decoder.fast_cache_info():
            print(info)
        try:
            decoder.clear_fast_network_cache()
        except Exception:
            pass
        del decoder
        cleanup_gpu()

    # Correctness / numerical consistency versus sequential v0.5 behavior.
    baseline = {r["case"]: r for r in rows if r["mode"] == "sequential"}
    checks = []
    if baseline:
        for row in rows:
            if row["mode"] == "sequential":
                continue
            ref = baseline[row["case"]]
            prob_diff = abs(float(row["best_probability"]) - float(ref["best_probability"]))
            easy_lambda_equal = (
                row["best_lambda"] == ref["best_lambda"] if row["case"] == "easy" else None
            )
            ok = prob_diff <= max(1e-12, args.guard_rtol * 100.0)
            if row["case"] == "easy":
                ok = ok and bool(easy_lambda_equal)
            checks.append({
                "mode": row["mode"],
                "case": row["case"],
                "probability_abs_diff_vs_sequential": prob_diff,
                "easy_lambda_equal": easy_lambda_equal,
                "pass": bool(ok),
            })

    csv_path = out_dir / "p6_easy_hard_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "version": "v0.6",
        "optimization": "P6 same-level final-parent micro-batching",
        "bb72": {"n": 72, "k": 12, "block_size": 9, "logical_order": list(range(12))},
        "p": args.p,
        "profile": str(profile),
        "settings": {
            "memory_limit": args.memory_limit,
            "guard_rtol": args.guard_rtol,
            "constant_cache": args.constant_cache,
            "gpu_resident_operands": args.gpu_resident_operands,
            "retain_scratch": args.retain_scratch,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "cases": {
            name: {
                "error_indices": c["error_indices"],
                "error_weight": c["error_weight"],
                "syndrome_weight": c["syndrome_weight"],
                "true_lambda": "".join(map(str, c["true_lambda"])),
            }
            for name, c in cases.items()
        },
        "rows": rows,
        "consistency_checks": checks,
    }
    json_path = out_dir / "p6_easy_hard_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print("\n===== SUMMARY =====")
    for row in rows:
        print(
            f"{row['mode']:10s} {row['case']:4s}: "
            f"median={row['wall_median_s']:.6f}s "
            f"physical_oracles={row['oracle_calls_median']:.1f} "
            f"batch_calls={row['batch_calls_median']:.1f} "
            f"batched_parents={row['batched_parent_queries_median']:.1f}"
        )
    if baseline:
        print("\nSpeedups versus sequential:")
        for row in rows:
            if row["mode"] == "sequential":
                continue
            ref = baseline[row["case"]]
            speedup = float(ref["wall_median_s"]) / float(row["wall_median_s"])
            print(f"{row['mode']:10s} {row['case']:4s}: {speedup:.4f}x")
    print("\nconsistency checks =", checks)
    print("wrote", csv_path)
    print("wrote", json_path)

    if any(not row["pass"] for row in checks):
        raise SystemExit("P6 numerical consistency check failed; inspect JSON before trusting timings")


if __name__ == "__main__":
    main()
