from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from exact_block_bnb import ExactBlockBnB
from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder
from p7_environment_reuse import LogicalTailEnvironmentEngine


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
    dec.set_fast_cache_max_entries(8)
    info = dec.load_fast_path_profile(profile)
    return dec, info


def standard_tail(
    decoder,
    syndrome_list,
    fixed_logicals,
    open_logicals,
    *,
    memory_limit,
    constant_cache,
    gpu_resident_operands,
):
    return decoder.contract_logical_mass_fast(
        syndrome=syndrome_list,
        fixed_logicals=dict(fixed_logicals),
        open_logicals=list(open_logicals),
        autotune_iterations=0,
        memory_limit=memory_limit,
        release_workspace=True,
        use_constant_qualifiers=constant_cache,
        use_gpu_resident_operands=gpu_resident_operands,
        retain_constant_cache=constant_cache,
        retain_scratch_workspace=False,
    )


def decode_once(
    *,
    decoder,
    p7_engine,
    syndrome: np.ndarray,
    mode: str,
    memory_limit: str,
    guard_rtol: float,
    constant_cache: bool,
    gpu_resident_operands: bool,
):
    syndrome_list = syndrome.astype(float).tolist()
    calls: Counter[str] = Counter()
    tail_count = 0
    session = p7_engine.new_session(syndrome_list) if mode == "p7_always" else None

    def oracle(fixed_logicals, open_logicals):
        nonlocal tail_count, session
        open_tuple = tuple(int(i) for i in open_logicals)
        is_tail = open_tuple == (9, 10, 11)

        if is_tail:
            tail_count += 1
            if mode == "p7_always":
                calls["p7_tail"] += 1
                sync_cuda()
                out = session.contract(dict(fixed_logicals))
                sync_cuda()
                return out
            if mode == "p7_adaptive" and tail_count >= 2:
                if session is None:
                    session = p7_engine.new_session(syndrome_list)
                calls["p7_tail"] += 1
                sync_cuda()
                out = session.contract(dict(fixed_logicals))
                sync_cuda()
                return out

        calls["cutn:" + "-".join(map(str, open_tuple))] += 1
        sync_cuda()
        out = standard_tail(
            decoder,
            syndrome_list,
            fixed_logicals,
            open_tuple,
            memory_limit=memory_limit,
            constant_cache=constant_cache,
            gpu_resident_operands=gpu_resident_operands,
        )
        sync_cuda()
        return out

    searcher = ExactBlockBnB(
        k=12,
        block_size=9,
        oracle=oracle,
        parent_batch_size=1,  # P6 deliberately disabled for the P7 baseline.
        logical_order=list(range(12)),
        guard_rtol=float(guard_rtol),
        print_tree=False,
    )
    result = searcher.search()
    p7_stats = {} if session is None else session.stats.as_dict()
    return result, dict(calls), p7_stats


def med(xs):
    return float(statistics.median(float(x) for x in xs))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "BB72 v0.7/P7: compare validated v0.6 parent_batch_size=1 baseline "
            "against contraction-tree shared-environment reuse on the b9 tail."
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
        "--modes",
        default="baseline,p7_always,p7_adaptive",
        help="comma separated: baseline,p7_always,p7_adaptive",
    )
    ap.add_argument("--p7-cache-budget", default="8GB")
    ap.add_argument("--p7-max-entry", default="2GB")
    ap.add_argument("--p7-max-dependency-bits", type=int, default=6)
    ap.add_argument("--output-dir", default="p7_easy_hard_results")
    ap.add_argument("--no-constant-cache", dest="constant_cache", action="store_false")
    ap.add_argument(
        "--no-gpu-resident-operands",
        dest="gpu_resident_operands",
        action="store_false",
    )
    ap.set_defaults(constant_cache=True, gpu_resident_operands=True)
    args = ap.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    allowed = {"baseline", "p7_always", "p7_adaptive"}
    if not modes or any(x not in allowed for x in modes):
        raise ValueError(f"--modes must be chosen from {sorted(allowed)}")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats>=1 and warmup>=0 required")

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

    print("===== BB72 v0.7 / P7 environment reuse =====")
    print("profile =", profile)
    print("p =", args.p)
    print("modes =", modes)
    print("repeats =", args.repeats, "warmup =", args.warmup)
    print("P6 parent_batch_size = 1 (disabled)")
    print("P2 constant cache =", args.constant_cache)
    print("P3 GPU resident =", args.gpu_resident_operands)
    print(
        "P7 cache =",
        args.p7_cache_budget,
        "max_entry =",
        args.p7_max_entry,
        "max_dependency_bits =",
        args.p7_max_dependency_bits,
    )
    for name, c in cases.items():
        print(
            f"{name}: error={c['error_indices']} wt={c['error_weight']} "
            f"syndrome_wt={c['syndrome_weight']} "
            f"true_lambda={''.join(map(str, c['true_lambda']))}"
        )

    rows = []
    reference = {}

    for mode in modes:
        cleanup_gpu()
        decoder, profile_info = make_decoder(
            H, LZ, p=args.p, profile=profile, memory_limit=args.memory_limit
        )
        # Build the topology-only P7 plan once.  Sessions/caches are fresh for
        # every decode so no result is memoized across benchmark repetitions.
        p7_engine = LogicalTailEnvironmentEngine(
            decoder,
            open_logicals=(9, 10, 11),
            cache_budget=args.p7_cache_budget,
            max_entry_bytes=args.p7_max_entry,
            max_dependency_bits=args.p7_max_dependency_bits,
        )
        print("\n" + "=" * 78)
        print("MODE", mode)
        print("profile topologies =", profile_info.get("open_logicals"))
        print("P7 engine =", p7_engine.info())

        for _ in range(args.warmup):
            for case_name in ("easy", "hard"):
                decode_once(
                    decoder=decoder,
                    p7_engine=p7_engine,
                    syndrome=cases[case_name]["syndrome"],
                    mode=mode,
                    memory_limit=args.memory_limit,
                    guard_rtol=args.guard_rtol,
                    constant_cache=args.constant_cache,
                    gpu_resident_operands=args.gpu_resident_operands,
                )

        for case_name in ("easy", "hard"):
            results = []
            times = []
            pair_exec = []
            hits = []
            misses = []
            reused = []
            peak = []
            calls_example = None
            stats_example = None

            for rep in range(args.repeats):
                result, calls, p7s = decode_once(
                    decoder=decoder,
                    p7_engine=p7_engine,
                    syndrome=cases[case_name]["syndrome"],
                    mode=mode,
                    memory_limit=args.memory_limit,
                    guard_rtol=args.guard_rtol,
                    constant_cache=args.constant_cache,
                    gpu_resident_operands=args.gpu_resident_operands,
                )
                results.append(result)
                times.append(result.stats.wall_seconds)
                pair_exec.append(p7s.get("pair_contractions_executed", 0))
                hits.append(p7s.get("cache_hits", 0))
                misses.append(p7s.get("cache_misses", 0))
                reused.append(p7s.get("bytes_reused", 0))
                peak.append(p7s.get("cache_peak_bytes", 0))
                calls_example = calls
                stats_example = p7s
                print(
                    f"{mode:12s} {case_name:4s} rep={rep+1}: "
                    f"wall={result.stats.wall_seconds:.6f}s "
                    f"oracles={result.stats.oracle_calls} "
                    f"leaves={result.stats.leaves_evaluated} "
                    f"P={result.best_probability:.15g} "
                    f"lambda={''.join(map(str, result.best_lambda))} "
                    f"p7_pair_exec={p7s.get('pair_contractions_executed',0)} "
                    f"p7_hits={p7s.get('cache_hits',0)} "
                    f"reused_MB={p7s.get('bytes_reused',0)/1024**2:.1f}"
                )

            r0 = results[0]
            if mode == "baseline":
                reference[case_name] = r0
            row = {
                "mode": mode,
                "case": case_name,
                "wall_median_s": med(times),
                "wall_min_s": float(min(times)),
                "wall_max_s": float(max(times)),
                "oracle_calls_median": med([r.stats.oracle_calls for r in results]),
                "leaves_median": med([r.stats.leaves_evaluated for r in results]),
                "best_probability": float(r0.best_probability),
                "best_lambda": "".join(map(str, r0.best_lambda)),
                "true_lambda": "".join(map(str, cases[case_name]["true_lambda"])),
                "p7_pair_contractions_median": med(pair_exec),
                "p7_cache_hits_median": med(hits),
                "p7_cache_misses_median": med(misses),
                "p7_bytes_reused_median": med(reused),
                "p7_cache_peak_bytes_median": med(peak),
                "call_counter_example": json.dumps(calls_example or {}, sort_keys=True),
                "p7_stats_example": json.dumps(stats_example or {}, sort_keys=True),
            }
            rows.append(row)

        try:
            decoder.clear_fast_network_cache()
        except Exception:
            pass
        del decoder
        cleanup_gpu()

    # Correctness comparison against the ordinary P0-P5/P6-off baseline.
    checks = []
    if "baseline" in modes:
        for row in rows:
            if row["mode"] == "baseline":
                continue
            ref = reference[row["case"]]
            lam_equal = row["best_lambda"] == "".join(map(str, ref.best_lambda))
            pdiff = abs(float(row["best_probability"]) - float(ref.best_probability))
            checks.append({
                "mode": row["mode"],
                "case": row["case"],
                "lambda_equal": bool(lam_equal),
                "probability_abs_diff": float(pdiff),
                "pass": bool(lam_equal and pdiff <= 1e-10),
            })

    print("\n===== SUMMARY =====")
    baseline_rows = {r["case"]: r for r in rows if r["mode"] == "baseline"}
    for row in rows:
        speed = float("nan")
        if row["case"] in baseline_rows:
            speed = baseline_rows[row["case"]]["wall_median_s"] / row["wall_median_s"]
        print(
            f"{row['mode']:12s} {row['case']:4s}: "
            f"median={row['wall_median_s']:.6f}s "
            f"speedup={speed:.4f}x "
            f"p7_hits={row['p7_cache_hits_median']:.1f} "
            f"reused_MB={row['p7_bytes_reused_median']/1024**2:.1f}"
        )
    print("consistency checks =", checks)

    csv_path = out_dir / "p7_easy_hard_summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    payload = {
        "version": "v0.7",
        "optimization": "P7 contraction-tree shared-environment reuse",
        "baseline": "v0.6 with parent_batch_size=1",
        "bb72": {"n": 72, "k": 12, "block_size": 9, "logical_order": list(range(12))},
        "p": float(args.p),
        "profile": str(profile),
        "settings": {
            "memory_limit": args.memory_limit,
            "guard_rtol": float(args.guard_rtol),
            "constant_cache": bool(args.constant_cache),
            "gpu_resident_operands": bool(args.gpu_resident_operands),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "p7_cache_budget": args.p7_cache_budget,
            "p7_max_entry": args.p7_max_entry,
            "p7_max_dependency_bits": int(args.p7_max_dependency_bits),
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
    json_path = out_dir / "p7_easy_hard_results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print("wrote", csv_path)
    print("wrote", json_path)


if __name__ == "__main__":
    main()
