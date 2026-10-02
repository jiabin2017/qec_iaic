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


HARD_DEGENERATE_SECTORS = {
    "001110011011",
    "111011000000",
    "100110011001",
}

EASY_REFERENCE_P = 0.9999999955333834
HARD_REFERENCE_P = 0.3333164874732356


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

    eye = np.eye(LZ.shape[0], dtype=np.uint8)
    if not np.array_equal((LX @ LZ.T) % 2, eye):
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
    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[float(p)] * int(H.shape[1]),
        device="cuda",
    )
    # Only root b9 and tail b3 topologies are needed, but leave a little room.
    decoder.set_fast_cache_max_entries(8)
    loaded = decoder.load_fast_path_profile(profile)

    meta = loaded.get("metadata", {})
    profile_mem = meta.get("memory_limit")
    if profile_mem is not None and str(profile_mem) != str(memory_limit):
        print(
            f"WARNING: profile memory_limit={profile_mem}, "
            f"run memory_limit={memory_limit}"
        )
    return decoder, loaded


def decode_once(
    *,
    decoder,
    syndrome: np.ndarray,
    memory_limit: str,
    guard_rtol: float,
    constant_cache: bool,
    gpu_resident_operands: bool,
    retain_scratch: bool,
    scratch_retain_budget: str,
):
    syndrome_list = syndrome.astype(float).tolist()
    calls: Counter[tuple[int, ...]] = Counter()

    def oracle(fixed_logicals, open_logicals):
        key = tuple(int(i) for i in open_logicals)
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

    searcher = ExactBlockBnB(
        k=12,
        block_size=9,
        oracle=oracle,
        logical_order=list(range(12)),
        guard_rtol=float(guard_rtol),
        print_tree=False,
    )
    result = searcher.search()
    return result, calls


def median(xs):
    return float(statistics.median(float(x) for x in xs))


def check_reference(case_name: str, result, *, atol: float = 1e-10) -> dict:
    lam = "".join(map(str, result.best_lambda))
    if case_name == "easy":
        lam_ok = lam == "000000000000"
        p_ref = EASY_REFERENCE_P
    elif case_name == "hard":
        # This syndrome has a true three-fold ML degeneracy under homogeneous
        # iid X noise; any of these sectors is an exact ML answer.
        lam_ok = lam in HARD_DEGENERATE_SECTORS
        p_ref = HARD_REFERENCE_P
    else:
        raise ValueError(case_name)

    p_diff = abs(float(result.best_probability) - float(p_ref))
    return {
        "lambda": lam,
        "lambda_ok": bool(lam_ok),
        "probability": float(result.best_probability),
        "reference_probability": float(p_ref),
        "probability_abs_diff": float(p_diff),
        "pass": bool(lam_ok and p_diff <= atol),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frozen BB72 P0-P5 production smoke benchmark (b9 identity)."
    )
    ap.add_argument(
        "--profile",
        default="profiles/canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl",
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output-dir", default="production_smoke_results")

    # Current production defaults:
    # P2 on, P3 on, P4 scratch retention off.
    ap.add_argument(
        "--no-constant-cache",
        dest="constant_cache",
        action="store_false",
    )
    ap.add_argument(
        "--no-gpu-resident-operands",
        dest="gpu_resident_operands",
        action="store_false",
    )
    ap.add_argument("--retain-scratch", action="store_true")
    ap.add_argument("--scratch-retain-budget", default="70%")
    ap.set_defaults(constant_cache=True, gpu_resident_operands=True)
    args = ap.parse_args()

    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats>=1 and warmup>=0 are required")

    profile = Path(args.profile).resolve()
    if not profile.exists():
        raise FileNotFoundError(profile)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, LZ, LX = build_bb72()
    cases = {
        "easy": make_case(H, LZ, [28]),
        "hard": make_case(H, LZ, [0, 27, 30]),
    }

    print("===== BB72 v0.5 FINAL PRODUCTION =====")
    print("P0-P5 validated production baseline")
    print("layout = [0..8] | [9..11], block_size = 9")
    print("profile =", profile)
    print("p =", args.p)
    print("P2 constant cache =", args.constant_cache)
    print("P3 GPU resident operands =", args.gpu_resident_operands)
    print("P4 scratch retention =", args.retain_scratch)
    print("warmup =", args.warmup, "repeats =", args.repeats)

    cleanup_gpu()
    decoder, profile_info = make_decoder(
        H, LZ, p=args.p, profile=profile, memory_limit=args.memory_limit
    )
    print("profile topologies =", profile_info.get("open_logicals"))

    for _ in range(args.warmup):
        for case_name in ("easy", "hard"):
            decode_once(
                decoder=decoder,
                syndrome=cases[case_name]["syndrome"],
                memory_limit=args.memory_limit,
                guard_rtol=args.guard_rtol,
                constant_cache=args.constant_cache,
                gpu_resident_operands=args.gpu_resident_operands,
                retain_scratch=args.retain_scratch,
                scratch_retain_budget=args.scratch_retain_budget,
            )

    rows = []
    checks = []

    for case_name in ("easy", "hard"):
        times = []
        results = []
        calls_list = []
        for rep in range(args.repeats):
            result, calls = decode_once(
                decoder=decoder,
                syndrome=cases[case_name]["syndrome"],
                memory_limit=args.memory_limit,
                guard_rtol=args.guard_rtol,
                constant_cache=args.constant_cache,
                gpu_resident_operands=args.gpu_resident_operands,
                retain_scratch=args.retain_scratch,
                scratch_retain_budget=args.scratch_retain_budget,
            )
            results.append(result)
            calls_list.append(calls)
            times.append(result.stats.wall_seconds)
            print(
                f"{case_name:4s} rep={rep+1}: "
                f"wall={result.stats.wall_seconds:.6f}s "
                f"oracles={result.stats.oracle_calls} "
                f"leaves={result.stats.leaves_evaluated} "
                f"P={result.best_probability:.15g} "
                f"lambda={''.join(map(str, result.best_lambda))}"
            )

        r0 = results[0]
        check = check_reference(case_name, r0)
        check["case"] = case_name
        checks.append(check)

        row = {
            "case": case_name,
            "wall_median_s": median(times),
            "wall_min_s": float(min(times)),
            "wall_max_s": float(max(times)),
            "oracle_calls_median": median([r.stats.oracle_calls for r in results]),
            "leaves_median": median([r.stats.leaves_evaluated for r in results]),
            "pruned_leaves_median": median([r.stats.pruned_leaves for r in results]),
            "best_probability": float(r0.best_probability),
            "best_lambda": "".join(map(str, r0.best_lambda)),
            "true_lambda": "".join(map(str, cases[case_name]["true_lambda"])),
            "call_counter_example": json.dumps(
                {"-".join(map(str, k)): int(v) for k, v in calls_list[0].items()},
                sort_keys=True,
            ),
        }
        rows.append(row)

    print("\n===== SUMMARY =====")
    for row in rows:
        print(
            f"{row['case']:4s}: median={row['wall_median_s']:.6f}s "
            f"oracles={row['oracle_calls_median']:.1f} "
            f"leaves={row['leaves_median']:.1f} "
            f"P={row['best_probability']:.15g} "
            f"lambda={row['best_lambda']}"
        )
    print("reference checks =", checks)

    csv_path = out_dir / "production_easy_hard_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "version": "v0.5_final_production",
        "scope": "P0-P5 validated production baseline",
        "bb72": {
            "n": 72,
            "k": 12,
            "block_size": 9,
            "logical_order": list(range(12)),
        },
        "profile": str(profile),
        "p": float(args.p),
        "settings": {
            "memory_limit": args.memory_limit,
            "guard_rtol": float(args.guard_rtol),
            "constant_cache": bool(args.constant_cache),
            "gpu_resident_operands": bool(args.gpu_resident_operands),
            "retain_scratch": bool(args.retain_scratch),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "rows": rows,
        "reference_checks": checks,
    }
    json_path = out_dir / "production_easy_hard_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print("wrote", csv_path)
    print("wrote", json_path)

    if any(not x["pass"] for x in checks):
        raise SystemExit(
            "Production reference check failed. Inspect output before using this build."
        )


if __name__ == "__main__":
    main()
