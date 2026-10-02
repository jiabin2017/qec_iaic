from __future__ import annotations

import argparse
import gc
import pickle
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from sympy.abc import x, y
from qldpc import codes
from qldpc.objects import Pauli

from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder
from p5_utils import block_topologies, parse_logical_order, order_label


def sync_cuda() -> None:
    """Synchronize the active CUDA device for meaningful wall-clock timing."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return
    except Exception:
        pass
    try:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass



def gpu_memory_info() -> dict[str, int | None]:
    """Best-effort CUDA/CuPy memory diagnostics for long scan jobs."""
    info: dict[str, int | None] = {
        "free_bytes": None,
        "total_bytes": None,
        "cupy_pool_used_bytes": None,
        "cupy_pool_total_bytes": None,
    }
    try:
        import cupy as cp
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        info["free_bytes"] = int(free_b)
        info["total_bytes"] = int(total_b)
        pool = cp.get_default_memory_pool()
        info["cupy_pool_used_bytes"] = int(pool.used_bytes())
        info["cupy_pool_total_bytes"] = int(pool.total_bytes())
    except Exception:
        pass
    return info


def aggressive_gpu_cleanup(*, verbose: bool = False) -> None:
    """Release allocator caches between independent offline path trials."""
    try:
        sync_cuda()
    except Exception:
        pass
    gc.collect()
    try:
        import cupy as cp
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    if verbose:
        print("GPU cleanup:", gpu_memory_info())


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name())
    except Exception:
        pass
    return None


def cuquantum_version() -> str | None:
    try:
        import cuquantum
        return str(getattr(cuquantum, "__version__", "unknown"))
    except Exception:
        return None


def build_bb72_decoder(p: float) -> tuple[MultiLogicalTensorNetworkDecoder, np.ndarray, np.ndarray]:
    code = codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    )
    H = np.asarray(code.matrix_z, dtype=np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[p] * H.shape[1],
        device="cuda",
    )
    decoder.set_fast_cache_max_entries(8)
    return decoder, H, LZ


def build_network_spec(
    decoder: MultiLogicalTensorNetworkDecoder,
    syndrome: np.ndarray,
    open_logicals: tuple[int, ...],
) -> tuple[str, list[Any], tuple[str, ...]]:
    syndrome_list = syndrome.astype(float).tolist()
    fixed, open_list, marginalized = decoder._validate_logical_request({}, open_logicals)
    decoder.flip_syndromes(syndrome_list)
    tn, output_inds, _ = decoder._prepare_projector_mass_network(
        fixed, open_list, marginalized
    )
    return tn.get_equation(output_inds=output_inds), list(tn.arrays), output_inds


def make_network(equation: str, operands: list[Any], decoder, memory_limit: str):
    from cuquantum.tensornet import Network

    options: dict[str, Any] = {"device_id": decoder.contractor_config.device_id}
    if memory_limit:
        options["memory_limit"] = memory_limit
    return Network(equation, *operands, options=options)


def candidate_optimizer(samples: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": int(seed)}
    if int(samples) > 0:
        out["samples"] = int(samples)
    return out


def search_one_candidate(
    *,
    equation: str,
    operands: list[Any],
    decoder,
    memory_limit: str,
    samples: int,
    seed: int,
    benchmark_repeats: int,
) -> dict[str, Any]:
    network = make_network(equation, operands, decoder, memory_limit)
    try:
        t0 = time.perf_counter()
        path, info = network.contract_path(
            optimize=candidate_optimizer(samples=samples, seed=seed)
        )
        path_search_seconds = time.perf_counter() - t0

        times: list[float] = []
        # Benchmark the exact plan returned by the optimizer.  The first call is
        # retained as a real measurement because v0.1 itself also uses
        # release_workspace=True, so each oracle call pays workspace allocation.
        for _ in range(max(1, int(benchmark_repeats))):
            sync_cuda()
            t1 = time.perf_counter()
            _ = network.contract(release_workspace=True)
            sync_cuda()
            times.append(time.perf_counter() - t1)

        return {
            "path": [tuple(int(v) for v in pair) for pair in path],
            "slices": [tuple(item) for item in getattr(info, "slices", ())],
            "opt_cost": float(getattr(info, "opt_cost", float("nan"))),
            "largest_intermediate": float(
                getattr(info, "largest_intermediate", float("nan"))
            ),
            "num_slices": int(getattr(info, "num_slices", 1)),
            "seed": int(seed),
            "samples": int(samples),
            "path_search_seconds": float(path_search_seconds),
            "benchmark_times": [float(v) for v in times],
            "benchmark_median_seconds": float(statistics.median(times)),
        }
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


def tune_topology(
    *,
    decoder,
    syndrome: np.ndarray,
    open_logicals: tuple[int, ...],
    memory_limit: str,
    trials: int,
    samples: int,
    benchmark_repeats: int,
    seed0: int,
) -> dict[str, Any]:
    equation, operands, _ = build_network_spec(decoder, syndrome, open_logicals)

    candidates: list[dict[str, Any]] = []
    print("\n" + "=" * 72)
    print("TUNING open_logicals =", open_logicals)
    print("=" * 72)

    for trial in range(int(trials)):
        seed = int(seed0) + trial
        # Start every offline optimizer trial from a clean allocator state.
        aggressive_gpu_cleanup()
        try:
            row = search_one_candidate(
                equation=equation,
                operands=operands,
                decoder=decoder,
                memory_limit=memory_limit,
                samples=samples,
                seed=seed,
                benchmark_repeats=benchmark_repeats,
            )
        except Exception as exc:
            print(f"trial {trial + 1}/{trials} seed={seed} FAILED: {exc}")
            if "alloc" in str(exc).lower() or "memory" in str(exc).lower():
                aggressive_gpu_cleanup(verbose=True)
            continue

        candidates.append(row)
        print(
            f"trial {trial + 1}/{trials} seed={seed} "
            f"median={row['benchmark_median_seconds']:.6f}s "
            f"opt_cost={row['opt_cost']:.6g} "
            f"largest={row['largest_intermediate']:.6g} "
            f"slices={row['num_slices']} "
            f"path_search={row['path_search_seconds']:.3f}s"
        )

    if not candidates:
        raise RuntimeError(f"all path-search trials failed for {open_logicals}")

    # Primary selection criterion is measured runtime on this GPU.  FLOP count
    # breaks ties and provides a deterministic secondary criterion.
    candidates.sort(
        key=lambda row: (
            row["benchmark_median_seconds"],
            row["opt_cost"],
            row["num_slices"],
        )
    )
    best = dict(candidates[0])
    best["num_candidates"] = len(candidates)
    best["candidate_summary"] = [
        {
            "seed": r["seed"],
            "benchmark_median_seconds": r["benchmark_median_seconds"],
            "opt_cost": r["opt_cost"],
            "largest_intermediate": r["largest_intermediate"],
            "num_slices": r["num_slices"],
            "path_search_seconds": r["path_search_seconds"],
        }
        for r in candidates
    ]
    print(
        "BEST",
        open_logicals,
        "median=", best["benchmark_median_seconds"],
        "s",
        "opt_cost=", best["opt_cost"],
        "slices=", best["num_slices"],
        "seed=", best["seed"],
    )
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "v0.4/P5 offline cuTensorNet path compiler for arbitrary BB72 "
            "block size and logical ordering.  The profile is reusable across "
            "syndromes because only operand values change, not topology."
        )
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument(
        "--logical-order",
        default="identity",
        help=(
            "identity, reverse, weight-asc, weight-desc, overlap-max, "
            "overlap-min, random, or an explicit comma-separated permutation."
        ),
    )
    ap.add_argument("--order-seed", type=int, default=9107)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--benchmark-repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7300)
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--no-full-reference-topology",
        action="store_true",
        help="Do not compile the 12-open full-reference topology.",
    )
    args = ap.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark-repeats must be >= 1")

    decoder, H, LZ = build_bb72_decoder(args.p)
    n = H.shape[1]
    k = LZ.shape[0]
    if args.block_size < 1 or args.block_size > k:
        raise ValueError(f"--block-size must be in [1,{k}]")

    logical_order = parse_logical_order(
        args.logical_order,
        k=k,
        logical_obs=LZ,
        block_size=args.block_size,
        seed=args.order_seed,
    )

    # Path optimization depends on network equation/shapes, not the syndrome
    # values. Use the familiar deterministic weight-1 syndrome.
    error = np.zeros(n, dtype=np.uint8)
    error[28] = 1
    syndrome = ((H @ error) % 2).astype(float)

    topologies = list(block_topologies(logical_order, args.block_size))
    if not args.no_full_reference_topology:
        full_topology = tuple(range(k))
        if full_topology not in topologies:
            topologies.append(full_topology)

    if args.output is None:
        args.output = f"bb72_p5_b{args.block_size}_order_{order_label(logical_order)}.pkl"

    print("BB72 v0.4/P5 offline path compiler")
    print("n =", n, "k =", k, "p =", args.p)
    print("block size =", args.block_size)
    print("logical order =", list(logical_order))
    print("BnB topologies =", list(block_topologies(logical_order, args.block_size)))
    print("GPU =", gpu_name())
    print("cuQuantum =", cuquantum_version())
    print("memory limit =", args.memory_limit)
    print("trials/topology =", args.trials)
    print("samples/trial =", args.samples)
    print("benchmark repeats =", args.benchmark_repeats)

    entries: dict[tuple[int, ...], dict[str, Any]] = {}
    total_start = time.perf_counter()
    for topo_index, open_logicals in enumerate(topologies):
        entries[open_logicals] = tune_topology(
            decoder=decoder,
            syndrome=syndrome,
            open_logicals=open_logicals,
            memory_limit=args.memory_limit,
            trials=args.trials,
            samples=args.samples,
            benchmark_repeats=args.benchmark_repeats,
            seed0=args.seed + 1000 * topo_index,
        )

    payload = {
        "format_version": 2,
        "metadata": {
            "code": "BB72",
            "n": int(n),
            "k": int(k),
            "optimization": "v0.4/P5 block-size + logical-order topology tuning",
            "block_size": int(args.block_size),
            "logical_order": list(logical_order),
            "bnb_topologies": [list(t) for t in block_topologies(logical_order, args.block_size)],
            "p_used_for_benchmark": float(args.p),
            "memory_limit": str(args.memory_limit),
            "trials": int(args.trials),
            "samples": int(args.samples),
            "benchmark_repeats": int(args.benchmark_repeats),
            "gpu_name": gpu_name(),
            "cuquantum_version": cuquantum_version(),
            "selection": "minimum measured median contraction time",
            "compile_wall_seconds": float(time.perf_counter() - total_start),
        },
        "paths": entries,
    }

    output = Path(args.output)
    with output.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 72)
    print("WROTE", output)
    print("=" * 72)
    for key, row in entries.items():
        print(
            key,
            "median=", row["benchmark_median_seconds"],
            "s",
            "opt_cost=", row["opt_cost"],
            "slices=", row["num_slices"],
            "seed=", row["seed"],
        )

    decoder.clear_fast_network_cache()


if __name__ == "__main__":
    main()
