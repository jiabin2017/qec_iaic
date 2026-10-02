from __future__ import annotations

import argparse
import json
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
from sympy.abc import x, y
from qldpc import codes
from qldpc.objects import Pauli

from exact_block_bnb import ExactBlockBnB
from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder
from p5_utils import block_topologies, parse_logical_order


def sync_cuda():
    # v0.3 keeps v0.2 P3 CuPy-resident operands, so synchronize CuPy explicitly.
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


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        try:
            x = x.get()
        except Exception:
            pass
    return np.asarray(x, dtype=np.float64)


def cache_hits_by_open(decoder) -> dict[tuple[int, ...], int]:
    return {
        tuple(int(i) for i in row["open_logicals"]): int(row["hits"])
        for row in decoder.fast_cache_info()
    }


def cache_hit_delta(before: dict[tuple[int, ...], int], after: dict[tuple[int, ...], int]):
    keys = sorted(set(before) | set(after))
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def find_weight3_logical_collision(H: np.ndarray, LZ: np.ndarray):
    """Find same-syndrome weight-3 errors in different logical sectors.

    BB72 has distance 6, so a pair of weight-3 errors whose difference is a
    weight-6 non-trivial logical operator is a natural hard-decoding candidate.
    The two microscopic errors also have exactly the same independent-noise
    prior because their Hamming weights are equal.
    """
    n = H.shape[1]
    seen: dict[bytes, tuple[bytes, tuple[int, int, int], np.ndarray]] = {}

    for pos in itertools.combinations(range(n), 3):
        cols = np.fromiter(pos, dtype=np.int64)
        syndrome = np.bitwise_xor.reduce(H[:, cols], axis=1).astype(np.uint8)
        logical = np.bitwise_xor.reduce(LZ[:, cols], axis=1).astype(np.uint8)
        skey = np.packbits(syndrome).tobytes()
        lkey = np.packbits(logical).tobytes()

        old = seen.get(skey)
        if old is None:
            seen[skey] = (lkey, pos, logical.copy())
            continue

        old_lkey, old_pos, old_logical = old
        if old_lkey != lkey:
            e1 = np.zeros(n, dtype=np.uint8)
            e2 = np.zeros(n, dtype=np.uint8)
            e1[list(old_pos)] = 1
            e2[list(pos)] = 1
            # Defensive verification.
            assert np.array_equal((H @ e1) % 2, (H @ e2) % 2)
            assert not np.array_equal((LZ @ e1) % 2, (LZ @ e2) % 2)
            return e1, e2, syndrome, old_logical, logical

    raise RuntimeError(
        "No same-syndrome weight-3 collision across logical sectors was found."
    )


def run_bnb_once(
    *,
    decoder,
    syndrome: np.ndarray,
    k: int,
    memory_limit: str,
    autotune: int,
    guard_rtol: float,
    print_tree: bool,
    label: str,
    gpu_resident_operands: bool,
    constant_cache: bool,
    retain_scratch: bool,
    scratch_retain_budget: str | int | None,
    block_size: int,
    logical_order: tuple[int, ...],
):
    syndrome_list = syndrome.astype(float).tolist()

    def oracle(fixed_logicals, open_logicals):
        sync_cuda()
        out = decoder.contract_logical_mass_fast(
            syndrome=syndrome_list,
            fixed_logicals=dict(fixed_logicals),
            open_logicals=list(open_logicals),
            autotune_iterations=autotune,
            memory_limit=memory_limit,
            # v0.3/P4: keep SCRATCH resident when requested. If P4 is
            # disabled this recovers the v0.2 policy (release SCRATCH, keep
            # constant-intermediate CACHE).
            release_workspace=not retain_scratch,
            use_constant_qualifiers=constant_cache,
            use_gpu_resident_operands=gpu_resident_operands,
            retain_constant_cache=constant_cache,
            retain_scratch_workspace=retain_scratch,
            scratch_retention_budget=scratch_retain_budget,
        )
        sync_cuda()
        return out

    before = cache_hits_by_open(decoder)
    searcher = ExactBlockBnB(
        k=k,
        block_size=block_size,
        oracle=oracle,
        logical_order=list(logical_order),
        guard_rtol=guard_rtol,
        print_tree=print_tree,
    )

    print(f"\n--- {label} ---")
    result = searcher.search()

    if print_tree:
        print("\nsearch tree")
        for line in result.tree_lines:
            print(line)

    after = cache_hits_by_open(decoder)
    print("ML lambda =", list(result.best_lambda))
    print("ML normalized probability =", result.best_probability)
    print("oracle calls =", result.stats.oracle_calls)
    print("expanded internal nodes =", result.stats.expanded_internal_nodes)
    print("leaves evaluated =", result.stats.leaves_evaluated)
    print("pruned complete sectors =", result.stats.pruned_leaves)
    print("clipped negative entries =", result.stats.clipped_negative_entries)
    print("most negative raw mass =", result.stats.most_negative_raw_mass)
    if result.root_mass:
        print(
            "most negative posterior-scale mass =",
            result.stats.most_negative_raw_mass / result.root_mass,
        )
        print(
            "max local pruning guard (posterior scale) =",
            result.stats.max_local_guard_abs / result.root_mass,
        )
    print("oracle time =", result.stats.oracle_seconds, "s")
    print("BnB wall time =", result.stats.wall_seconds, "s")
    print("cache-hit delta =", cache_hit_delta(before, after))
    print("fast cache info =", decoder.fast_cache_info())
    print("P3 GPU operand info =", decoder.fast_gpu_operand_info())
    print("P4 workspace info =", decoder.fast_workspace_info())
    fallback = decoder.last_fast_fallback()
    if fallback is not None:
        print("fast fallback =", fallback)
    return result


def full_reference(
    decoder, syndrome, k, memory_limit, gpu_resident_operands, constant_cache
):
    syndrome_list = syndrome.astype(float).tolist()
    sync_cuda()
    t0 = time.perf_counter()
    full = decoder.contract_logical_mass_fast(
        syndrome=syndrome_list,
        fixed_logicals={},
        open_logicals=list(range(k)),
        memory_limit=memory_limit,
        # Reference is not part of online BnB. Keep P3 device operands, but
        # do not retain a large P2 CACHE workspace that could interfere with
        # the four decoder topologies used by the next syndrome.
        release_workspace=True,
        use_constant_qualifiers=False,
        use_gpu_resident_operands=gpu_resident_operands,
        retain_constant_cache=False,
    )
    sync_cuda()
    # Include the small device->host result transfer in reference latency so
    # v0.3 reference timing remains comparable to the earlier NumPy-output runs.
    full = to_numpy(full)
    dt = time.perf_counter() - t0

    idx = tuple(int(v) for v in np.unravel_index(np.argmax(full), full.shape))
    mass = float(full[idx])
    total = float(full.sum())
    return full, idx, mass, total, mass / total, dt


def run_case(
    *,
    name: str,
    decoder,
    error: np.ndarray,
    syndrome: np.ndarray,
    true_lambda: np.ndarray,
    k: int,
    args,
    competitor_error: np.ndarray | None = None,
    competitor_lambda: np.ndarray | None = None,
):
    # v0.1 / P1 deliberately does NOT clear the fast topology cache here.
    # Case A starts cold because the decoder is new. Case B therefore tests
    # whether the same prepared P5 topologies survive a syndrome change.
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print("physical error positions =", np.where(error)[0].tolist())
    print("physical error weight =", int(error.sum()))
    print("syndrome weight =", int(syndrome.sum()))
    print("chosen-error lambda =", true_lambda.tolist())
    if competitor_error is not None:
        print("equal-weight competitor positions =", np.where(competitor_error)[0].tolist())
        print("equal-weight competitor lambda =", competitor_lambda.tolist())
        print("same syndrome =", bool(np.array_equal((decoder.parity_check_matrix @ error) % 2,
                                                    (decoder.parity_check_matrix @ competitor_error) % 2)))
        print("NOTE: chosen physical-error sector need not be the ML sector; the full-sector masses decide.")

    is_easy = name.startswith("CASE A")
    first_label = (
        "BnB RUN 1 (first use; compiled path if profile is loaded)"
        if is_easy
        else "BnB RUN 1 (new syndrome; should reuse prepared P5 topologies)"
    )
    r1 = run_bnb_once(
        decoder=decoder,
        syndrome=syndrome,
        k=k,
        memory_limit=args.memory_limit,
        autotune=args.autotune,
        guard_rtol=args.guard_rtol,
        print_tree=(not args.no_tree),
        label=first_label,
        gpu_resident_operands=args.gpu_resident_operands,
        constant_cache=args.constant_cache,
        retain_scratch=args.retain_scratch,
        scratch_retain_budget=args.scratch_retain_budget,
        block_size=args.block_size,
        logical_order=args.logical_order_resolved,
    )

    r2 = run_bnb_once(
        decoder=decoder,
        syndrome=syndrome,
        k=k,
        memory_limit=args.memory_limit,
        autotune=args.autotune,
        guard_rtol=args.guard_rtol,
        print_tree=args.tree_second,
        label="BnB RUN 2 (same syndrome, should reuse prepared topologies)",
        gpu_resident_operands=args.gpu_resident_operands,
        constant_cache=args.constant_cache,
        retain_scratch=args.retain_scratch,
        scratch_retain_budget=args.scratch_retain_budget,
        block_size=args.block_size,
        logical_order=args.logical_order_resolved,
    )

    same_runs = tuple(r1.best_lambda) == tuple(r2.best_lambda)
    print("\nRUN1 lambda == RUN2 lambda =", same_runs)
    if not same_runs:
        raise AssertionError("Repeated BnB runs disagreed on ML sector")

    return r1, r2


def validate_full_reference_case(
    *,
    label: str,
    decoder,
    syndrome: np.ndarray,
    k: int,
    bnb_result,
    args,
):
    """Run the full-4096 reference after P4 performance measurements finish."""
    print("\n--- FULL 4096 REFERENCE:", label, "---")
    full, idx, mass, total, prob, dt = full_reference(
        decoder,
        syndrome,
        k,
        args.memory_limit,
        args.gpu_resident_operands,
        args.constant_cache,
    )
    print("full time =", dt, "s")
    print("full ML lambda =", list(idx))
    print("full ML normalized probability =", prob)

    bnb_idx = tuple(bnb_result.best_lambda)
    bnb_mass_in_full = float(full[bnb_idx])
    bnb_prob_in_full = bnb_mass_in_full / total
    full_gap_abs = mass - bnb_mass_in_full
    full_gap_prob = full_gap_abs / max(abs(total), 1.0)

    print("BnB lambda == full4096 argmax index =", bnb_idx == idx)
    print("BnB sector probability evaluated in SAME full4096 tensor =", bnb_prob_in_full)
    print("full argmax - BnB sector posterior gap =", full_gap_prob)
    print(
        "cross-contraction BnB/full-ML mass difference (posterior scale) =",
        abs(bnb_result.best_mass - mass) / max(abs(total), 1.0),
    )

    flat = full.reshape(-1)
    topn = min(8, flat.size)
    order = np.argsort(flat)[::-1][:topn]
    print("top sectors from the same full4096 tensor:")
    for rank, flat_i in enumerate(order, 1):
        lam = tuple(int(v) for v in np.unravel_index(int(flat_i), full.shape))
        p = float(flat[int(flat_i)] / total)
        gap = float((mass - flat[int(flat_i)]) / max(abs(total), 1.0))
        marker = "  <-- BnB" if lam == bnb_idx else ""
        print(f"  #{rank}: {''.join(map(str, lam))}  P={p:.15g}  gap={gap:.3e}{marker}")

    numerical_tol = max(
        float(args.guard_rtol),
        float(bnb_result.stats.max_local_guard_abs / max(abs(bnb_result.root_mass), 1.0)),
    )
    print("numerical ML-equivalence tolerance =", numerical_tol)

    if bnb_idx == idx:
        print("VALIDATION: PASS (same argmax index)")
    elif full_gap_prob <= numerical_tol:
        print(
            "VALIDATION: PASS (numerically ML-equivalent near-degenerate sector; "
            "argmax index is not resolved at current float64 precision)"
        )
    else:
        raise AssertionError(
            f"BnB sector is below full4096 argmax by {full_gap_prob}, "
            f"larger than numerical tolerance {numerical_tol}. "
            f"BnB={bnb_idx}, full={idx}"
        )



def main():
    ap = argparse.ArgumentParser(
        description="BB72 v0.4/P5 exact BnB: configurable block size + logical order"
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Optional p5_best_config.json from scan_bb72_p5.py; overrides block/order/profile.",
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--block-size", type=int, default=3, help="P5 logical block size.")
    ap.add_argument(
        "--logical-order",
        default="identity",
        help=(
            "P5 logical order: identity, reverse, weight-asc, weight-desc, "
            "overlap-max, overlap-min, random, or an explicit comma-separated permutation."
        ),
    )
    ap.add_argument("--order-seed", type=int, default=9107)
    ap.add_argument(
        "--no-gpu-resident-operands",
        dest="gpu_resident_operands",
        action="store_false",
        help="Disable v0.2/P3 and fall back to v0.1 operand reset behavior.",
    )
    ap.add_argument(
        "--no-constant-cache",
        dest="constant_cache",
        action="store_false",
        help="Disable v0.2/P2 constant-intermediate caching.",
    )
    ap.add_argument(
        "--no-retain-scratch",
        dest="retain_scratch",
        action="store_false",
        help="Disable v0.3/P4 and recover v0.2 scratch-release behavior.",
    )
    ap.add_argument(
        "--scratch-retain-budget",
        default="70%",
        help=(
            "Global P4 budget for simultaneously retained SCRATCH workspaces. "
            "Examples: 70%%, 28GB, 24GiB. LRU scratch buffers are evicted "
            "without discarding prepared Networks or compiled paths."
        ),
    )
    ap.set_defaults(
        gpu_resident_operands=True,
        constant_cache=True,
        retain_scratch=True,
    )
    ap.add_argument(
        "--path-profile",
        default=None,
        help=(
            "P5 offline-compiled cuTensorNet path profile from "
            "tune_bb72_p5_paths.py. If omitted or missing, first use of each "
            "topology falls back to runtime path search."
        ),
    )
    ap.add_argument("--autotune", type=int, default=0)
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--tree-second", action="store_true")
    ap.add_argument("--no-full-reference", dest="full_reference", action="store_false")
    ap.set_defaults(full_reference=True)
    args = ap.parse_args()
    if args.config:
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text())
        args.block_size = int(cfg["block_size"])
        args.logical_order = ",".join(str(int(v)) for v in cfg["logical_order"])
        if cfg.get("profile_path"):
            prof = Path(str(cfg["profile_path"]))
            if not prof.is_absolute() and not prof.exists():
                candidate = cfg_path.parent / prof.name
                if candidate.exists():
                    prof = candidate
            args.path_profile = str(prof)
        print("loaded P5 config =", str(cfg_path))
    if args.constant_cache and not args.gpu_resident_operands:
        print(
            "NOTE: --no-gpu-resident-operands also disables P2 constant cache "
            "because v0.2 P2 requires stable GPU operand pointers."
        )
        args.constant_cache = False

    code = codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    )
    H = np.asarray(code.matrix_z, dtype=np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    LX = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)

    n = H.shape[1]
    k = LZ.shape[0]
    if args.block_size < 1 or args.block_size > k:
        raise ValueError(f"--block-size must be in [1,{k}]")
    args.logical_order_resolved = parse_logical_order(
        args.logical_order,
        k=k,
        logical_obs=LZ,
        block_size=args.block_size,
        seed=args.order_seed,
    )
    args.bnb_topologies = block_topologies(args.logical_order_resolved, args.block_size)
    if not np.array_equal((LX @ LZ.T) % 2, np.eye(k, dtype=np.uint8)):
        raise RuntimeError("LX @ LZ.T != I over GF(2)")

    # Easy case: the deterministic weight-1 case used throughout the prior tests.
    rng = np.random.default_rng(args.seed)
    easy_error = (rng.random(n) < args.p).astype(np.uint8)
    if easy_error.sum() == 0:
        easy_error[0] = 1
    easy_syndrome = ((H @ easy_error) % 2).astype(float)
    easy_lambda = ((LZ @ easy_error) % 2).astype(np.uint8)

    print("Searching deterministically for a harder weight-3 syndrome candidate...")
    hard_error, hard_competitor, hard_syn_u8, hard_lambda, competitor_lambda = \
        find_weight3_logical_collision(H, LZ)
    hard_syndrome = hard_syn_u8.astype(float)

    print("BB72 v0.4/P5 repeated exact TN-BnB")
    print("n =", n, "k =", k, "p =", args.p)
    print("memory limit =", args.memory_limit)
    print("guard rtol =", args.guard_rtol)
    print("v0.4 P5 block size =", args.block_size)
    print("v0.4 P5 logical order =", list(args.logical_order_resolved))
    print("v0.4 P5 BnB topologies =", list(args.bnb_topologies))
    print("v0.4 P3 gpu-resident operands =", args.gpu_resident_operands)
    print("v0.4 P2 constant-intermediate cache =", args.constant_cache)
    print("v0.4 P4 retain scratch =", args.retain_scratch)
    print("v0.4 P4 scratch retention budget =", args.scratch_retain_budget)
    print("Hard candidate construction: two weight-3 errors share a syndrome but have different logical sectors.")

    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[args.p] * n,
        device="cuda",
    )
    # P5 may use anywhere from 1 to k BnB topologies. Keep all of them plus
    # the full-reference topology resident at the Network-object level.
    decoder.set_fast_cache_max_entries(max(8, len(args.bnb_topologies) + 2))

    profile_path = Path(args.path_profile) if args.path_profile else None
    if profile_path is not None and profile_path.exists():
        loaded = decoder.load_fast_path_profile(profile_path)
        print("loaded path profile =", loaded["filename"])
        print("compiled topologies =", loaded["open_logicals"])
        profile_meta = loaded.get("metadata", {})
        profile_mem = profile_meta.get("memory_limit")
        profile_b = profile_meta.get("block_size")
        profile_order = profile_meta.get("logical_order")
        if profile_b is not None and int(profile_b) != int(args.block_size):
            print("WARNING: path profile block_size =", profile_b, "but run block_size =", args.block_size)
        if profile_order is not None and tuple(int(v) for v in profile_order) != tuple(args.logical_order_resolved):
            print("WARNING: path profile logical_order =", profile_order, "but run logical_order =", list(args.logical_order_resolved))
        if profile_mem is not None and str(profile_mem) != str(args.memory_limit):
            print(
                "WARNING: path profile memory_limit =", profile_mem,
                "but run memory_limit =", args.memory_limit,
            )
    elif profile_path is not None:
        print(
            "path profile not found:", str(profile_path),
            "-- first use of each topology will run the native path optimizer",
        )

    easy_r1, easy_r2 = run_case(
        name="CASE A: EASY SYNDROME",
        decoder=decoder,
        error=easy_error,
        syndrome=easy_syndrome,
        true_lambda=easy_lambda,
        k=k,
        args=args,
    )

    hard_r1, hard_r2 = run_case(
        name="CASE B: HARDER SAME-SYNDROME / DIFFERENT-LOGICAL WEIGHT-3 COLLISION",
        decoder=decoder,
        error=hard_error,
        syndrome=hard_syndrome,
        true_lambda=hard_lambda,
        competitor_error=hard_competitor,
        competitor_lambda=competitor_lambda,
        k=k,
        args=args,
    )

    # P4 performance measurements are complete before full-reference work.
    # A full 12-open contraction wants ~20 GB scratch with the current profile;
    # combining that with retained P5 workspaces may exceed the GPU
    # A100. Release only BnB SCRATCH now, after the online timings are frozen.
    if args.full_reference:
        print("\nP4 workspace state before validation =", decoder.fast_workspace_info())
        released = decoder.release_fast_scratch_workspaces()
        print("released retained BnB scratch workspaces before full references =", released)
        print("P4 workspace state after release =", decoder.fast_workspace_info())
        validate_full_reference_case(
            label="CASE A EASY",
            decoder=decoder,
            syndrome=easy_syndrome,
            k=k,
            bnb_result=easy_r2,
            args=args,
        )
        validate_full_reference_case(
            label="CASE B HARD",
            decoder=decoder,
            syndrome=hard_syndrome,
            k=k,
            bnb_result=hard_r2,
            args=args,
        )

    print("\nFINAL fast cache info =", decoder.fast_cache_info())
    print("FINAL P3 GPU operand info =", decoder.fast_gpu_operand_info())
    print("FINAL P4 workspace info =", decoder.fast_workspace_info())
    decoder.clear_fast_network_cache()


if __name__ == "__main__":
    main()
