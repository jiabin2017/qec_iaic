from __future__ import annotations

import argparse
import itertools
import time
from typing import Any

import numpy as np
from sympy.abc import x, y
from qldpc import codes
from qldpc.objects import Pauli

from exact_block_bnb import ExactBlockBnB
from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder


def sync_cuda():
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
            release_workspace=True,
            use_constant_qualifiers=False,
        )
        sync_cuda()
        return out

    before = cache_hits_by_open(decoder)
    searcher = ExactBlockBnB(
        k=k,
        block_size=3,
        oracle=oracle,
        logical_order=list(range(k)),
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
    return result


def full_reference(decoder, syndrome, k, memory_limit):
    syndrome_list = syndrome.astype(float).tolist()
    sync_cuda()
    t0 = time.perf_counter()
    full = decoder.contract_logical_mass_fast(
        syndrome=syndrome_list,
        fixed_logicals={},
        open_logicals=list(range(k)),
        memory_limit=memory_limit,
        release_workspace=True,
        use_constant_qualifiers=False,
    )
    sync_cuda()
    dt = time.perf_counter() - t0

    full = to_numpy(full)
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
    # Make each syndrome's first BnB run genuinely cold.  The second run is
    # deliberately left warm to test path/Network reuse for the same syndrome.
    decoder.clear_fast_network_cache()

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

    r1 = run_bnb_once(
        decoder=decoder,
        syndrome=syndrome,
        k=k,
        memory_limit=args.memory_limit,
        autotune=args.autotune,
        guard_rtol=args.guard_rtol,
        print_tree=(not args.no_tree),
        label="BnB RUN 1 (cold topology cache)",
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
    )

    same_runs = tuple(r1.best_lambda) == tuple(r2.best_lambda)
    print("\nRUN1 lambda == RUN2 lambda =", same_runs)
    if not same_runs:
        raise AssertionError("Repeated BnB runs disagreed on ML sector")

    if args.full_reference:
        print("\n--- FULL 4096 REFERENCE ---")
        full, idx, mass, total, prob, dt = full_reference(
            decoder, syndrome, k, args.memory_limit
        )
        print("full time =", dt, "s")
        print("full ML lambda =", list(idx))
        print("full ML normalized probability =", prob)

        bnb_idx = tuple(r2.best_lambda)
        bnb_mass_in_full = float(full[bnb_idx])
        bnb_prob_in_full = bnb_mass_in_full / total
        full_gap_abs = mass - bnb_mass_in_full
        full_gap_prob = full_gap_abs / max(abs(total), 1.0)

        print("BnB lambda == full4096 argmax index =", bnb_idx == idx)
        print("BnB sector probability evaluated in SAME full4096 tensor =", bnb_prob_in_full)
        print("full argmax - BnB sector posterior gap =", full_gap_prob)
        print(
            "cross-contraction BnB/full-ML mass difference (posterior scale) =",
            abs(r2.best_mass - mass) / max(abs(total), 1.0),
        )

        # Show the leading sectors using ONE common contraction.  This is the
        # meaningful way to diagnose near-degenerate/tied ML candidates.
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

        # Strict index equality is inappropriate when several sectors are
        # separated by less than the TN contraction's observed numerical noise.
        # Use the larger of the nominal guard and the largest adaptive local
        # guard observed by BnB, expressed on posterior scale.
        numerical_tol = max(
            float(args.guard_rtol),
            float(r2.stats.max_local_guard_abs / max(abs(r2.root_mass), 1.0)),
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

    decoder.clear_fast_network_cache()


def main():
    ap = argparse.ArgumentParser(
        description="BB72 exact b=3 BnB: easy + hard syndrome, each repeated twice"
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--autotune", type=int, default=0)
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--tree-second", action="store_true")
    ap.add_argument("--no-full-reference", dest="full_reference", action="store_false")
    ap.set_defaults(full_reference=True)
    args = ap.parse_args()

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

    print("BB72 repeated exact b=3 TN-BnB")
    print("n =", n, "k =", k, "p =", args.p)
    print("memory limit =", args.memory_limit)
    print("guard rtol =", args.guard_rtol)
    print("Hard candidate construction: two weight-3 errors share a syndrome but have different logical sectors.")

    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[args.p] * n,
        device="cuda",
    )
    # Four b=3 depth topologies plus the 12-open reference fit in this cache.
    decoder.set_fast_cache_max_entries(8)

    run_case(
        name="CASE A: EASY SYNDROME",
        decoder=decoder,
        error=easy_error,
        syndrome=easy_syndrome,
        true_lambda=easy_lambda,
        k=k,
        args=args,
    )

    run_case(
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


if __name__ == "__main__":
    main()
