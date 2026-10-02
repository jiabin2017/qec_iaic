from __future__ import annotations

import argparse
import gc
import json
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

from exact_block_bnb import ExactBlockBnB
from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder
from p5_utils import block_topologies
from validation_utils import (
    DEFAULT_P_VALUES,
    DEFAULT_STRESS_WEIGHTS,
    completed_ids,
    ensure_datasets,
    human_seconds,
    open_results_db,
    parse_float_list,
    parse_int_list,
    resolve_layouts,
    write_summaries,
)


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


def make_decoder(*, H, LZ, p: float, profile_path: Path, num_topologies: int, memory_limit: str):
    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[float(p)] * int(H.shape[1]),
        device="cuda",
    )
    decoder.set_fast_cache_max_entries(max(8, int(num_topologies) + 2))
    loaded = decoder.load_fast_path_profile(profile_path)
    meta = loaded.get("metadata", {})
    profile_mem = meta.get("memory_limit")
    if profile_mem is not None and str(profile_mem) != str(memory_limit):
        print(f"WARNING: profile memory_limit={profile_mem}, run memory_limit={memory_limit}")
    return decoder


def decode_once(
    *, decoder, syndrome: np.ndarray, k: int, block_size: int, logical_order: tuple[int, ...],
    memory_limit: str, guard_rtol: float, gpu_resident_operands: bool,
    constant_cache: bool, retain_scratch: bool, scratch_retain_budget: str,
):
    syndrome_list = syndrome.astype(float).tolist()
    call_counts: Counter[tuple[int, ...]] = Counter()

    def oracle(fixed_logicals, open_logicals):
        key = tuple(int(i) for i in open_logicals)
        call_counts[key] += 1
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
        k=int(k),
        block_size=int(block_size),
        oracle=oracle,
        logical_order=list(logical_order),
        guard_rtol=float(guard_rtol),
        print_tree=False,
    )
    result = searcher.search()
    return result, call_counts


def insert_layout_meta(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO layout_meta(layout_tag, block_size, logical_order, profile_path,
                                scan_easy_seconds, scan_hard_seconds, scan_mean_seconds)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(layout_tag) DO UPDATE SET
            block_size=excluded.block_size,
            logical_order=excluded.logical_order,
            profile_path=excluded.profile_path,
            scan_easy_seconds=excluded.scan_easy_seconds,
            scan_hard_seconds=excluded.scan_hard_seconds,
            scan_mean_seconds=excluded.scan_mean_seconds
        """,
        (
            str(row["tag"]), int(row["block_size"]), json.dumps(row["logical_order"]),
            str(row["profile_path_resolved"]), float(row.get("easy_median_seconds", float("nan"))),
            float(row.get("hard_median_seconds", float("nan"))),
            float(row.get("mean_easy_hard_seconds", float("nan"))),
        ),
    )
    conn.commit()


def choose_pending_group(specs, conn, layout_tag: str, p: float):
    # Natural p group first, then stress if it uses the same p. This lets p=0.005
    # natural + stress share one prepared decoder instance.
    candidates = [s for s in specs if abs(float(s.p) - float(p)) < 1e-15]
    candidates.sort(key=lambda s: (0 if s.kind == "natural" else 1, s.name))
    for spec in candidates:
        done = completed_ids(conn, layout_tag, spec.name)
        if len(done) < spec.shots:
            return spec, done
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BB72 exact-BnB P5 multi-syndrome validation, single GPU, resumable SQLite checkpoints"
    )
    ap.add_argument("--scan-dir", default="p5_1_scan", help="Directory containing P5.1 profiles/results.")
    ap.add_argument("--output-dir", default="p5_multisyndrome_validation")
    ap.add_argument("--layout-selection", default="validation6", choices=["validation6", "all15", "stageA", "stageB"])
    ap.add_argument("--layout-tags", default=None, help="Optional comma-separated exact scan tags; overrides --layout-selection.")
    ap.add_argument("--p-values", default="0.005,0.01,0.02,0.03,0.04,0.05")
    ap.add_argument("--natural-shots", type=int, default=5000)
    ap.add_argument("--stress-shots", type=int, default=5000)
    ap.add_argument("--stress-weights", default="2,3,4,5")
    ap.add_argument("--stress-p", type=float, default=0.005)
    ap.add_argument("--dataset-seed", type=int, default=20260919)
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--summary-every", type=int, default=250)
    ap.add_argument("--max-wall-hours", type=float, default=0.0, help="0 = no self-stop; otherwise stop cleanly after this many hours.")
    ap.add_argument("--retain-scratch", action="store_true", help="Enable P4 scratch retention (off by default; BB72 showed negligible gain).")
    ap.add_argument("--scratch-retain-budget", default="70%")
    ap.add_argument("--no-constant-cache", dest="constant_cache", action="store_false")
    ap.add_argument("--no-gpu-resident-operands", dest="gpu_resident_operands", action="store_false")
    ap.set_defaults(constant_cache=True, gpu_resident_operands=True)
    args = ap.parse_args()

    if args.constant_cache and not args.gpu_resident_operands:
        print("NOTE: disabling P2 constant cache because P3 GPU-resident operands are disabled")
        args.constant_cache = False

    t_job0 = time.monotonic()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scan_dir = Path(args.scan_dir).resolve()

    H, LZ, _LX = build_bb72()
    n, k = int(H.shape[1]), int(LZ.shape[0])
    p_values = parse_float_list(args.p_values)
    stress_weights = parse_int_list(args.stress_weights)

    specs, dataset_manifest = ensure_datasets(
        output_dir=outdir,
        H=H,
        LZ=LZ,
        p_values=p_values,
        natural_shots=args.natural_shots,
        stress_shots=args.stress_shots,
        stress_weights=stress_weights,
        stress_p=args.stress_p,
        seed=args.dataset_seed,
    )
    layouts = resolve_layouts(scan_dir, args.layout_selection, args.layout_tags)
    db_path = outdir / "results.sqlite"
    conn = open_results_db(db_path)
    for layout in layouts:
        insert_layout_meta(conn, layout)

    run_manifest = {
        "format_version": 1,
        "code": "BB72 [[72,12,6]]",
        "channel": "X-error decoding using H_Z and logical-Z observables",
        "single_gpu": True,
        "scan_dir": str(scan_dir),
        "layout_selection": args.layout_selection,
        "layout_tags": [r["tag"] for r in layouts],
        "datasets": dataset_manifest,
        "memory_limit": args.memory_limit,
        "guard_rtol": args.guard_rtol,
        "constant_cache": args.constant_cache,
        "gpu_resident_operands": args.gpu_resident_operands,
        "retain_scratch": args.retain_scratch,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    print("BB72 multi-syndrome validation")
    print("n =", n, "k =", k)
    print("layouts =", len(layouts), [r["tag"] for r in layouts])
    print("datasets =", [(s.name, s.shots) for s in specs])
    print("database =", db_path)
    print("resume behavior: existing (layout,dataset,shot_id) rows are skipped")
    if args.max_wall_hours > 0:
        print("clean self-stop after", args.max_wall_hours, "hours")

    total_new = 0
    for li, layout in enumerate(layouts, 1):
        tag = str(layout["tag"])
        b = int(layout["block_size"])
        order = tuple(int(x) for x in layout["logical_order"])
        topologies = block_topologies(order, b)
        profile = Path(layout["profile_path_resolved"])
        print("\n" + "=" * 88)
        print(f"LAYOUT {li}/{len(layouts)}: {tag}")
        print("block_size =", b, "order =", list(order))
        print("topologies =", list(topologies))
        print("profile =", profile)

        # Process each distinct noise p once. Natural p=stress_p and the stress
        # set share the same decoder instance.
        p_schedule = []
        for spec in specs:
            if spec.p not in p_schedule:
                p_schedule.append(spec.p)

        for p in p_schedule:
            pending_spec, _ = choose_pending_group(specs, conn, tag, p)
            if pending_spec is None:
                print(f"  p={p:g}: all datasets already complete; skip decoder creation")
                continue

            print(f"  preparing decoder for p={p:g}")
            decoder = make_decoder(
                H=H, LZ=LZ, p=float(p), profile_path=profile,
                num_topologies=len(topologies), memory_limit=args.memory_limit,
            )

            # Warm once using the first pending shot at this p. This prepares
            # Networks/operands/path state but is deliberately not inserted into DB.
            first_spec, first_done = choose_pending_group(specs, conn, tag, p)
            with np.load(first_spec.path) as data:
                first_id = next(i for i in range(first_spec.shots) if i not in first_done)
                first_syn = np.asarray(data["syndromes"][first_id], dtype=np.uint8)
                # Prefer a different real syndrome for warm-up so the first timed
                # shot still pays the normal in-place syndrome update cost.
                warm_syn = first_syn
                for j in range(first_id + 1, first_spec.shots):
                    if j in first_done:
                        continue
                    cand = np.asarray(data["syndromes"][j], dtype=np.uint8)
                    if not np.array_equal(cand, first_syn):
                        warm_syn = cand
                        break
            sync_cuda()
            t_warm = time.perf_counter()
            _warm_result, _ = decode_once(
                decoder=decoder, syndrome=warm_syn, k=k, block_size=b, logical_order=order,
                memory_limit=args.memory_limit, guard_rtol=args.guard_rtol,
                gpu_resident_operands=args.gpu_resident_operands, constant_cache=args.constant_cache,
                retain_scratch=args.retain_scratch, scratch_retain_budget=args.scratch_retain_budget,
            )
            sync_cuda()
            print(f"  warm-up = {time.perf_counter()-t_warm:.3f}s (not recorded)")

            group_specs = [s for s in specs if abs(float(s.p) - float(p)) < 1e-15]
            group_specs.sort(key=lambda s: (0 if s.kind == "natural" else 1, s.name))
            for spec in group_specs:
                done = completed_ids(conn, tag, spec.name)
                if len(done) >= spec.shots:
                    print(f"  {spec.name}: complete {len(done)}/{spec.shots}; skip")
                    continue

                print(f"  {spec.name}: resume at {len(done)}/{spec.shots}")
                group_t0 = time.monotonic()
                recent_times: list[float] = []
                completed_count = len(done)
                with np.load(spec.path) as data:
                    syndromes = data["syndromes"]
                    true_lambdas = data["true_lambdas"]
                    error_weights = data["error_weights"]
                    syndrome_weights = data["syndrome_weights"]

                    for shot_id in range(spec.shots):
                        if shot_id in done:
                            continue
                        if args.max_wall_hours > 0 and (time.monotonic() - t_job0) >= args.max_wall_hours * 3600.0:
                            print("\nReached --max-wall-hours; writing summaries and stopping cleanly.")
                            write_summaries(conn, outdir)
                            conn.close()
                            decoder.clear_fast_network_cache()
                            cleanup_gpu()
                            return

                        syn = np.asarray(syndromes[shot_id], dtype=np.uint8)
                        true_lam = tuple(int(x) for x in np.asarray(true_lambdas[shot_id], dtype=np.uint8))
                        try:
                            result, call_counts = decode_once(
                                decoder=decoder, syndrome=syn, k=k, block_size=b, logical_order=order,
                                memory_limit=args.memory_limit, guard_rtol=args.guard_rtol,
                                gpu_resident_operands=args.gpu_resident_operands,
                                constant_cache=args.constant_cache, retain_scratch=args.retain_scratch,
                                scratch_retain_budget=args.scratch_retain_budget,
                            )
                        except Exception as exc:
                            err_path = outdir / "FAILED_SHOT.txt"
                            err_path.write_text(
                                f"layout={tag}\ndataset={spec.name}\nshot_id={shot_id}\np={p}\nerror={type(exc).__name__}: {exc}\n"
                            )
                            write_summaries(conn, outdir)
                            raise

                        best_lam = tuple(int(x) for x in result.best_lambda)
                        success = int(best_lam == true_lam)
                        call_counts_json = json.dumps(
                            {"-".join(map(str, kk)): int(vv) for kk, vv in sorted(call_counts.items())},
                            sort_keys=True,
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO shots(
                                layout_tag,dataset_name,shot_id,p,error_weight,syndrome_weight,
                                wall_seconds,oracle_seconds,oracle_calls,expanded_internal_nodes,
                                leaves_evaluated,pruned_leaves,best_probability,best_lambda,true_lambda,
                                logical_success,root_mass,clipped_negative_entries,most_negative_raw_mass,
                                max_local_guard_abs,topology_call_counts
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                tag, spec.name, int(shot_id), float(p), int(error_weights[shot_id]),
                                int(syndrome_weights[shot_id]), float(result.stats.wall_seconds),
                                float(result.stats.oracle_seconds), int(result.stats.oracle_calls),
                                int(result.stats.expanded_internal_nodes), int(result.stats.leaves_evaluated),
                                int(result.stats.pruned_leaves), float(result.best_probability),
                                "".join(map(str, best_lam)), "".join(map(str, true_lam)), success,
                                float(result.root_mass), int(result.stats.clipped_negative_entries),
                                float(result.stats.most_negative_raw_mass), float(result.stats.max_local_guard_abs),
                                call_counts_json,
                            ),
                        )
                        conn.commit()  # one-shot transaction: robust to scheduler termination
                        total_new += 1
                        completed_count += 1
                        recent_times.append(float(result.stats.wall_seconds))
                        if len(recent_times) > 200:
                            recent_times.pop(0)

                        completed_now = completed_count
                        if completed_now % max(1, args.progress_every) == 0 or completed_now == spec.shots:
                            mean_recent = float(np.mean(recent_times)) if recent_times else float("nan")
                            remaining = spec.shots - completed_now
                            eta = remaining * mean_recent
                            elapsed = time.monotonic() - group_t0
                            print(
                                f"    {completed_now:5d}/{spec.shots}  "
                                f"last={result.stats.wall_seconds:.4f}s  recent_mean={mean_recent:.4f}s  "
                                f"calls={result.stats.oracle_calls:3d} leaves={result.stats.leaves_evaluated:3d}  "
                                f"group_elapsed={human_seconds(elapsed)} ETA~{human_seconds(eta)}",
                                flush=True,
                            )
                        if total_new % max(1, args.summary_every) == 0:
                            write_summaries(conn, outdir)

            decoder.clear_fast_network_cache()
            del decoder
            cleanup_gpu()

    write_summaries(conn, outdir)
    conn.close()
    print("\nALL REQUESTED LAYOUTS/DATASETS COMPLETE")
    print("results =", db_path)
    print("summary =", outdir / "summary.csv")
    print("consistency =", outdir / "consistency_vs_b12.csv")


if __name__ == "__main__":
    main()
