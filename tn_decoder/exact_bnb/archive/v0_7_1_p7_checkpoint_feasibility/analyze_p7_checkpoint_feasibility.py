from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import opt_einsum as oe

from multi_logical_tensor_network_decoder import MultiLogicalTensorNetworkDecoder


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


def load_profile(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict) or "paths" not in obj:
        raise ValueError("unexpected compiled profile format")
    return obj


def term_nbytes(term: str, size_dict: dict[str, int], itemsize: int = 8) -> int:
    n = int(itemsize)
    for c in term:
        n *= int(size_dict[c])
    return n


def total_live_bytes(state, size_dict, itemsize=8):
    return sum(term_nbytes(x["term"], size_dict, itemsize) for x in state)


def max_live_bytes(state, size_dict, itemsize=8):
    if not state:
        return 0
    return max(term_nbytes(x["term"], size_dict, itemsize) for x in state)


def branch_sliced_term(term: str, prefix_chars: set[str]) -> str:
    # Fixing an output logical bit selects one coordinate and removes that axis.
    return "".join(c for c in term if c not in prefix_chars)


def snapshot_metrics(
    *,
    step: int,
    state,
    size_dict_unsliced,
    size_dict_per_slice,
    prefix_chars: set[str],
    tail_projector_ids: set[int],
    touched_tail_ids: set[int],
    completed_flops_per_slice: int,
    total_flops_per_slice: int,
    num_slices: int,
):
    # IMPORTANT:
    # A standalone [1,1] tail projector is still fully recoverable/openable.
    # Its lineage being present in the live state does NOT mean it has been
    # consumed.  A tail bit becomes irreversibly marginalized only after that
    # projector participates in an actual contraction step.
    untouched = set(tail_projector_ids) - set(touched_tail_ids)

    branch_terms = [
        branch_sliced_term(x["term"], prefix_chars)
        for x in state
    ]
    branch_state = [
        {"term": t, "tail_lineage": x["tail_lineage"]}
        for t, x in zip(branch_terms, state)
    ]

    per_slice_live = total_live_bytes(state, size_dict_per_slice)
    per_slice_branch = total_live_bytes(branch_state, size_dict_per_slice)
    unsliced_live = total_live_bytes(state, size_dict_unsliced)
    unsliced_branch = total_live_bytes(branch_state, size_dict_unsliced)

    return {
        "step_before_next_contraction": int(step),
        "live_tensors": int(len(state)),
        "tail_projectors_untouched": int(len(untouched)),
        "tail_projector_ids_untouched": sorted(int(x) for x in untouched),
        "completed_flops_per_slice": int(completed_flops_per_slice),
        "completed_flop_fraction": (
            float(completed_flops_per_slice / total_flops_per_slice)
            if total_flops_per_slice else 0.0
        ),
        "per_slice_live_bytes": int(per_slice_live),
        "per_slice_max_tensor_bytes": int(max_live_bytes(state, size_dict_per_slice)),
        "retain_all_slices_live_bytes": int(per_slice_live * num_slices),
        "per_slice_branch_sliced_live_bytes": int(per_slice_branch),
        "per_slice_branch_sliced_max_tensor_bytes": int(
            max_live_bytes(branch_state, size_dict_per_slice)
        ),
        "retain_all_slices_branch_sliced_bytes": int(per_slice_branch * num_slices),
        "unsliced_live_bytes": int(unsliced_live),
        "unsliced_branch_sliced_live_bytes": int(unsliced_branch),
    }


def human_bytes(n: int) -> str:
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or u == "TiB":
            return f"{x:.3f} {u}"
        x /= 1024.0
    return f"{x:.3f} TiB"


def analyze_root_path(decoder, profile_obj, *, open_prefix=tuple(range(9))):
    open_prefix = tuple(int(x) for x in open_prefix)
    fixed, open_list, marginalized = decoder._validate_logical_request(
        {}, list(open_prefix)
    )
    tn, output_inds, num_base = decoder._prepare_projector_mass_network(
        fixed, open_list, marginalized
    )

    equation = tn.get_equation(output_inds=output_inds)
    lhs, rhs = equation.split("->")
    input_terms = lhs.split(",")
    output_term = rhs
    shapes = [tuple(int(d) for d in a.shape) for a in tn.arrays]

    if len(input_terms) != len(shapes):
        raise RuntimeError("equation operand count != tensor array count")

    # Prefix output symbols are ordered exactly as output_inds/open_prefix.
    if len(output_term) != len(open_prefix):
        raise RuntimeError(
            f"expected {len(open_prefix)} prefix output symbols, got {len(output_term)}"
        )
    prefix_char_by_logical = {
        int(logical): output_term[pos]
        for pos, logical in enumerate(open_prefix)
    }
    prefix_chars = set(prefix_char_by_logical.values())

    # _prepare_projector_mass_network appends one boundary tensor for every
    # non-open logical in logical-index order.  For b9 these are 9,10,11.
    non_open = [i for i in range(decoder.num_logicals) if i not in set(open_prefix)]
    boundary_positions = {
        logical: num_base + j for j, logical in enumerate(non_open)
    }
    if max(boundary_positions.values()) >= len(input_terms):
        raise RuntimeError("boundary operand positions are inconsistent")

    tail_char_by_logical = {}
    for logical, pos in boundary_positions.items():
        term = input_terms[pos]
        if len(term) != 1:
            raise RuntimeError(
                f"expected rank-1 boundary operand for logical {logical}, got term {term!r}"
            )
        tail_char_by_logical[int(logical)] = term

    # Dimension dictionary.
    size_dict = {}
    for term, shape in zip(input_terms, shapes):
        if len(term) != len(shape):
            raise RuntimeError(f"term/shape rank mismatch: {term}, {shape}")
        for c, d in zip(term, shape):
            d = int(d)
            if c in size_dict and size_dict[c] != d:
                raise RuntimeError(f"inconsistent dimension for einsum symbol {c!r}")
            size_dict[c] = d

    paths = profile_obj["paths"]
    row = paths.get(tuple(open_prefix))
    if row is None:
        # Be robust to stringified keys in a hand-edited pickle-like profile.
        for k, v in paths.items():
            if tuple(k) == tuple(open_prefix):
                row = v
                break
    if row is None:
        raise KeyError(f"profile has no topology {open_prefix}")

    path = [tuple(int(i) for i in step) for step in row["path"]]
    slices = list(row.get("slices", ()))
    num_slices = int(row.get("num_slices", 1))

    # cuTensorNet slicing: use dimension 1 for the sliced mode when estimating
    # the per-slice checkpoint memory.  Profile num_slices is authoritative.
    sliced_symbols = []
    for item in slices:
        if isinstance(item, (tuple, list)) and item:
            sliced_symbols.append(str(item[0]))
        else:
            sliced_symbols.append(str(item))

    size_dict_per_slice = dict(size_dict)
    slice_symbols_found = []
    slice_symbols_missing = []
    for c in sliced_symbols:
        if c in size_dict_per_slice:
            size_dict_per_slice[c] = 1
            slice_symbols_found.append(c)
        else:
            slice_symbols_missing.append(c)

    _, path_info = oe.contract_path(
        equation,
        *shapes,
        shapes=True,
        optimize=path,
    )
    contractions = list(path_info.contraction_list)

    if len(contractions) != len(path):
        raise RuntimeError("opt_einsum contraction list length != compiled path length")

    tail_projector_ids = {
        int(boundary_positions[i]) for i in non_open
    }

    state = []
    for operand_id, term in enumerate(input_terms):
        state.append({
            "term": term,
            "tail_lineage": (
                {operand_id} if operand_id in tail_projector_ids else set()
            ),
        })

    # First compute a per-step FLOP estimate under the sliced dimensions.
    step_flops = []
    shadow = [{"term": x["term"]} for x in state]
    for contraction in contractions:
        inds, idx_removed, einsum_str, _remaining, _blas = contraction
        selected = [shadow[int(i)] for i in inds]
        involved = set()
        for x in selected:
            involved.update(x["term"])
        inner = bool(idx_removed)
        flops = int(
            oe.helpers.flop_count(
                involved,
                inner=inner,
                num_terms=max(1, len(selected)),
                size_dictionary=size_dict_per_slice,
            )
        )
        step_flops.append(flops)
        for i in inds:
            shadow.pop(int(i))
        result_term = einsum_str.split("->", 1)[1]
        shadow.append({"term": result_term})

    total_flops_per_slice = int(sum(step_flops))

    snapshots = []
    completed = 0
    first_touch_step = None
    first_touch_logicals = []
    first_touch_ids = []
    touched_tail_ids = set()

    # Snapshot before step 0: all three [1,1] tail projectors are still
    # standalone operands, so all three tail logical bits are recoverable.
    snapshots.append(
        snapshot_metrics(
            step=0,
            state=state,
            size_dict_unsliced=size_dict,
            size_dict_per_slice=size_dict_per_slice,
            prefix_chars=prefix_chars,
            tail_projector_ids=tail_projector_ids,
            touched_tail_ids=touched_tail_ids,
            completed_flops_per_slice=0,
            total_flops_per_slice=total_flops_per_slice,
            num_slices=num_slices,
        )
    )

    for step_id, contraction in enumerate(contractions):
        inds, _idx_removed, einsum_str, _remaining, _blas = contraction
        selected = [state[int(i)] for i in inds]

        selected_tail = set()
        for x in selected:
            selected_tail |= set(x["tail_lineage"])

        newly_touched = selected_tail - touched_tail_ids
        if newly_touched and first_touch_step is None:
            first_touch_step = int(step_id)
            first_touch_ids = sorted(int(x) for x in newly_touched)
            inv = {v: k for k, v in boundary_positions.items()}
            first_touch_logicals = sorted(int(inv[x]) for x in newly_touched)

        touched_tail_ids |= selected_tail

        lineage = set()
        for x in selected:
            lineage |= set(x["tail_lineage"])

        for i in inds:
            state.pop(int(i))
        result_term = einsum_str.split("->", 1)[1]
        state.append({"term": result_term, "tail_lineage": lineage})

        completed += int(step_flops[step_id])
        snapshots.append(
            snapshot_metrics(
                step=step_id + 1,
                state=state,
                size_dict_unsliced=size_dict,
                size_dict_per_slice=size_dict_per_slice,
                prefix_chars=prefix_chars,
                tail_projector_ids=tail_projector_ids,
                touched_tail_ids=touched_tail_ids,
                completed_flops_per_slice=completed,
                total_flops_per_slice=total_flops_per_slice,
                num_slices=num_slices,
            )
        )

    # "Safe" means no tail-[1,1] projector has participated in any contraction,
    # so all three tail bits can still be lifted to open legs without lost info.
    safe = [
        s for s in snapshots
        if s["tail_projectors_untouched"] == len(tail_projector_ids)
    ]
    if not safe:
        raise RuntimeError(
            "Internal bookkeeping error: no safe checkpoint was found, not even "
            "the pre-contraction state. Check tail-projector touch tracking."
        )
    latest_safe = safe[-1]

    # Identify a few low-memory safe checkpoints and Pareto-like choices.
    safe_sorted_by_mem = sorted(
        safe,
        key=lambda s: (
            s["retain_all_slices_live_bytes"],
            -s["completed_flop_fraction"],
        ),
    )
    best_memory_safe = safe_sorted_by_mem[0]

    # Crude structural support statistics.
    LZ = np.asarray(decoder.logical_obs, dtype=np.uint8)
    prefix_rows = np.asarray(open_prefix, dtype=int)
    tail_rows = np.asarray(non_open, dtype=int)
    prefix_union = np.flatnonzero(np.any(LZ[prefix_rows] != 0, axis=0))
    tail_union = np.flatnonzero(np.any(LZ[tail_rows] != 0, axis=0))
    overlap = np.intersect1d(prefix_union, tail_union)

    result = {
        "topology": {
            "open_prefix": list(open_prefix),
            "marginalized_tail": list(non_open),
            "num_base_operands": int(num_base),
            "num_total_operands": int(len(input_terms)),
            "path_steps": int(len(path)),
            "profile_num_slices": int(num_slices),
            "profile_sliced_symbols": sliced_symbols,
            "slice_symbols_found_in_equation": slice_symbols_found,
            "slice_symbols_missing_from_equation": slice_symbols_missing,
            "profile_opt_cost": float(row.get("opt_cost", float("nan"))),
            "profile_largest_intermediate": float(
                row.get("largest_intermediate", float("nan"))
            ),
        },
        "logical_symbols": {
            "prefix_char_by_logical": {
                str(k): v for k, v in prefix_char_by_logical.items()
            },
            "tail_boundary_char_by_logical": {
                str(k): v for k, v in tail_char_by_logical.items()
            },
            "tail_boundary_operand_position": {
                str(k): int(v) for k, v in boundary_positions.items()
            },
        },
        "first_tail_projector_touch": {
            "step_zero_based": first_touch_step,
            "logical_bits": first_touch_logicals,
            "operand_ids": first_touch_ids,
        },
        "latest_safe_checkpoint": latest_safe,
        "minimum_memory_safe_checkpoint": best_memory_safe,
        "total_flops_per_slice_estimate": int(total_flops_per_slice),
        "total_flops_all_slices_estimate": int(total_flops_per_slice * num_slices),
        "logical_support": {
            "prefix_row_weights": [int(LZ[i].sum()) for i in open_prefix],
            "tail_row_weights": [int(LZ[i].sum()) for i in non_open],
            "prefix_union_physical_bits": int(len(prefix_union)),
            "tail_union_physical_bits": int(len(tail_union)),
            "prefix_tail_union_overlap_physical_bits": int(len(overlap)),
        },
        "all_safe_checkpoints": safe,
    }
    return result


def main():
    ap = argparse.ArgumentParser(
        description=(
            "P7 feasibility: inspect the validated b9 root contraction tree and "
            "find the latest checkpoint before any [1,1] tail marginalizer is consumed."
        )
    )
    ap.add_argument(
        "--profile",
        default="profiles/canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl",
    )
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--output-dir", default="p7_checkpoint_feasibility")
    args = ap.parse_args()

    profile_path = Path(args.profile).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)

    H, LZ, LX = build_bb72()
    profile = load_profile(profile_path)

    decoder = MultiLogicalTensorNetworkDecoder(
        H=H,
        logical_obs=LZ,
        noise_model=[float(args.p)] * int(H.shape[1]),
        device="cuda",
    )
    decoder.load_fast_path_profile(profile_path)

    result = analyze_root_path(decoder, profile, open_prefix=tuple(range(9)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "p7_checkpoint_feasibility.json"
    json_path.write_text(json.dumps(result, indent=2))

    safe_csv = out / "p7_safe_checkpoints.csv"
    safe_rows = result["all_safe_checkpoints"]
    with safe_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(safe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(safe_rows)

    latest = result["latest_safe_checkpoint"]
    minimum = result["minimum_memory_safe_checkpoint"]

    print("===== P7 CHECKPOINT FEASIBILITY =====")
    print("profile =", profile_path)
    print(
        "b9 root: open logicals 0..8; tail 9..11 currently marginalized by [1,1]^T"
    )
    print("path steps =", result["topology"]["path_steps"])
    print("num slices =", result["topology"]["profile_num_slices"])
    print(
        "first tail-[1,1] touch = step",
        result["first_tail_projector_touch"]["step_zero_based"],
        "logical bits",
        result["first_tail_projector_touch"]["logical_bits"],
    )
    print()
    print("LATEST SAFE CHECKPOINT (all three tail bits still recoverable)")
    print("  completed step count =", latest["step_before_next_contraction"])
    print(
        "  completed FLOP fraction =",
        f"{100.0 * latest['completed_flop_fraction']:.3f}%",
    )
    print("  live tensors =", latest["live_tensors"])
    print(
        "  checkpoint memory per slice =",
        human_bytes(latest["per_slice_live_bytes"]),
    )
    print(
        "  checkpoint memory if all slices retained =",
        human_bytes(latest["retain_all_slices_live_bytes"]),
    )
    print(
        "  after fixing one 9-bit prefix, residual live memory per slice =",
        human_bytes(latest["per_slice_branch_sliced_live_bytes"]),
    )
    print(
        "  branch-sliced residual if all slices retained =",
        human_bytes(latest["retain_all_slices_branch_sliced_bytes"]),
    )
    print(
        "  largest branch-sliced tensor per slice =",
        human_bytes(latest["per_slice_branch_sliced_max_tensor_bytes"]),
    )
    print()
    print("MINIMUM-MEMORY SAFE CHECKPOINT")
    print("  completed step count =", minimum["step_before_next_contraction"])
    print(
        "  completed FLOP fraction =",
        f"{100.0 * minimum['completed_flop_fraction']:.3f}%",
    )
    print(
        "  all-slices memory =",
        human_bytes(minimum["retain_all_slices_live_bytes"]),
    )
    print()
    print("LOGICAL SUPPORT")
    print(
        "  tail row weights =",
        result["logical_support"]["tail_row_weights"],
    )
    print(
        "  physical bits touched by tail logicals =",
        result["logical_support"]["tail_union_physical_bits"],
    )
    print(
        "  overlap with prefix-logical support =",
        result["logical_support"]["prefix_tail_union_overlap_physical_bits"],
    )
    print()
    print("wrote", json_path)
    print("wrote", safe_csv)


if __name__ == "__main__":
    main()
