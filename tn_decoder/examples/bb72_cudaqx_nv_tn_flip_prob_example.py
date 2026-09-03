from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TN_DECODER_ROOT = Path(__file__).resolve().parents[1]
if str(TN_DECODER_ROOT) not in sys.path:
    sys.path.insert(0, str(TN_DECODER_ROOT))
SCRIPT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

try:
    import cudaq_qec as qec
except ImportError as exc:  # pragma: no cover - local Windows host cannot import this package
    raise SystemExit(
        "Failed to import cudaq_qec. Run this script on the Linux machine where "
        "you installed cudaq-qec[tensor-network-decoder]."
    ) from exc

from common.bb72_baseline_common import build_problem


WEIGHT6_LOGICAL_X_SUPPORTS: tuple[tuple[tuple[str, int, int], ...], ...] = (
    (("L", 0, 0), ("L", 0, 3), ("L", 1, 0), ("L", 2, 0), ("L", 4, 3), ("L", 5, 0)),
    (("L", 0, 0), ("L", 0, 3), ("L", 1, 3), ("L", 2, 3), ("L", 4, 0), ("L", 5, 3)),
    (("L", 0, 0), ("L", 1, 0), ("L", 1, 3), ("L", 2, 0), ("L", 3, 0), ("L", 5, 3)),
    (("L", 0, 0), ("L", 1, 0), ("L", 3, 3), ("L", 4, 0), ("L", 5, 0), ("L", 5, 3)),
    (("L", 0, 0), ("L", 2, 2), ("L", 4, 4), ("R", 0, 2), ("R", 2, 4), ("R", 4, 0)),
    (("L", 0, 1), ("L", 0, 4), ("L", 1, 1), ("L", 2, 1), ("L", 4, 4), ("L", 5, 1)),
    (("L", 0, 1), ("L", 1, 1), ("L", 1, 4), ("L", 2, 1), ("L", 3, 1), ("L", 5, 4)),
    (("L", 0, 1), ("L", 2, 3), ("L", 4, 5), ("R", 0, 3), ("R", 2, 5), ("R", 4, 1)),
    (("L", 0, 2), ("L", 2, 4), ("L", 4, 0), ("R", 0, 4), ("R", 2, 0), ("R", 4, 2)),
    (("L", 0, 3), ("L", 2, 5), ("L", 4, 1), ("R", 0, 5), ("R", 2, 1), ("R", 4, 3)),
    (("R", 0, 0), ("R", 0, 1), ("R", 0, 2), ("R", 0, 3), ("R", 3, 1), ("R", 3, 5)),
    (("R", 0, 0), ("R", 0, 1), ("R", 0, 2), ("R", 0, 5), ("R", 3, 0), ("R", 3, 4)),
)

DEFAULT_LOGICAL_INDICES = "0,1,2,3"


def parse_index_list(text: str, length: int, name: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if value < 0 or value >= length:
            raise ValueError(f"{name} index out of range: {value} not in [0, {length - 1}]")
    return values


def parse_binary_vector(text: str, length: int, name: str) -> np.ndarray:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    vector = np.asarray(values, dtype=np.uint8)
    if not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must be binary")
    return vector


def build_logical_mask(
    num_logicals: int,
    logical_indices_text: str,
    logical_mask_text: str | None,
) -> np.ndarray:
    if logical_mask_text is not None:
        mask = parse_binary_vector(logical_mask_text, num_logicals, "logical-mask")
    else:
        mask = np.zeros(num_logicals, dtype=np.uint8)
        for index in parse_index_list(logical_indices_text, num_logicals, "logical-indices"):
            mask[index] ^= 1
    if not np.any(mask):
        raise ValueError("Selected logical mask is all zeros; choose at least one logical bit")
    return mask


def build_logical_observable(logical_basis: np.ndarray, logical_mask: np.ndarray) -> np.ndarray:
    observable = np.mod(logical_mask @ logical_basis, 2).astype(np.uint8)
    return observable.reshape(1, -1)


def logical_signature_in_basis(
    error: np.ndarray,
    logical_basis: np.ndarray,
) -> np.ndarray:
    return np.mod(logical_basis @ error, 2).astype(np.uint8)


def bb72_qubit_index(kind: str, a: int, b: int) -> int:
    base = 0 if kind == "L" else 36
    return base + 6 * a + b


def build_weight6_logical_x_basis() -> np.ndarray:
    basis = np.zeros((12, 72), dtype=np.uint8)
    for row, support in enumerate(WEIGHT6_LOGICAL_X_SUPPORTS):
        for kind, a, b in support:
            basis[row, bb72_qubit_index(kind, a, b)] = 1
    return basis


def resolve_logical_basis(problem: object, error_type: str, source: str) -> tuple[np.ndarray, str]:
    normalized_source = source
    if normalized_source == "auto":
        normalized_source = "weight6-x" if error_type == "Z" else "qldpc-dual"

    if normalized_source == "weight6-x":
        if error_type != "Z":
            raise ValueError(
                "weight6-x is aligned with Z-error decoding only, because it is "
                "an X-type logical basis built into this example."
            )
        return build_weight6_logical_x_basis(), normalized_source

    if normalized_source == "qldpc-dual":
        return np.asarray(problem.dual_logicals, dtype=np.uint8), normalized_source

    raise ValueError(f"Unsupported logical basis source: {source}")


def sample_error(
    num_qubits: int,
    physical_error_rate: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < physical_error_rate < 1.0:
        raise ValueError(f"p must satisfy 0 < p < 1, got {physical_error_rate}")
    rng = np.random.default_rng(seed)
    return (rng.random(num_qubits) < physical_error_rate).astype(np.uint8)


def build_syndrome(
    problem: object,
    mode: str,
    syndrome_bits: str | None,
    physical_error_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "zero":
        return np.zeros(problem.num_checks, dtype=np.uint8), None
    if mode == "manual":
        if syndrome_bits is None:
            raise ValueError("--syndrome-bits is required when --syndrome-mode=manual")
        syndrome = parse_binary_vector(syndrome_bits, problem.num_checks, "syndrome-bits")
        return syndrome, None
    if mode == "sample":
        error = sample_error(problem.n, physical_error_rate, seed)
        syndrome = problem.syndrome(error)
        return syndrome, error
    raise ValueError(f"Unsupported syndrome mode: {mode}")


def normalize_coset_masses(z0: float, z1: float) -> tuple[float, float]:
    total = z0 + z1
    if total <= 0.0:
        raise ValueError(f"Non-positive total coset mass: z0={z0}, z1={z1}")
    return z0 / total, z1 / total


def tn_raw_coset_masses(
    decoder: object,
    syndrome: np.ndarray,
    optimize_arg: str | None,
) -> tuple[float, float]:
    syndrome_list = [float(x) for x in syndrome]
    decoder.flip_syndromes(syndrome_list)

    if decoder.path_single is None:
        decoder.optimize_path(optimize=optimize_arg)

    contraction = decoder.contractor_config.contractor(
        decoder.full_tn.get_equation(output_inds=(decoder.logical_obs_inds[0],)),
        decoder.full_tn.arrays,
        optimize=decoder.path_single,
        slicing=decoder.slicing_single,
        device_id=decoder.contractor_config.device_id,
    )
    z0, z1 = np.asarray(contraction, dtype=np.float64).reshape(2)
    return float(z0), float(z1)


def build_decoder(
    h_matrix: np.ndarray,
    logical_obs: np.ndarray,
    noise_model: np.ndarray,
    device: str,
) -> object:
    return qec.get_decoder(
        "tensor_network_decoder",
        h_matrix.astype(np.uint8),
        logical_obs=logical_obs.astype(np.uint8),
        noise_model=[float(x) for x in noise_model],
        contract_noise_model=True,
        dtype="float64",
        device=device,
    )


def pick_optimize_arg(decoder: object) -> str | None:
    return None if decoder.contractor_config.contractor_name == "cutensornet" else "auto"


def format_support(vector: np.ndarray) -> str:
    support = np.flatnonzero(vector).tolist()
    return str(support)


def format_support_labels(vector: np.ndarray) -> str:
    labels: list[str] = []
    for index in np.flatnonzero(vector):
        if index < 36:
            kind = "L"
            local = int(index)
        else:
            kind = "R"
            local = int(index) - 36
        a, b = divmod(local, 6)
        labels.append(f"{kind}_{{{a},{b}}}")
    return str(labels)


def make_output_path(
    output_txt: Path | None,
    error_type: str,
    logical_mask: np.ndarray,
) -> Path:
    if output_txt is not None:
        return output_txt
    mask_bits = "".join(str(int(bit)) for bit in logical_mask.tolist())
    stem = f"bb72_cudaqx_nv_tn_flip_prob_{error_type}_a{mask_bits}.txt"
    SCRIPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return SCRIPT_RESULTS_DIR / stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BB72 CUDA-QX tensor-network example. This computes the two coset masses "
            "for one selected logical parity a·lambda in the [[72,12,6]] code."
        )
    )
    parser.add_argument("--error-type", choices=["X", "Z"], default="Z")
    parser.add_argument("--p", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--device",
        default="cuda",
        help='Decoder device: "cuda", "cuda:0", or "cpu". Default: cuda.',
    )
    parser.add_argument(
        "--logical-indices",
        default=DEFAULT_LOGICAL_INDICES,
        help=(
            "Comma-separated logical basis indices to XOR together. "
            "Default: 0,1,2,3. Ignored if --logical-mask is set."
        ),
    )
    parser.add_argument(
        "--logical-mask",
        default=None,
        help=(
            "Explicit 12-bit logical parity mask a as comma-separated 0/1 values. "
            "Example: 1,0,0,0,0,0,0,0,0,0,0,0"
        ),
    )
    parser.add_argument(
        "--logical-basis-source",
        default="auto",
        choices=["auto", "qldpc-dual", "weight6-x"],
        help=(
            "Logical-basis source. auto picks weight6-x for Z-error decoding "
            "and qldpc-dual for X-error decoding."
        ),
    )
    parser.add_argument(
        "--syndrome-mode",
        default="sample",
        choices=["sample", "zero", "manual"],
        help="How to obtain the input syndrome. Default: sample.",
    )
    parser.add_argument(
        "--syndrome-bits",
        default=None,
        help="Manual syndrome bits as comma-separated 0/1 values.",
    )
    parser.add_argument(
        "--skip-path-opt",
        action="store_true",
        help="Skip explicit optimize_path() and let decode() optimize lazily.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Optional output text file. Default: results/bb72_cudaqx_nv_tn_flip_prob_*.txt",
    )
    return parser.parse_args()


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()

    problem = build_problem(args.error_type)
    h_matrix = np.asarray(problem.check_matrix, dtype=np.uint8)
    logical_basis, logical_basis_source = resolve_logical_basis(
        problem,
        args.error_type,
        args.logical_basis_source,
    )
    logical_mask = build_logical_mask(problem.k, args.logical_indices, args.logical_mask)
    logical_obs = build_logical_observable(logical_basis, logical_mask)
    noise_model = np.full(problem.n, float(args.p), dtype=np.float64)
    output_path = make_output_path(
        args.output_txt,
        args.error_type,
        logical_mask,
    )

    syndrome, sampled_error = build_syndrome(
        problem,
        args.syndrome_mode,
        args.syndrome_bits,
        args.p,
        args.seed,
    )

    true_signature = None
    true_selected_parity = None
    if sampled_error is not None:
        true_signature = logical_signature_in_basis(sampled_error, logical_basis)
        true_selected_parity = int((logical_mask @ true_signature) % 2)

    decoder = build_decoder(h_matrix, logical_obs, noise_model, args.device)
    optimize_arg = pick_optimize_arg(decoder)

    actual_device = decoder.contractor_config.device
    contractor_name = decoder.contractor_config.contractor_name

    path_info_text = "path optimization skipped"
    path_opt_seconds = 0.0
    if not args.skip_path_opt:
        path_start = time.perf_counter()
        try:
            path_info = decoder.optimize_path(optimize=optimize_arg)
        except RuntimeError as exc:
            raise RuntimeError(
                "optimize_path() failed. On CPU-only runs, CUDA-QX may require extra "
                "cuQuantum memory-limit configuration. Try a GPU node or rerun with "
                "--skip-path-opt if you only want a minimal smoke test."
            ) from exc
        path_opt_seconds = time.perf_counter() - path_start
        path_info_text = str(path_info)

    start = time.perf_counter()
    public_result = decoder.decode([float(x) for x in syndrome])
    decode_seconds = time.perf_counter() - start

    public_p1 = float(np.asarray(public_result.result, dtype=np.float64).reshape(-1)[0])
    public_p0 = 1.0 - public_p1
    raw_start = time.perf_counter()
    tn_z0, tn_z1 = tn_raw_coset_masses(decoder, syndrome, optimize_arg)
    raw_mass_seconds = time.perf_counter() - raw_start
    tn_p0, tn_p1 = normalize_coset_masses(tn_z0, tn_z1)
    total_runtime_seconds = time.perf_counter() - total_start

    logical_basis_type = "Z" if args.error_type == "X" else "X"
    selected_indices = np.flatnonzero(logical_mask).tolist()
    logical_obs_vector = logical_obs.reshape(-1)
    lines = [
        "code=[[72,12,6]]",
        f"error_type={args.error_type}",
        f"requested_device={args.device}",
        f"actual_device={actual_device}",
        f"contractor={contractor_name}",
        f"num_checks={problem.num_checks}",
        f"num_qubits={problem.n}",
        f"num_logicals={problem.k}",
        f"p={args.p}",
        f"syndrome_mode={args.syndrome_mode}",
        f"logical_basis_type={logical_basis_type}",
        f"logical_basis_source={logical_basis_source}",
        f"logical_indices_xor={selected_indices}",
        f"logical_mask_a={logical_mask.tolist()}",
        f"logical_obs_weight={int(np.count_nonzero(logical_obs_vector))}",
        f"logical_obs_support={format_support(logical_obs_vector)}",
        f"logical_obs_support_labels={format_support_labels(logical_obs_vector)}",
        f"syndrome_weight={int(np.count_nonzero(syndrome))}",
        f"syndrome={syndrome.tolist()}",
        f"path_opt_seconds={path_opt_seconds:.6f}",
        f"decode_seconds={decode_seconds:.6f}",
        f"raw_mass_seconds={raw_mass_seconds:.6f}",
        f"total_runtime_seconds={total_runtime_seconds:.6f}",
        "",
        "Interpretation:",
        "  The decoder is answering whether a·lambda = 0 or 1 for this syndrome.",
        "  Here a is logical_mask_a and lambda is the 12-bit BB72 logical sector.",
        "  Z0/Z1 are the two aggregated sector masses after splitting all 4096 sectors",
        "  into the a·lambda=0 half and the a·lambda=1 half.",
        "",
        "Path information:",
        path_info_text,
        "",
    ]
    if sampled_error is not None:
        lines.extend(
            [
                "Sampled underlying error:",
                f"  true_error_weight={int(np.count_nonzero(sampled_error))}",
                f"  true_error_support={format_support(sampled_error)}",
                f"  true_error_support_labels={format_support_labels(sampled_error)}",
                f"  true_logical_signature_lambda={true_signature.tolist()}",
                f"  true_selected_parity_a_dot_lambda={true_selected_parity}",
                "",
            ]
        )
    lines.extend(
        [
            "Decoder outputs:",
            f"  raw_tn_masses: Z0={tn_z0:.12e}, Z1={tn_z1:.12e}",
            f"  raw_tn_probs:  P(a·lambda=0|s)={tn_p0:.12f}, P(a·lambda=1|s)={tn_p1:.12f}",
            (
                f"  public_decode: P(a·lambda=0|s)={public_p0:.12f}, "
                f"P(a·lambda=1|s)={public_p1:.12f}, "
                f"converged={bool(public_result.converged)}"
            ),
            f"  decision: {'a·lambda=1' if public_p1 >= 0.5 else 'a·lambda=0'}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_txt={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
