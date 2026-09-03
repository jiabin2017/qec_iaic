from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

try:
    import cudaq_qec as qec
except ImportError as exc:  # pragma: no cover - local Windows host cannot import this package
    raise SystemExit(
        "Failed to import cudaq_qec. Run this script on the Linux machine where "
        "you installed cudaq-qec[tensor-network-decoder]."
    ) from exc


def brute_force_coset_masses(
    h_matrix: np.ndarray,
    logical_obs: np.ndarray,
    noise_model: np.ndarray,
    syndrome: np.ndarray,
) -> tuple[float, float]:
    """Enumerate all error patterns for a tiny code and sum the two coset masses."""
    z0 = 0.0
    z1 = 0.0
    num_errors = h_matrix.shape[1]

    for bits in itertools.product((0, 1), repeat=num_errors):
        error = np.fromiter(bits, dtype=np.uint8)
        trial_syndrome = (h_matrix @ error) % 2
        if not np.array_equal(trial_syndrome, syndrome):
            continue

        mass = float(np.prod(np.where(error == 1, noise_model, 1.0 - noise_model)))
        logical_flip = int((logical_obs[0] @ error) % 2)
        if logical_flip == 0:
            z0 += mass
        else:
            z1 += mass

    return z0, z1


def normalize_coset_masses(z0: float, z1: float) -> tuple[float, float]:
    total = z0 + z1
    if total <= 0.0:
        raise ValueError(f"Non-positive total coset mass: z0={z0}, z1={z1}")
    return z0 / total, z1 / total


def tn_raw_coset_masses(decoder: object, syndrome: np.ndarray) -> tuple[float, float]:
    """Use the Python-only decoder internals to expose the raw [Z0, Z1] contraction."""
    syndrome_list = [float(x) for x in syndrome]
    decoder.flip_syndromes(syndrome_list)

    if decoder.path_single is None:
        decoder.optimize_path()

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
    """Use cuQuantum pathfinding on CUDA, quimb/opt_einsum on CPU."""
    return None if decoder.contractor_config.contractor_name == "cutensornet" else "auto"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal CUDA-QX tensor-network example that computes logical coset "
            "probabilities on a [3,1] repetition code."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Decoder device: "cuda", "cuda:0", or "cpu". Default: cuda.',
    )
    parser.add_argument(
        "--skip-path-opt",
        action="store_true",
        help="Skip the explicit optimize_path() calls and let decode() optimize lazily.",
    )
    args = parser.parse_args()

    h_matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    logical_obs = np.array([[1, 1, 1]], dtype=np.uint8)
    noise_model = np.array([0.1, 0.1, 0.1], dtype=np.float64)

    syndrome_batch = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.float64,
    )

    decoder = build_decoder(h_matrix, logical_obs, noise_model, args.device)
    optimize_arg = pick_optimize_arg(decoder)
    actual_device = decoder.contractor_config.device
    contractor_name = decoder.contractor_config.contractor_name

    if not args.skip_path_opt:
        single_info = decoder.optimize_path(optimize=optimize_arg)
        batch_info = decoder.optimize_path(
            optimize=optimize_arg,
            batch_size=syndrome_batch.shape[0],
        )
        print("single_path_info:", single_info)
        print("batch_path_info:", batch_info)

    batch_start = time.perf_counter()
    batch_result = decoder.decode_batch(syndrome_batch)
    batch_seconds = time.perf_counter() - batch_start
    batch_posteriors = np.asarray(batch_result.result, dtype=np.float64).reshape(-1)
    batch_converged = np.asarray(batch_result.converged, dtype=bool)

    print(f"requested_device={args.device}")
    print(f"actual_device={actual_device}")
    print(f"contractor={contractor_name}")
    print("H=")
    print(h_matrix)
    print("logical_obs=")
    print(logical_obs)
    print(f"noise_model={noise_model.tolist()}")
    print(f"batch_decode_seconds={batch_seconds:.6f}")
    print()
    print("For this decoder, Z0/Z1 are the two logical coset masses.")
    print("The public API returns P(L=1 | s) = Z1 / (Z0 + Z1).")
    print()

    for row, syndrome in enumerate(syndrome_batch.astype(np.uint8)):
        public_result = decoder.decode([float(x) for x in syndrome])
        public_p1 = float(public_result.result[0])
        public_p0 = 1.0 - public_p1

        tn_z0, tn_z1 = tn_raw_coset_masses(decoder, syndrome)
        tn_p0, tn_p1 = normalize_coset_masses(tn_z0, tn_z1)

        brute_z0, brute_z1 = brute_force_coset_masses(
            h_matrix, logical_obs, noise_model, syndrome
        )
        brute_p0, brute_p1 = normalize_coset_masses(brute_z0, brute_z1)

        print(f"syndrome_{row}={syndrome.tolist()}")
        print(f"  raw_tn_masses: Z0={tn_z0:.12f}, Z1={tn_z1:.12f}")
        print(f"  raw_tn_probs:  P(C0|s)={tn_p0:.12f}, P(C1|s)={tn_p1:.12f}")
        print(
            f"  public_decode: P(C0|s)={public_p0:.12f}, "
            f"P(C1|s)={public_p1:.12f}, converged={bool(public_result.converged)}"
        )
        print(
            f"  batch_decode:  P(C1|s)={batch_posteriors[row]:.12f}, "
            f"converged={bool(batch_converged[row])}"
        )
        print(f"  brute_force:   Z0={brute_z0:.12f}, Z1={brute_z1:.12f}")
        print(
            f"  brute_probs:   P(C0|s)={brute_p0:.12f}, P(C1|s)={brute_p1:.12f}"
        )
        print(
            f"  abs_error:     |TN-C1 - brute-C1|={abs(tn_p1 - brute_p1):.3e}, "
            f"|public-C1 - brute-C1|={abs(public_p1 - brute_p1):.3e}"
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
