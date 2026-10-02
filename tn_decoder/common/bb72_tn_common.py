from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


CACHE_FORMAT_VERSION = 1


def parse_binary_vector(text: str, length: int, name: str) -> np.ndarray:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    vector = np.asarray(values, dtype=np.uint8)
    if not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must be binary")
    return vector


def enumerate_masks(num_logicals: int) -> np.ndarray:
    values = np.arange(1 << num_logicals, dtype=np.uint16)
    return ((values[:, None] >> np.arange(num_logicals, dtype=np.uint16)) & 1).astype(
        np.uint8
    )


def mask_to_bitstring(mask: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in mask.tolist())


def sector_index_from_bits(bits: np.ndarray) -> int:
    return int(sum(int(bit) << i for i, bit in enumerate(bits.tolist())))


def format_syndrome_labels(vector: np.ndarray) -> str:
    labels = [f"S_{{{index // 6},{index % 6}}}" for index in np.flatnonzero(vector)]
    return str(labels)


def fwht_inplace(values: np.ndarray) -> None:
    length = values.shape[0]
    step = 1
    while step < length:
        jump = step << 1
        for start in range(0, length, jump):
            left = values[start : start + step].copy()
            right = values[start + step : start + jump].copy()
            values[start : start + step] = left + right
            values[start + step : start + jump] = left - right
        step = jump


def reconstruct_coset_probs(flip_probabilities: np.ndarray) -> np.ndarray:
    moments = 1.0 - 2.0 * flip_probabilities.astype(np.float64, copy=False)
    spectrum = moments.copy()
    fwht_inplace(spectrum)
    probabilities = spectrum / spectrum.size
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    min_probability = float(np.min(probabilities))
    if min_probability < -1e-10:
        raise RuntimeError(
            f"Walsh-Hadamard reconstruction produced a large negative value: {min_probability}"
        )
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise RuntimeError("Reconstructed probabilities have non-positive total mass")
    return probabilities / total


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def probe_torch_cuda() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_device_name_0": None,
    }
    try:
        import torch
    except Exception as exc:
        info["torch_import_error"] = repr(exc)
        return info

    info["torch_available"] = True
    info["torch_version"] = str(torch.__version__)
    try:
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if info["torch_cuda_available"] and info["torch_cuda_device_count"] > 0:
            info["torch_cuda_device_name_0"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        info["torch_cuda_probe_error"] = repr(exc)
    return info


def array_sha256(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def build_cache_meta(
    problem: object,
    h_matrix: np.ndarray,
    logical_basis: np.ndarray,
    *,
    error_type: str,
    logical_basis_source: str,
    contractor_name: str,
) -> dict[str, Any]:
    logical_basis_type = "Z" if error_type == "X" else "X"
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "code": "[[72,12,6]]",
        "error_type": error_type,
        "logical_basis_source": logical_basis_source,
        "logical_basis_type": logical_basis_type,
        "contractor": contractor_name,
        "dtype": "float64",
        "contract_noise_model": True,
        "num_qubits": int(problem.n),
        "num_checks": int(problem.num_checks),
        "num_logicals": int(problem.k),
        "check_matrix_sha256": array_sha256(h_matrix),
        "logical_basis_sha256": array_sha256(logical_basis),
    }


def initialize_path_cache(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "meta": dict(meta),
        "paths": {},
    }


def cache_meta_matches(loaded_meta: dict[str, Any], expected_meta: dict[str, Any]) -> bool:
    for key, expected_value in expected_meta.items():
        if loaded_meta.get(key) != expected_value:
            return False
    return True


def load_path_cache(
    path: Path,
    expected_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        return initialize_path_cache(expected_meta), {
            "state": "missing",
            "entries_loaded": 0,
            "message": f"cache file does not exist: {path}",
        }

    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception as exc:
        return initialize_path_cache(expected_meta), {
            "state": "load-failed",
            "entries_loaded": 0,
            "message": f"failed to load cache {path}: {exc}",
        }

    if not isinstance(raw, dict):
        return initialize_path_cache(expected_meta), {
            "state": "invalid-format",
            "entries_loaded": 0,
            "message": f"cache {path} is not a dictionary",
        }

    loaded_meta = raw.get("meta")
    loaded_paths = raw.get("paths")
    if not isinstance(loaded_meta, dict) or not isinstance(loaded_paths, dict):
        return initialize_path_cache(expected_meta), {
            "state": "invalid-format",
            "entries_loaded": 0,
            "message": f"cache {path} is missing meta or paths sections",
        }

    if not cache_meta_matches(loaded_meta, expected_meta):
        return initialize_path_cache(expected_meta), {
            "state": "meta-mismatch",
            "entries_loaded": 0,
            "message": f"cache {path} metadata does not match the current run",
        }

    normalized_paths = {str(key): value for key, value in loaded_paths.items()}
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "meta": dict(expected_meta),
        "paths": normalized_paths,
    }, {
        "state": "loaded",
        "entries_loaded": len(normalized_paths),
        "message": f"loaded {len(normalized_paths)} cached paths from {path}",
    }


def atomic_pickle_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def optimize_decoder_path(decoder: object, optimize_arg: str | None) -> tuple[Any, float]:
    path_start = time.perf_counter()
    try:
        path_info = decoder.optimize_path(optimize=optimize_arg)
    except RuntimeError as exc:
        raise RuntimeError(
            "optimize_path() failed. On CPU-only runs, CUDA-QX may require extra "
            "cuQuantum memory-limit configuration. Try a GPU node or rerun with "
            "--skip-path-opt if you only want a minimal smoke test."
        ) from exc
    return path_info, time.perf_counter() - path_start


def make_cache_entry(
    decoder: object,
    logical_obs: np.ndarray,
    path_info_text: str,
) -> dict[str, Any]:
    path_single = getattr(decoder, "path_single", None)
    if path_single is None:
        raise RuntimeError("Decoder did not expose path_single after path optimization")
    entry = {
        "path_single": path_single,
        "slicing_single": getattr(decoder, "slicing_single", None),
        "logical_obs_weight": int(np.count_nonzero(logical_obs)),
        "logical_obs_support": np.flatnonzero(logical_obs.reshape(-1)).tolist(),
        "path_info_text": path_info_text,
    }
    try:
        pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(
            "The optimized CUDA-QX path object could not be serialized into the cache file. "
            "This usually means the installed contractor path object is not pickle-compatible."
        ) from exc
    return entry


def restore_cached_path(decoder: object, entry: dict[str, Any]) -> None:
    decoder.path_single = entry["path_single"]
    decoder.slicing_single = entry.get("slicing_single")


def raw_coset_masses_from_decoder(
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


def normalize_raw_coset_masses(
    z0: float,
    z1: float,
    *,
    relative_negative_tol: float = 1e-9,
    absolute_negative_tol: float = 1e-12,
) -> dict[str, Any]:
    raw = np.asarray([z0, z1], dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise RuntimeError(f"Raw TN masses are not finite: z0={z0}, z1={z1}")

    scale = float(np.max(np.abs(raw)))
    if scale == 0.0:
        raise RuntimeError("Raw TN masses are both zero")

    clipped = raw.copy()
    negative = np.maximum(-clipped, 0.0)
    max_negative = float(np.max(negative))
    relative_negative = max_negative / scale
    clipped_count = int(np.count_nonzero(negative > 0.0))

    if clipped_count:
        allowed = (
            max_negative <= absolute_negative_tol
            or relative_negative <= relative_negative_tol
        )
        if not allowed:
            raise RuntimeError(
                "Raw TN masses contain a negative value larger than the configured "
                f"tolerance: z0={z0}, z1={z1}, max_negative={max_negative}, "
                f"relative_negative={relative_negative}"
            )
        clipped = np.maximum(clipped, 0.0)

    total = float(np.sum(clipped))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            f"Raw TN masses have non-positive total after clipping: z0={z0}, z1={z1}"
        )

    p0 = float(clipped[0] / total)
    p1 = float(clipped[1] / total)
    return {
        "p0": p0,
        "p1": p1,
        "z0": float(raw[0]),
        "z1": float(raw[1]),
        "z0_clipped": float(clipped[0]),
        "z1_clipped": float(clipped[1]),
        "clipped_mass_count": clipped_count,
        "max_negative_mass": max_negative,
        "relative_negative_mass": relative_negative,
    }


def query_flip_probability_with_fallback(
    decoder: object,
    syndrome: np.ndarray,
    optimize_arg: str | None,
    *,
    public_probability_tol: float = 1e-12,
    raw_relative_negative_tol: float = 1e-9,
    raw_absolute_negative_tol: float = 1e-12,
) -> dict[str, Any]:
    public_decode_seconds = 0.0
    raw_fallback_seconds = 0.0
    public_p1: float | None = None
    public_converged: bool | None = None
    public_exception: str | None = None
    fallback_reason: str | None = None

    public_start = time.perf_counter()
    try:
        public_result = decoder.decode([float(x) for x in syndrome])
        public_decode_seconds = time.perf_counter() - public_start
        public_p1 = float(np.asarray(public_result.result, dtype=np.float64).reshape(-1)[0])
        converged_value = getattr(public_result, "converged", None)
        public_converged = None if converged_value is None else bool(converged_value)
    except Exception as exc:
        public_decode_seconds = time.perf_counter() - public_start
        public_exception = repr(exc)
        fallback_reason = "public_exception"

    public_valid = False
    if public_exception is None:
        if public_converged is False:
            fallback_reason = "public_not_converged"
        elif public_p1 is None or not np.isfinite(public_p1):
            fallback_reason = "public_probability_not_finite"
        elif public_p1 < -public_probability_tol or public_p1 > 1.0 + public_probability_tol:
            fallback_reason = "public_probability_out_of_range"
        else:
            public_valid = True

    if public_valid and public_p1 is not None:
        p1 = float(np.clip(public_p1, 0.0, 1.0))
        return {
            "p1": p1,
            "p0": 1.0 - p1,
            "source": "public",
            "public_valid": True,
            "public_converged": public_converged,
            "public_p1": public_p1,
            "public_exception": public_exception,
            "fallback_reason": "",
            "raw_used": False,
            "raw_z0": None,
            "raw_z1": None,
            "raw_z0_clipped": None,
            "raw_z1_clipped": None,
            "raw_p0": None,
            "raw_p1": None,
            "raw_clipped_mass_count": 0,
            "raw_max_negative_mass": 0.0,
            "raw_relative_negative_mass": 0.0,
            "public_decode_seconds": public_decode_seconds,
            "raw_fallback_seconds": raw_fallback_seconds,
            "query_decode_seconds": public_decode_seconds,
        }

    raw_start = time.perf_counter()
    raw_z0, raw_z1 = raw_coset_masses_from_decoder(decoder, syndrome, optimize_arg)
    raw_info = normalize_raw_coset_masses(
        raw_z0,
        raw_z1,
        relative_negative_tol=raw_relative_negative_tol,
        absolute_negative_tol=raw_absolute_negative_tol,
    )
    raw_fallback_seconds = time.perf_counter() - raw_start

    return {
        "p1": raw_info["p1"],
        "p0": raw_info["p0"],
        "source": "raw_fallback",
        "public_valid": False,
        "public_converged": public_converged,
        "public_p1": public_p1,
        "public_exception": public_exception,
        "fallback_reason": fallback_reason or "public_invalid",
        "raw_used": True,
        "raw_z0": raw_info["z0"],
        "raw_z1": raw_info["z1"],
        "raw_z0_clipped": raw_info["z0_clipped"],
        "raw_z1_clipped": raw_info["z1_clipped"],
        "raw_p0": raw_info["p0"],
        "raw_p1": raw_info["p1"],
        "raw_clipped_mass_count": raw_info["clipped_mass_count"],
        "raw_max_negative_mass": raw_info["max_negative_mass"],
        "raw_relative_negative_mass": raw_info["relative_negative_mass"],
        "public_decode_seconds": public_decode_seconds,
        "raw_fallback_seconds": raw_fallback_seconds,
        "query_decode_seconds": public_decode_seconds + raw_fallback_seconds,
    }
