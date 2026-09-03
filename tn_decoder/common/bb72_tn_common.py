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
