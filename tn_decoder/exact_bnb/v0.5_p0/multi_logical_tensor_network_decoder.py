"""Multi-logical extension for CUDA-QX's TensorNetworkDecoder.

V5-fast multi-logical extension with v0.3 P4 workspace retention. Targeted at NVIDIA/cudaqx main commit:
    3d75f56c469343d1a7e9ecb133dccb8ad11aeb99 (2026-09-16)

This module intentionally subclasses the existing CUDA-QX decoder instead of
rewriting its TN machinery. It reuses:
  * parity-check TN construction
  * syndrome TNs
  * noise-model TNs
  * quimb TensorNetwork
  * opt_einsum / cuTensorNet contractors
  * CUDA-QX path optimization and slicing

The correctness primitive is ``contract_logical_mass``. V5-fast adds ``contract_logical_mass_fast`` using fixed-topology boundary projectors and persistent cuTensorNet Network objects. It supports three states for
logical bits:
  * fixed: select one 0/1 value before contraction,
  * open: retain as an output leg,
  * marginalized: sum over the leg before contraction.


Important implementation detail (V4)
------------------------------------
For multiple overlapping logical rows we must disable Quimb's default index
collision mangling when assembling the full network. Otherwise a physical-error
index that appears in two logical rows becomes an inner/hyper index inside the
logical subnetwork and is silently renamed when the subnetworks are combined.
That disconnects the logical observables from the actual physical-error variable.

The returned tensor contains *unnormalized* exact logical-sector/subtree masses.
A common TN normalization factor is left intact. Ratios and branch comparisons
are therefore valid, and summing child masses reproduces the parent mass.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import numpy.typing as npt
from quimb.tensor import Tensor, TensorNetwork

from cudaq_qec.plugins.decoders.tensor_network_decoder import TensorNetworkDecoder
from cudaq_qec.plugins.decoders.tensor_network_utils.contractors import optimize_path
from cudaq_qec.plugins.decoders.tensor_network_utils.tensor_network_factory import (
    tensor_network_from_parity_check,
)


@dataclass
class _FastNetworkEntry:
    network: Any
    open_logicals: tuple[int, ...]
    equation: str
    num_operands: int
    optimizer_info: Any = None
    autotuned: bool = False
    uses_qualifiers: bool = False
    gpu_resident_operands: bool = False
    boundary_gpu_operands: list[Any] | None = None
    constant_operand_indices: tuple[int, ...] = tuple()
    variable_operand_indices: tuple[int, ...] = tuple()
    p2_cache_requested: bool = False
    p2_cache_retained: bool = False
    p2_scratch_only_release_supported: bool = False
    p4_retain_scratch_requested: bool = False
    p4_scratch_retained: bool = False
    p4_scratch_reuse_hits: int = 0
    p4_scratch_evictions: int = 0
    hits: int = 0
    path_source: str = "runtime"


class MultiLogicalTensorNetworkDecoder(TensorNetworkDecoder):
    """CUDA-QX TN decoder with multiple logical observable output legs.

    Parameters are the same as :class:`TensorNetworkDecoder`, except that
    ``logical_obs`` has shape ``(k, n_errors)`` rather than being restricted to
    ``(1, n_errors)``.

    Notes
    -----
    This class is not registered as a CUDA-QX ``qec.get_decoder`` plugin on
    purpose. Instantiate it directly. This avoids changing the public CUDA-QX
    decoder registry and keeps the extension minimally invasive.
    """

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        noise_model: TensorNetwork | list[float],
        check_inds: list[str] | None = None,
        error_inds: list[str] | None = None,
        logical_inds: list[str] | None = None,
        logical_tags: list[str] | None = None,
        contract_noise_model: bool = True,
        dtype: str = "float64",
        device: str = "cuda",
    ) -> None:
        # IMPORTANT V4 DESIGN:
        #
        # Do *not* let the CUDA-QX parent constructor ever see a multi-row
        # logical TN. The parent assembles full_tn with virtual=True and the
        # default check_collisions=True. For overlapping multi-logical rows,
        # that assembly can mangle shared physical-error indices on the
        # underlying logical_tn tensors themselves. Rebuilding afterwards is
        # then too late because logical_tn has already been mutated.
        #
        # Instead:
        #   1. validate and save the requested k-row logical matrix,
        #   2. initialize the parent with only the first logical row (the exact
        #      supported stock CUDA-QX case),
        #   3. after parent initialization is complete, construct a *fresh*
        #      k-row logical_tn and assemble full_tn with
        #      check_collisions=False from the start.
        logical_obs_full = np.asarray(logical_obs)
        H_arr = np.asarray(H)
        if logical_obs_full.ndim != 2:
            raise ValueError(
                "logical_obs must be a 2D array with shape (k, n_errors)"
            )
        if logical_obs_full.shape[0] < 1:
            raise ValueError(
                "logical_obs must contain at least one logical observable"
            )
        if logical_obs_full.shape[1] != H_arr.shape[1]:
            raise ValueError(
                f"logical_obs must have {H_arr.shape[1]} columns, "
                f"got {logical_obs_full.shape[1]}"
            )

        k_requested = int(logical_obs_full.shape[0])
        if logical_inds is not None and len(logical_inds) != k_requested:
            raise ValueError(
                f"logical_inds must have length {k_requested}"
            )
        if logical_tags is not None and len(logical_tags) != k_requested:
            raise ValueError(
                f"logical_tags must have length {k_requested}"
            )

        requested_logical_inds = (
            None if logical_inds is None else list(logical_inds)
        )
        requested_logical_tags = (
            None if logical_tags is None else list(logical_tags)
        )

        self._contract_noise_model_requested = bool(contract_noise_model)

        # Parent initialization is intentionally single-logical only.
        super().__init__(
            H=H,
            logical_obs=logical_obs_full[:1],
            noise_model=noise_model,
            check_inds=check_inds,
            error_inds=error_inds,
            logical_inds=(
                None
                if requested_logical_inds is None
                else requested_logical_inds[:1]
            ),
            logical_tags=(
                None
                if requested_logical_tags is None
                else requested_logical_tags[:1]
            ),
            # Never eagerly contract the temporary parent network.
            contract_noise_model=False,
            dtype=dtype,
            device=device,
        )

        # Now install the requested multi-logical layer from scratch. This call
        # creates a brand-new logical_tn; it does not reuse the temporary
        # single-logical tensors that went through the parent assembly.
        self.replace_logical_observable(
            logical_obs_full,
            logical_inds=requested_logical_inds,
            logical_tags=requested_logical_tags,
        )

        # Cache paths by topology. Fixed *values* do not affect topology, only
        # the set of fixed indices does, so sibling nodes can share a path.
        self._logical_mass_path_cache: dict[tuple[Any, ...], tuple[Any, tuple]] = {}
        self._logical_mass_info_cache: dict[tuple[Any, ...], Any] = {}

        # V5-fast stateful cuTensorNet cache. One prepared Network is kept per
        # ordered tuple of open logical indices. For a fixed syndrome, all
        # non-open logical legs are represented by rank-1 boundary projectors,
        # so fixed-0, fixed-1, and marginalized branches share exactly the same
        # tensor topology and shapes.
        self._fast_network_cache: OrderedDict[tuple[Any, ...], _FastNetworkEntry] = OrderedDict()
        self._fast_cache_max_entries = 8
        self._fast_cache_syndrome_key: tuple[float, ...] | None = None

        # v0.1 / P0: optional offline-compiled contraction paths.  Keys are
        # ordered tuples of open logical indices.  The profile stores both the
        # contraction path and the slicing plan, so first use of a topology can
        # skip expensive path search and simply install the precompiled plan.
        self._fast_path_profile: dict[tuple[int, ...], dict[str, Any]] = {}
        self._fast_path_profile_metadata: dict[str, Any] = {}

        # v0.2 / P3: a single shared copy of the base TN operands resides on
        # the GPU and is referenced by every prepared topology.  Syndrome
        # tensors are updated in place only when the syndrome changes; static
        # code/logical/noise tensors are copied once and never touched again.
        self._gpu_base_operands: list[Any] | None = None
        self._gpu_base_syndrome_indices: tuple[int, ...] = tuple()
        self._gpu_base_constant_indices: tuple[int, ...] = tuple()
        self._gpu_base_syndrome_key: tuple[float, ...] | None = None
        self._gpu_base_nbytes: int = 0
        self._gpu_base_uploads: int = 0
        self._gpu_syndrome_updates: int = 0
        self._gpu_boundary_updates: int = 0

        # v0.3 / P4: keep cuTensorNet SCRATCH workspaces resident across
        # repeated contractions whenever a configurable device-memory budget
        # allows it.  This targets the allocation/free churn that remained in
        # v0.2 after path search, operand upload, and constant-cache reuse had
        # already been removed from the steady-state critical path.
        self._p4_scratch_evictions: int = 0
        self._p4_scratch_reuse_hits: int = 0
        self._p4_last_budget_bytes: int | None = None

    @property
    def num_logicals(self) -> int:
        return int(self.logical_obs.shape[0])

    def _rebuild_full_tn_no_mangle(self) -> None:
        """Rebuild the full TN while preserving intentionally shared indices.

        Quimb's TensorNetwork.combine() defaults to check_collisions=True.
        With multiple overlapping logical observables, a physical-error index
        (e_j) can appear two or more times *inside logical_tn*, which makes it
        an inner/hyper index. The default collision handling then mangles that
        e_j when logical_tn is combined with code_tn, silently disconnecting
        the logical layer from the physical-error variable.

        Here the collisions are intentional: check, logical, syndrome, and
        noise networks are meant to meet on the same named indices. Therefore
        we explicitly disable collision mangling.
        """
        self.full_tn = TensorNetwork()
        for part in (self.code_tn, self.logical_tn, self.syndrome_tn):
            self.full_tn = self.full_tn.combine(
                part, virtual=True, check_collisions=False
            )

        if hasattr(self, "noise_model"):
            self.full_tn = self.full_tn.combine(
                self.noise_model, virtual=True, check_collisions=False
            )

        if hasattr(self, "contractor_config"):
            self._set_tensor_type(self.full_tn)

        # Match the parent decoder's optional eager contraction of the
        # physical-error indices. contract_ind contracts *all* tensors sharing
        # e_j, which is exactly what is wanted once the shared names have been
        # preserved.
        if getattr(self, "_contract_noise_model_requested", False) and hasattr(self, "noise_model"):
            for ie in self.error_inds:
                self.full_tn.contract_ind(ie)

    @staticmethod
    def _logical_output_network(
        logical_inds: Sequence[str],
        logical_obs_inds: Sequence[str],
        logical_tags: Sequence[str],
    ) -> TensorNetwork:
        """Connect each internal logical index to its own open logical leg.

        CUDA-QX's current ``tensor_network_from_logical_observable`` helper is
        hard-coded to one logical index. Mathematically its construction is an
        identity matrix represented with Hadamard edges. Generalizing it to k
        logicals is simply the k x k identity.
        """
        k = len(logical_inds)
        if len(logical_obs_inds) != k:
            raise ValueError("logical_obs_inds must have length k")
        if len(logical_tags) != k:
            raise ValueError("logical_tags must have length k")

        return tensor_network_from_parity_check(
            np.eye(k, dtype=np.uint8),
            row_inds=list(logical_inds),
            col_inds=list(logical_obs_inds),
            tags=list(logical_tags),
        )

    def replace_logical_observable(
        self,
        logical_obs: npt.NDArray[Any],
        logical_inds: list[str] | None = None,
        logical_tags: list[str] | None = None,
    ) -> None:
        """Replace the logical-observable matrix with a k-row matrix.

        ``logical_obs[i]`` defines logical bit i using the same convention as
        the existing CUDA-QX single-logical decoder.
        """
        logical_obs = np.asarray(logical_obs)
        if logical_obs.ndim != 2:
            raise ValueError("logical_obs must be a 2D array with shape (k, n_errors)")
        if logical_obs.shape[1] != len(self.error_inds):
            raise ValueError(
                f"logical_obs must have {len(self.error_inds)} columns, "
                f"got {logical_obs.shape[1]}"
            )
        if logical_obs.shape[0] < 1:
            raise ValueError("logical_obs must contain at least one logical observable")

        k = logical_obs.shape[0]

        if logical_inds is None:
            logical_inds = [f"l_{i}" for i in range(k)]
        elif len(logical_inds) != k:
            raise ValueError(f"logical_inds must have length {k}")

        if logical_tags is None:
            logical_tags = [f"LOG_{i}" for i in range(k)]
        elif len(logical_tags) != k:
            raise ValueError(f"logical_tags must have length {k}")

        self.logical_obs = logical_obs.copy()
        self.logical_inds = list(logical_inds)
        self.logical_tags = list(logical_tags)
        self.logical_obs_inds = [f"obs_{i}" for i in range(k)]

        # 1) Logical rows couple to the physical-error indices exactly as the
        #    existing single-logical TensorNetworkDecoder does.
        self.logical_tn = tensor_network_from_parity_check(
            self.logical_obs,
            col_inds=self.error_inds,
            row_inds=self.logical_inds,
            tags=self.logical_tags,
        )

        # 2) Add one open dimension-2 output leg per logical row.
        self.logical_tn = self.logical_tn.combine(
            self._logical_output_network(
                self.logical_inds,
                self.logical_obs_inds,
                self.logical_tags,
            ),
            virtual=True,
            check_collisions=False,
        )

        # If this is a post-construction replacement, rebuild the complete TN
        # with intentional shared indices preserved.
        if hasattr(self, "full_tn"):
            self._rebuild_full_tn_no_mangle()

        if hasattr(self, "_logical_mass_path_cache"):
            self.clear_logical_mass_path_cache()
        if hasattr(self, "_fast_network_cache"):
            self.clear_fast_network_cache()
        if hasattr(self, "_fast_path_profile"):
            # A compiled path profile is tied to the logical-layer topology.
            self.clear_fast_path_profile()

    def _set_contractor(
        self,
        contractor: str,
        device: str,
        backend: str,
        dtype: str | None = None,
    ) -> None:
        # Reuse the CUDA-QX implementation, then invalidate paths because paths
        # and slicing plans can be backend/contractor dependent.
        super()._set_contractor(contractor, device, backend, dtype)
        if hasattr(self, "_logical_mass_path_cache"):
            self.clear_logical_mass_path_cache()
        if hasattr(self, "_fast_network_cache"):
            self.clear_fast_network_cache()
        if hasattr(self, "_fast_path_profile"):
            # Compiled cuTensorNet paths are contractor/backend specific.
            self.clear_fast_path_profile()

    def clear_logical_mass_path_cache(self) -> None:
        self._logical_mass_path_cache.clear()
        self._logical_mass_info_cache.clear()

    def _clear_gpu_operand_cache(self) -> None:
        """Drop v0.2 shared GPU operand references."""
        self._gpu_base_operands = None
        self._gpu_base_syndrome_indices = tuple()
        self._gpu_base_constant_indices = tuple()
        self._gpu_base_syndrome_key = None
        self._gpu_base_nbytes = 0

    def clear_fast_network_cache(self) -> None:
        """Free all persistent cuTensorNet Network objects and GPU operands."""
        for entry in list(self._fast_network_cache.values()):
            try:
                entry.network.free()
            except Exception:
                pass
        self._fast_network_cache.clear()
        self._fast_cache_syndrome_key = None
        self._last_fast_fallback = None
        if hasattr(self, "_gpu_base_operands"):
            self._clear_gpu_operand_cache()

    def clear_fast_path_profile(self) -> None:
        """Forget offline-compiled fast contraction paths.

        This does not affect correctness and does not clear ordinary V4 path
        caches. Existing persistent fast Network objects are freed because they
        may already have a different path installed.
        """
        if hasattr(self, "_fast_network_cache"):
            self.clear_fast_network_cache()
        self._fast_path_profile.clear()
        self._fast_path_profile_metadata = {}

    @staticmethod
    def _normalize_profile_open_key(key: Any) -> tuple[int, ...]:
        if isinstance(key, tuple):
            return tuple(int(x) for x in key)
        if isinstance(key, list):
            return tuple(int(x) for x in key)
        if isinstance(key, str):
            text = key.strip().strip("()[]")
            if not text:
                return tuple()
            return tuple(int(x.strip()) for x in text.split(",") if x.strip())
        raise TypeError(f"unsupported path-profile key {key!r}")

    def set_fast_path_profile(self, profile: Mapping[str, Any] | Mapping[Any, Any]) -> None:
        """Install an offline-compiled cuTensorNet path/slicing profile.

        The preferred payload format is ``{"format_version": 1, "metadata":
        {...}, "paths": {...}}``.  For convenience, a direct mapping from
        ``open_logicals`` tuples to entry dictionaries is also accepted.

        Each entry must contain ``path`` and may contain ``slices`` plus
        diagnostic metadata such as benchmark time or optimizer cost.
        """
        if "paths" in profile:
            entries = profile["paths"]  # type: ignore[index]
            metadata = dict(profile.get("metadata", {}))  # type: ignore[union-attr]
        else:
            entries = profile
            metadata = {}

        normalized: dict[tuple[int, ...], dict[str, Any]] = {}
        for raw_key, raw_entry in entries.items():  # type: ignore[union-attr]
            key = self._normalize_profile_open_key(raw_key)
            if not isinstance(raw_entry, Mapping):
                raise TypeError(f"path-profile entry for {key} must be a mapping")
            if "path" not in raw_entry:
                raise ValueError(f"path-profile entry for {key} has no 'path'")
            path = [tuple(int(v) for v in pair) for pair in raw_entry["path"]]
            slices = [tuple(item) for item in raw_entry.get("slices", ())]
            row = dict(raw_entry)
            row["path"] = path
            row["slices"] = slices
            normalized[key] = row

        # Do not let already-prepared Networks hide whether the new profile is
        # actually being used.  The path profile itself survives syndrome
        # changes and future fast-cache evictions.
        self.clear_fast_network_cache()
        self._fast_path_profile = normalized
        self._fast_path_profile_metadata = metadata

    def load_fast_path_profile(self, filename: str | Path) -> dict[str, Any]:
        """Load a locally generated pickle path profile and install it."""
        path = Path(filename)
        with path.open("rb") as fh:
            profile = pickle.load(fh)
        if not isinstance(profile, Mapping):
            raise TypeError("path profile must contain a mapping")
        version = profile.get("format_version", 1)
        if int(version) not in (1, 2):
            raise ValueError(f"unsupported path-profile format_version={version}")
        self.set_fast_path_profile(profile)
        return {
            "filename": str(path),
            "num_topologies": len(self._fast_path_profile),
            "open_logicals": sorted(self._fast_path_profile),
            "metadata": dict(self._fast_path_profile_metadata),
        }

    def fast_path_profile_info(self) -> dict[str, Any]:
        """Return metadata for the currently installed offline path profile."""
        return {
            "num_topologies": len(self._fast_path_profile),
            "open_logicals": sorted(self._fast_path_profile),
            "metadata": dict(self._fast_path_profile_metadata),
            "entries": {
                key: {
                    name: value
                    for name, value in row.items()
                    if name not in ("path", "slices")
                }
                for key, row in self._fast_path_profile.items()
            },
        }

    def _compiled_optimizer_options(self, open_list: Sequence[int]) -> dict[str, Any] | None:
        # P5.1: prefer an exact ordered-key entry, but fall back to the
        # canonical sorted logical *set*.  For a fixed set of open logicals the
        # operand graph is identical; changing within-block order only permutes
        # output axes.  cuTensorNet contraction paths/slicing are therefore
        # reusable across e.g. (6,7,8,9,10,11) and (11,10,9,8,7,6).
        ordered = tuple(int(i) for i in open_list)
        row = self._fast_path_profile.get(ordered)
        if row is None:
            canonical = tuple(sorted(ordered))
            row = self._fast_path_profile.get(canonical)
        if row is None:
            return None
        return {
            "path": row["path"],
            "slicing": row.get("slices", ()),
        }

    def _invalidate_constant_fast_networks(self) -> None:
        """Invalidate only Networks whose base operands were marked constant.

        v0.1 keeps ordinary (non-qualified) Networks across syndrome changes.
        Constant-intermediate reuse is deliberately postponed to P2; if a
        caller enables qualifiers manually, those Networks must still be
        rebuilt when syndrome values change.
        """
        doomed = [
            key for key, entry in self._fast_network_cache.items()
            if entry.uses_qualifiers
        ]
        for key in doomed:
            entry = self._fast_network_cache.pop(key)
            try:
                entry.network.free()
            except Exception:
                pass

    def set_fast_cache_max_entries(self, max_entries: int) -> None:
        """Set the LRU capacity for persistent fast Network topologies."""
        max_entries = int(max_entries)
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._fast_cache_max_entries = max_entries
        while len(self._fast_network_cache) > max_entries:
            _, entry = self._fast_network_cache.popitem(last=False)
            try:
                entry.network.free()
            except Exception:
                pass

    def decode(self, syndrome: list[float]):
        """Retain parent behavior only for k=1.

        For k>1, silently returning the first logical marginal would be easy to
        misinterpret, so force callers to use ``contract_logical_mass``.
        """
        if self.num_logicals != 1:
            raise RuntimeError(
                "decode() is ambiguous for multiple logical observables. "
                "Use contract_logical_mass(..., open_logicals=[...])."
            )
        return super().decode(syndrome)

    def _validate_logical_request(
        self,
        fixed_logicals: Mapping[int, int] | None,
        open_logicals: Sequence[int] | None,
    ) -> tuple[dict[int, int], list[int], list[int]]:
        fixed = dict(fixed_logicals or {})
        open_list = list(open_logicals or [])

        if len(open_list) != len(set(open_list)):
            raise ValueError("open_logicals contains duplicates")

        for i, bit in fixed.items():
            if not isinstance(i, (int, np.integer)):
                raise TypeError("fixed logical indices must be integers")
            if int(i) < 0 or int(i) >= self.num_logicals:
                raise IndexError(f"logical index {i} out of range")
            if int(bit) not in (0, 1):
                raise ValueError(f"fixed logical value for {i} must be 0 or 1")

        for i in open_list:
            if not isinstance(i, (int, np.integer)):
                raise TypeError("open logical indices must be integers")
            if int(i) < 0 or int(i) >= self.num_logicals:
                raise IndexError(f"logical index {i} out of range")

        fixed = {int(i): int(bit) for i, bit in fixed.items()}
        open_list = [int(i) for i in open_list]

        overlap = set(fixed).intersection(open_list)
        if overlap:
            raise ValueError(
                f"logical indices cannot be both fixed and open: {sorted(overlap)}"
            )

        marginalized = [
            i for i in range(self.num_logicals)
            if i not in fixed and i not in set(open_list)
        ]
        return fixed, open_list, marginalized

    def _prepare_mass_network(
        self,
        fixed: Mapping[int, int],
        open_list: Sequence[int],
        marginalized: Sequence[int],
    ) -> tuple[TensorNetwork, tuple[str, ...]]:
        """Create a non-mutating TN view for one subtree query."""
        tn = self.full_tn

        # Fixing an outer logical index by integer selection is exactly a
        # projection onto |0> or |1>, and removes that index before contraction.
        if fixed:
            selectors = {
                self.logical_obs_inds[i]: int(bit) for i, bit in fixed.items()
            }
            tn = tn.isel(selectors, inplace=False)

        # Explicitly sum all logical indices that are neither fixed nor open.
        # Quimb documents sum_reduce as equivalent to contracting a vector of
        # ones into the index, i.e. classical marginalization.
        for i in marginalized:
            tn = tn.sum_reduce(self.logical_obs_inds[i], inplace=False)

        output_inds = tuple(self.logical_obs_inds[i] for i in open_list)
        return tn, output_inds

    def _path_cache_key(
        self,
        fixed: Mapping[int, int],
        open_list: Sequence[int],
    ) -> tuple[Any, ...]:
        # Fixed values do not change tensor shapes, so 0/1 siblings share path.
        return (
            tuple(sorted(fixed.keys())),
            tuple(open_list),
            self.contractor_config.contractor_name,
            self.contractor_config.backend,
            self._dtype,
        )

    def optimize_logical_mass_path(
        self,
        fixed_logicals: Mapping[int, int] | None = None,
        open_logicals: Sequence[int] | None = None,
        optimize: Any = None,
    ) -> Any:
        """Optimize and cache a path for a logical subtree topology.

        The cache key depends on which logical indices are fixed/open, but not
        on the fixed 0/1 values. Thus sibling branches with the same topology can
        reuse the contraction path and slicing plan.
        """
        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )
        tn, output_inds = self._prepare_mass_network(
            fixed, open_list, marginalized
        )

        if optimize is None:
            # Match CUDA-QX defaults: cuTensorNet gets its own optimizer when
            # optimize=None; CPU Quimb uses opt_einsum's 'auto'.
            optimize_arg = (
                None
                if self.contractor_config.contractor_name == "cutensornet"
                else "auto"
            )
        else:
            optimize_arg = optimize

        path, info = optimize_path(optimize_arg, output_inds, tn)
        slices = info.slices if hasattr(info, "slices") else tuple()

        key = self._path_cache_key(fixed, open_list)
        self._logical_mass_path_cache[key] = (path, slices)
        self._logical_mass_info_cache[key] = info
        return info

    def contract_logical_mass(
        self,
        syndrome: Sequence[float],
        fixed_logicals: Mapping[int, int] | None = None,
        open_logicals: Sequence[int] | None = None,
        *,
        optimize: Any = None,
        use_path_cache: bool = True,
    ) -> Any:
        """Contract exact logical-sector/subtree masses.

        Parameters
        ----------
        syndrome:
            Syndrome soft decisions in the same convention as CUDA-QX's
            TensorNetworkDecoder. Hard syndrome bits are supplied as 0.0/1.0.
        fixed_logicals:
            Mapping ``logical_index -> 0/1``. These logical legs are selected
            before contraction.
        open_logicals:
            Ordered logical indices to retain as output dimensions. Output axis
            j corresponds to ``open_logicals[j]``.
        optimize:
            Optional path optimizer. If omitted, CUDA-QX-like defaults are used.
        use_path_cache:
            Reuse a previously optimized path for the same fixed/open topology.

        Returns
        -------
        backend array or scalar
            Unnormalized exact masses. Shape is ``(2,) * len(open_logicals)``.
            If ``open_logicals=[]``, a scalar subtree mass is returned.

        Examples
        --------
        ``open_logicals=[4, 7, 11]`` returns shape ``(2,2,2)``. Entry
        ``mass[b4,b7,b11]`` equals the mass of the already-fixed ancestor plus
        those three logical assignments, marginalized over every other logical.
        """
        syndrome_list = [float(x) for x in syndrome]
        if len(syndrome_list) != len(self.check_inds):
            raise ValueError(
                f"syndrome length {len(syndrome_list)} does not match "
                f"number of checks {len(self.check_inds)}"
            )

        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )

        # This updates the syndrome tensors shared virtually with full_tn.
        self.flip_syndromes(syndrome_list)

        tn, output_inds = self._prepare_mass_network(
            fixed, open_list, marginalized
        )
        key = self._path_cache_key(fixed, open_list)

        path = None
        slices: tuple = tuple()
        if use_path_cache and key in self._logical_mass_path_cache:
            path, slices = self._logical_mass_path_cache[key]
        else:
            if optimize is None:
                optimize_arg = (
                    None
                    if self.contractor_config.contractor_name == "cutensornet"
                    else "auto"
                )
            else:
                optimize_arg = optimize

            path, info = optimize_path(optimize_arg, output_inds, tn)
            slices = info.slices if hasattr(info, "slices") else tuple()
            if use_path_cache:
                self._logical_mass_path_cache[key] = (path, slices)
                self._logical_mass_info_cache[key] = info

        value = self.contractor_config.contractor(
            tn.get_equation(output_inds=output_inds),
            tn.arrays,
            optimize=path,
            slicing=slices,
            device_id=self.contractor_config.device_id,
        )
        return value

    def _fast_cache_key(
        self,
        open_list: Sequence[int],
        use_constant_qualifiers: bool = False,
        use_gpu_resident_operands: bool = False,
    ) -> tuple[Any, ...]:
        # Boundary projectors make fixed vs marginalized states a value-only
        # distinction. Thus one prepared network is reusable for every branch
        # sharing the same ordered set of open logical output legs.  Qualifier
        # and GPU-residency state are included because they change operand
        # ownership/lifetime semantics.
        return (
            tuple(int(i) for i in open_list),
            self.contractor_config.contractor_name,
            self.contractor_config.backend,
            self._dtype,
            self.contractor_config.device_id,
            bool(use_constant_qualifiers),
            bool(use_gpu_resident_operands),
        )

    def _logical_boundary_vector(self, state: str | int) -> Any:
        """Return a rank-1 projector in the active contractor backend.

        CUDA-QX uses NumPy operands with cuTensorNet on GPU, but switches to
        Torch operands on CPU.  V5 originally always created NumPy projectors,
        which mixed NumPy and Torch operands in CPU opt_einsum contractions.
        """
        dtype = np.dtype(self._dtype)
        if state == "sum":
            arr = np.asarray([1.0, 1.0], dtype=dtype)
        else:
            bit = int(state)
            if bit == 0:
                arr = np.asarray([1.0, 0.0], dtype=dtype)
            elif bit == 1:
                arr = np.asarray([0.0, 1.0], dtype=dtype)
            else:
                raise ValueError(f"invalid logical boundary state {state!r}")

        if self.contractor_config.backend == "torch":
            import torch
            torch_dtype = getattr(torch, self._dtype)
            return torch.as_tensor(
                arr,
                dtype=torch_dtype,
                device=self.contractor_config.device,
            )
        return arr

    def _prepare_projector_mass_network(
        self,
        fixed: Mapping[int, int],
        open_list: Sequence[int],
        marginalized: Sequence[int],
    ) -> tuple[TensorNetwork, tuple[str, ...], int]:
        """Build a fixed-topology query using rank-1 logical projectors.

        For every non-open logical leg we append exactly one dimension-2
        boundary tensor:

          fixed 0      -> [1, 0]
          fixed 1      -> [0, 1]
          marginalized -> [1, 1]

        Consequently, for a given ``open_list`` the network connectivity and
        operand shapes are identical for all BnB sibling/ancestor assignments.
        Only the projector *values* change.
        """
        marginalized_set = set(int(i) for i in marginalized)
        open_set = set(int(i) for i in open_list)
        projector_tensors: list[Tensor] = []

        for i in range(self.num_logicals):
            if i in open_set:
                continue
            if i in fixed:
                data = self._logical_boundary_vector(int(fixed[i]))
            elif i in marginalized_set:
                data = self._logical_boundary_vector("sum")
            else:  # defensive: validation should make this unreachable
                raise RuntimeError(f"logical {i} has no query state")
            projector_tensors.append(
                Tensor(
                    data=data,
                    inds=(self.logical_obs_inds[i],),
                    tags=("LOGICAL_BOUNDARY", f"BOUNDARY_{i}"),
                )
            )

        num_base_operands = len(self.full_tn.arrays)
        if projector_tensors:
            boundary_tn = TensorNetwork(projector_tensors)
            tn = self.full_tn.combine(
                boundary_tn,
                virtual=False,
                check_collisions=False,
            )
        else:
            # No boundary legs when every logical is open. Avoid copying the
            # large base TN unnecessarily.
            tn = self.full_tn

        output_inds = tuple(self.logical_obs_inds[i] for i in open_list)
        return tn, output_inds, num_base_operands

    def contract_logical_mass_projector(
        self,
        syndrome: Sequence[float],
        fixed_logicals: Mapping[int, int] | None = None,
        open_logicals: Sequence[int] | None = None,
        *,
        optimize: Any = None,
        use_path_cache: bool = True,
    ) -> Any:
        """Stateless fixed-topology contraction used to validate V5 projectors.

        This method is mathematically equivalent to ``contract_logical_mass``
        but uses boundary vectors rather than ``isel``/``sum_reduce``. It works
        on CPU or GPU and is useful as a correctness reference for V5-fast.
        """
        syndrome_list = [float(x) for x in syndrome]
        if len(syndrome_list) != len(self.check_inds):
            raise ValueError(
                f"syndrome length {len(syndrome_list)} does not match "
                f"number of checks {len(self.check_inds)}"
            )
        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )
        self.flip_syndromes(syndrome_list)
        tn, output_inds, _ = self._prepare_projector_mass_network(
            fixed, open_list, marginalized
        )

        # Projector topology depends only on open_list, not which non-open bits
        # are fixed vs marginalized or on their 0/1 values.
        key = ("projector",) + self._fast_cache_key(open_list)
        path = None
        slices: tuple = tuple()
        if use_path_cache and key in self._logical_mass_path_cache:
            path, slices = self._logical_mass_path_cache[key]
        else:
            if optimize is None:
                optimize_arg = (
                    None
                    if self.contractor_config.contractor_name == "cutensornet"
                    else "auto"
                )
            else:
                optimize_arg = optimize
            path, info = optimize_path(optimize_arg, output_inds, tn)
            slices = info.slices if hasattr(info, "slices") else tuple()
            if use_path_cache:
                self._logical_mass_path_cache[key] = (path, slices)
                self._logical_mass_info_cache[key] = info

        return self.contractor_config.contractor(
            tn.get_equation(output_inds=output_inds),
            tn.arrays,
            optimize=path,
            slicing=slices,
            device_id=self.contractor_config.device_id,
        )

    @staticmethod
    def _array_to_numpy_host(x: Any) -> np.ndarray:
        """Convert a small operand to a NumPy host array without dtype loss."""
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "get"):
            try:
                x = x.get()
            except Exception:
                pass
        return np.asarray(x)

    def _classify_base_operands(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return (constant_indices, syndrome_indices) for ``full_tn``.

        Syndrome tensors are the only base operands whose numerical values
        change between decoding requests. Code, logical-observable, and noise
        tensors are static for the lifetime of this decoder instance.
        """
        tensors = list(self.full_tn.tensors)
        arrays = list(self.full_tn.arrays)
        if len(tensors) != len(arrays):
            raise RuntimeError("full_tn tensor/array ordering mismatch")

        syndrome: list[int] = []
        for i, tensor in enumerate(tensors):
            tags = {str(tag) for tag in tensor.tags}
            if any(tag.startswith("SYN_") for tag in tags) or "SYNDROME" in tags:
                syndrome.append(i)

        if not syndrome:
            raise RuntimeError(
                "could not identify syndrome operands in full_tn; "
                "P3/P2 requires explicit variable-tensor classification"
            )
        syndrome_set = set(syndrome)
        constant = [i for i in range(len(arrays)) if i not in syndrome_set]
        return tuple(constant), tuple(syndrome)

    def _ensure_gpu_base_operands(
        self,
        syndrome_key: tuple[float, ...],
    ) -> list[Any]:
        """Create one shared CuPy copy of the base TN operands.

        Every prepared topology references these same device buffers. This is
        the core of v0.2/P3: the large static TN is uploaded exactly once,
        instead of each Network owning a separate CPU->GPU staging copy.
        """
        if self._gpu_base_operands is not None:
            return self._gpu_base_operands

        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError(
                "v0.2 GPU-resident operands require CuPy in the CUDA environment"
            ) from exc

        constant, syndrome = self._classify_base_operands()
        arrays = list(self.full_tn.arrays)
        with cp.cuda.Device(int(self.contractor_config.device_id)):
            gpu = [cp.asarray(self._array_to_numpy_host(a)) for a in arrays]

        self._gpu_base_operands = gpu
        self._gpu_base_constant_indices = constant
        self._gpu_base_syndrome_indices = syndrome
        self._gpu_base_syndrome_key = tuple(syndrome_key)
        self._gpu_base_nbytes = int(sum(int(a.nbytes) for a in gpu))
        self._gpu_base_uploads += 1
        return gpu

    @staticmethod
    def _copy_host_into_gpu(dst: Any, src: Any) -> None:
        """Update an existing CuPy allocation in place, preserving pointer identity."""
        host = MultiLogicalTensorNetworkDecoder._array_to_numpy_host(src)
        if tuple(dst.shape) != tuple(host.shape):
            raise ValueError(
                f"GPU operand shape changed from {dst.shape} to {host.shape}"
            )
        if np.dtype(dst.dtype) != np.dtype(host.dtype):
            host = host.astype(dst.dtype, copy=False)
        dst.set(host)

    def _sync_gpu_syndrome_operands(
        self,
        syndrome_key: tuple[float, ...],
    ) -> None:
        """Copy only changed syndrome tensors into shared GPU buffers."""
        if self._gpu_base_operands is None:
            return
        if self._gpu_base_syndrome_key == tuple(syndrome_key):
            return

        arrays = list(self.full_tn.arrays)
        for i in self._gpu_base_syndrome_indices:
            self._copy_host_into_gpu(self._gpu_base_operands[i], arrays[i])
        self._gpu_base_syndrome_key = tuple(syndrome_key)
        self._gpu_syndrome_updates += 1

    @staticmethod
    def _parse_memory_size_bytes(value: int | str | None, *, total_bytes: int | None = None) -> int | None:
        """Parse a workspace-retention budget.

        Accepted forms are an integer number of bytes, strings such as
        ``"28GB"`` / ``"28GiB"``, percentages such as ``"70%"``, or
        ``None`` for no proactive P4 budget. Percentages require
        ``total_bytes``.
        """
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            if int(value) < 0:
                raise ValueError("workspace budget must be non-negative")
            return int(value)

        text = str(value).strip().lower()
        if not text:
            return None
        if text.endswith("%"):
            if total_bytes is None:
                raise ValueError("percentage workspace budget requires device total memory")
            pct = float(text[:-1])
            if pct <= 0 or pct > 100:
                raise ValueError("percentage workspace budget must be in (0, 100]")
            return int(total_bytes * pct / 100.0)

        import re
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)?", text)
        if m is None:
            raise ValueError(f"cannot parse workspace budget: {value!r}")
        number = float(m.group(1))
        unit = (m.group(2) or "b").lower()
        scale = {
            "b": 1,
            "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
            "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
        }[unit]
        return int(number * scale)

    def _resolve_p4_budget_bytes(self, value: int | str | None) -> int | None:
        if value is None:
            self._p4_last_budget_bytes = None
            return None
        total = None
        if isinstance(value, str) and value.strip().endswith("%"):
            try:
                import cupy as cp
                with cp.cuda.Device(int(self.contractor_config.device_id)):
                    _free, total = cp.cuda.runtime.memGetInfo()
                total = int(total)
            except Exception as exc:
                raise RuntimeError(
                    "percentage P4 scratch budget requires CuPy/CUDA memGetInfo"
                ) from exc
        budget = self._parse_memory_size_bytes(value, total_bytes=total)
        self._p4_last_budget_bytes = budget
        return budget

    @staticmethod
    def _entry_live_scratch_bytes(entry: _FastNetworkEntry) -> int:
        if getattr(entry.network, "workspace_scratch_ptr", None) is None:
            return 0
        size = getattr(entry.network, "workspace_scratch_size", None)
        return 0 if size is None else int(size)

    def _release_entry_scratch(self, entry: _FastNetworkEntry) -> bool:
        """Release one Network's SCRATCH workspace without touching P2 CACHE."""
        fn = getattr(entry.network, "_release_workspace_memory_perhaps", None)
        if fn is None:
            return False
        if getattr(entry.network, "workspace_scratch_ptr", None) is None:
            entry.p4_scratch_retained = False
            return True
        try:
            fn("scratch", True)
            entry.p4_scratch_retained = False
            return True
        except Exception:
            return False

    def _enforce_p4_scratch_budget(
        self,
        current: _FastNetworkEntry,
        budget_bytes: int | str | None,
    ) -> int:
        """Evict LRU retained SCRATCH buffers until ``current`` can fit.

        Network/path objects and P2 CACHE workspaces remain alive. The
        ``OrderedDict`` cache already tracks LRU order because each topology is
        moved to the end on access.
        """
        budget = self._resolve_p4_budget_bytes(budget_bytes)
        if budget is None:
            return 0

        current_live = self._entry_live_scratch_bytes(current)
        current_size = getattr(current.network, "workspace_scratch_size", None)
        current_need = 0 if current_live else int(current_size or 0)
        live_other = sum(
            self._entry_live_scratch_bytes(entry)
            for entry in self._fast_network_cache.values()
            if entry is not current
        )

        evicted = 0
        if live_other + current_need <= budget:
            return 0

        # Oldest entries are first. Never evict the topology we are about to use.
        for entry in list(self._fast_network_cache.values()):
            if entry is current:
                continue
            size = self._entry_live_scratch_bytes(entry)
            if size <= 0:
                continue
            if self._release_entry_scratch(entry):
                live_other -= size
                entry.p4_scratch_evictions += 1
                self._p4_scratch_evictions += 1
                evicted += 1
            if live_other + current_need <= budget:
                break
        return evicted

    def release_fast_scratch_workspaces(self) -> int:
        """Release all retained P4 SCRATCH buffers but keep paths/Networks/cache.

        This is useful before an intentionally large one-off contraction such
        as the full-4096 validation tensor.
        """
        released = 0
        for entry in self._fast_network_cache.values():
            if self._entry_live_scratch_bytes(entry) <= 0:
                continue
            if self._release_entry_scratch(entry):
                released += 1
        return released

    def fast_workspace_info(self) -> dict[str, Any]:
        """Aggregate P4 workspace-retention diagnostics."""
        live_scratch = 0
        live_cache = 0
        planned_scratch = 0
        for entry in self._fast_network_cache.values():
            net = entry.network
            size = getattr(net, "workspace_scratch_size", None)
            if size is not None:
                planned_scratch += int(size)
            if getattr(net, "workspace_scratch_ptr", None) is not None:
                live_scratch += int(size or 0)
            csize = getattr(net, "workspace_cache_size", None)
            if getattr(net, "workspace_cache_ptr", None) is not None:
                live_cache += int(csize or 0)
        return {
            "planned_scratch_bytes": int(planned_scratch),
            "live_scratch_bytes": int(live_scratch),
            "live_cache_bytes": int(live_cache),
            "scratch_reuse_hits": int(self._p4_scratch_reuse_hits),
            "scratch_evictions": int(self._p4_scratch_evictions),
            "last_budget_bytes": self._p4_last_budget_bytes,
        }

    def _release_scratch_keep_cache(self, entry: _FastNetworkEntry) -> bool:
        """Release only SCRATCH workspace while retaining P2 CACHE data.

        cuQuantum's public ``release_workspace=True`` releases both SCRATCH and
        CACHE workspaces. P2 needs CACHE to survive between contractions. The
        current cuQuantum Network implementation exposes a private helper that
        releases one workspace kind at a time; v0.2 uses it defensively and
        reports whether it was available. v0.3 will revisit workspace policy.
        """
        fn = getattr(entry.network, "_release_workspace_memory_perhaps", None)
        if fn is None:
            entry.p2_scratch_only_release_supported = False
            return False
        try:
            fn("scratch", True)
            entry.p2_scratch_only_release_supported = True
            entry.p2_cache_retained = bool(
                getattr(entry.network, "workspace_cache_ptr", None) is not None
            )
            return True
        except Exception:
            entry.p2_scratch_only_release_supported = False
            return False

    def _purge_p2_workspaces(self) -> int:
        """Emergency memory-pressure fallback: purge retained P2 workspaces.

        Prepared Network objects and compiled paths stay alive; only cached
        constant intermediates/scratch buffers are discarded. This is not the
        normal v0.2 path and is reported through ``last_fast_fallback``.
        """
        purged = 0
        for entry in self._fast_network_cache.values():
            fn = getattr(entry.network, "_release_workspace_memory_perhaps", None)
            if fn is None:
                continue
            try:
                fn("scratch", True)
            except Exception:
                pass
            try:
                fn("cache", True)
                entry.p2_cache_retained = False
                purged += 1
            except Exception:
                pass
        return purged

    def _contract_fast_entry(
        self,
        entry: _FastNetworkEntry,
        *,
        release_workspace: bool,
        retain_constant_cache: bool,
        retain_scratch_workspace: bool = False,
        scratch_retention_budget: int | str | None = None,
    ) -> Any:
        """Execute one contraction with v0.3 workspace policy.

        P4 mode keeps SCRATCH resident across calls. Before a first allocation
        for a topology, an optional global retention budget evicts LRU SCRATCH
        buffers from other prepared topologies. P2 CACHE remains untouched.
        """
        if retain_scratch_workspace:
            entry.p4_retain_scratch_requested = True
            self._enforce_p4_scratch_budget(entry, scratch_retention_budget)
            if getattr(entry.network, "workspace_scratch_ptr", None) is not None:
                entry.p4_scratch_reuse_hits += 1
                self._p4_scratch_reuse_hits += 1

            # release_workspace=False is the entire P4 point: cuTensorNet keeps
            # SCRATCH (and, when qualifiers are active, P2 CACHE) allocated.
            out = entry.network.contract(release_workspace=False)
            entry.p4_scratch_retained = bool(
                getattr(entry.network, "workspace_scratch_ptr", None) is not None
            )
            if retain_constant_cache and entry.uses_qualifiers:
                entry.p2_cache_requested = True
                entry.p2_cache_retained = bool(
                    getattr(entry.network, "workspace_cache_ptr", None) is not None
                )
            return out

        # v0.2 behavior retained as an ablation/fallback.
        entry.p4_scratch_retained = False
        if retain_constant_cache and entry.uses_qualifiers:
            entry.p2_cache_requested = True
            scratch_release_fn = getattr(
                entry.network, "_release_workspace_memory_perhaps", None
            )
            if release_workspace and scratch_release_fn is None:
                out = entry.network.contract(release_workspace=True)
                entry.p2_scratch_only_release_supported = False
                entry.p2_cache_retained = False
                return out

            out = entry.network.contract(release_workspace=False)
            entry.p2_cache_retained = bool(
                getattr(entry.network, "workspace_cache_ptr", None) is not None
            )
            if release_workspace:
                self._release_scratch_keep_cache(entry)
            return out

        return entry.network.contract(release_workspace=release_workspace)

    def _make_cutn_qualifiers(
        self,
        num_operands: int,
        constant_operand_indices: Sequence[int],
    ) -> Any | None:
        """Mark only truly immutable operands as cuTensorNet constants.

        v0.1 marked every base operand constant, which was only safe for a
        fixed syndrome. v0.2/P2 excludes syndrome tensors and logical boundary
        projectors, allowing constant-intermediate reuse across both BnB
        branches *and* different syndrome values.
        """
        try:
            from cuquantum.bindings import cutensornet as cutn
            qualifiers = np.zeros(num_operands, dtype=cutn.tensor_qualifiers_dtype)
            names = qualifiers.dtype.names or ()
            field = None
            for candidate in ("isConstant", "is_constant"):
                if candidate in names:
                    field = candidate
                    break
            if field is None:
                return None
            for i in constant_operand_indices:
                qualifiers[field][int(i)] = 1
            return qualifiers
        except Exception:
            return None

    def _evict_fast_cache_if_needed(self) -> None:
        while len(self._fast_network_cache) > self._fast_cache_max_entries:
            _, entry = self._fast_network_cache.popitem(last=False)
            try:
                entry.network.free()
            except Exception:
                pass

    def _get_or_create_fast_network(
        self,
        tn: TensorNetwork,
        output_inds: tuple[str, ...],
        open_list: Sequence[int],
        num_base_operands: int,
        syndrome_key: tuple[float, ...],
        *,
        optimize: Any = None,
        autotune_iterations: int = 0,
        memory_limit: int | str | None = None,
        use_constant_qualifiers: bool = False,
        use_gpu_resident_operands: bool = False,
    ) -> tuple[_FastNetworkEntry, bool]:
        key = self._fast_cache_key(
            open_list,
            use_constant_qualifiers,
            use_gpu_resident_operands,
        )
        cpu_operands = list(tn.arrays)
        equation = tn.get_equation(output_inds=output_inds)

        if use_constant_qualifiers and not use_gpu_resident_operands:
            raise ValueError(
                "v0.2 P2 constant-intermediate caching requires "
                "use_gpu_resident_operands=True so variable tensors can be "
                "updated in place without replacing constant operand storage"
            )

        # P3: make sure the shared base GPU buffers reflect the current syndrome
        # before either creating or reusing a topology-specific Network.
        gpu_base = None
        if use_gpu_resident_operands:
            gpu_base = self._ensure_gpu_base_operands(syndrome_key)
            self._sync_gpu_syndrome_operands(syndrome_key)

        if key in self._fast_network_cache:
            entry = self._fast_network_cache.pop(key)
            # Reinsert at the end for LRU behavior.
            self._fast_network_cache[key] = entry
            if entry.equation != equation or entry.num_operands != len(cpu_operands):
                try:
                    entry.network.free()
                except Exception:
                    pass
                del self._fast_network_cache[key]
            else:
                if entry.gpu_resident_operands:
                    # Base operands are shared and already syndrome-synchronized.
                    # Only the tiny topology-local logical boundary vectors vary
                    # from one BnB node to the next. Keep their device pointers
                    # stable and overwrite values in place.
                    boundary_cpu = cpu_operands[num_base_operands:]
                    boundary_gpu = entry.boundary_gpu_operands or []
                    if len(boundary_cpu) != len(boundary_gpu):
                        raise RuntimeError(
                            "boundary operand count changed for a fixed topology"
                        )
                    for dst, src in zip(boundary_gpu, boundary_cpu):
                        self._copy_host_into_gpu(dst, src)
                    self._gpu_boundary_updates += len(boundary_gpu)
                    # No reset_operands(): the Network keeps the same GPU
                    # pointers while values change in-place.
                else:
                    entry.network.reset_operands(*cpu_operands)

                if int(autotune_iterations) > 0 and not entry.autotuned:
                    entry.network.autotune(iterations=int(autotune_iterations))
                    entry.autotuned = True
                entry.hits += 1
                return entry, False

        try:
            from cuquantum.tensornet import Network
        except Exception as exc:
            raise RuntimeError(
                "contract_logical_mass_fast requires cuquantum.tensornet.Network"
            ) from exc

        options: dict[str, Any] = {
            "device_id": self.contractor_config.device_id,
        }
        if memory_limit is not None:
            options["memory_limit"] = memory_limit

        boundary_gpu: list[Any] | None = None
        if use_gpu_resident_operands:
            assert gpu_base is not None
            try:
                import cupy as cp
            except Exception as exc:
                raise RuntimeError("CuPy disappeared after GPU operand setup") from exc
            with cp.cuda.Device(int(self.contractor_config.device_id)):
                boundary_gpu = [
                    cp.asarray(self._array_to_numpy_host(a))
                    for a in cpu_operands[num_base_operands:]
                ]
            operands = list(gpu_base) + boundary_gpu
            constant_indices = tuple(self._gpu_base_constant_indices)
            variable_indices = tuple(
                list(self._gpu_base_syndrome_indices)
                + list(range(num_base_operands, len(operands)))
            )
        else:
            operands = cpu_operands
            constant_indices = tuple(
                i for i in range(num_base_operands)
                if i not in set(self._gpu_base_syndrome_indices)
            )
            variable_indices = tuple(
                i for i in range(len(operands)) if i not in set(constant_indices)
            )

        qualifiers = (
            self._make_cutn_qualifiers(len(operands), constant_indices)
            if use_constant_qualifiers
            else None
        )
        if use_constant_qualifiers and qualifiers is None:
            raise RuntimeError(
                "P2 requested, but this cuQuantum installation does not expose "
                "tensor constant qualifiers"
            )

        kwargs: dict[str, Any] = {"options": options}
        if qualifiers is not None:
            kwargs["qualifiers"] = qualifiers

        network = Network(equation, *operands, **kwargs)

        # v0.1 / P0: install the offline-compiled path+slicing plan.
        opt_arg = optimize
        path_source = "explicit" if optimize is not None else "runtime_search"
        if optimize is None:
            compiled = self._compiled_optimizer_options(open_list)
            if compiled is not None:
                opt_arg = compiled
                path_source = "compiled_profile"

        try:
            path, info = network.contract_path(optimize=opt_arg)
        except Exception:
            if path_source != "compiled_profile":
                raise
            try:
                network.free()
            except Exception:
                pass
            network = Network(equation, *operands, **kwargs)
            path, info = network.contract_path(optimize=None)
            path_source = "runtime_fallback_from_profile"

        autotuned = False
        if int(autotune_iterations) > 0:
            network.autotune(iterations=int(autotune_iterations))
            autotuned = True

        entry = _FastNetworkEntry(
            network=network,
            open_logicals=tuple(int(i) for i in open_list),
            equation=equation,
            num_operands=len(operands),
            optimizer_info=info,
            autotuned=autotuned,
            uses_qualifiers=qualifiers is not None,
            gpu_resident_operands=bool(use_gpu_resident_operands),
            boundary_gpu_operands=boundary_gpu,
            constant_operand_indices=tuple(int(i) for i in constant_indices),
            variable_operand_indices=tuple(int(i) for i in variable_indices),
            p2_cache_requested=bool(use_constant_qualifiers),
            hits=0,
            path_source=path_source,
        )
        self._fast_network_cache[key] = entry
        self._evict_fast_cache_if_needed()
        return entry, True

    def contract_logical_mass_fast(
        self,
        syndrome: Sequence[float],
        fixed_logicals: Mapping[int, int] | None = None,
        open_logicals: Sequence[int] | None = None,
        *,
        optimize: Any = None,
        autotune_iterations: int = 0,
        memory_limit: int | str | None = None,
        release_workspace: bool = False,
        fallback_to_projector: bool = True,
        use_constant_qualifiers: bool = False,
        use_gpu_resident_operands: bool = False,
        retain_constant_cache: bool = False,
        retain_scratch_workspace: bool = False,
        scratch_retention_budget: int | str | None = None,
    ) -> Any:
        """Fast repeated logical-subtree contraction for GPU/cuTensorNet.

        v0.2 adds two orthogonal optimizations on top of v0.1:

        P3 -- GPU-resident operands
            One shared CuPy copy of the base TN is uploaded once. Syndrome
            tensors and topology-local logical boundary projectors are then
            overwritten in place; cache hits do not call ``reset_operands``.

        P2 -- constant-intermediate caching
            Code/logical/noise tensors are marked ``isConstant`` while syndrome
            tensors and boundary projectors remain variable. cuTensorNet can
            therefore cache intermediate contractions that depend only on the
            immutable tensors and reuse them across branches and syndromes.

        P4 -- retained SCRATCH workspace
            With ``retain_scratch_workspace=True`` the topology's cuTensorNet
            SCRATCH allocation survives repeated contractions. An optional
            ``scratch_retention_budget`` evicts least-recently-used SCRATCH
            buffers from other topologies while keeping prepared Networks,
            compiled paths, GPU operands, and P2 CACHE state alive.

        Setting ``retain_scratch_workspace=False`` recovers the v0.2 policy:
        release SCRATCH after each contraction while retaining P2 CACHE.
        """
        syndrome_list = [float(x) for x in syndrome]
        if len(syndrome_list) != len(self.check_inds):
            raise ValueError(
                f"syndrome length {len(syndrome_list)} does not match "
                f"number of checks {len(self.check_inds)}"
            )

        if retain_constant_cache and not use_constant_qualifiers:
            raise ValueError(
                "retain_constant_cache=True requires use_constant_qualifiers=True"
            )
        if use_constant_qualifiers and not use_gpu_resident_operands:
            raise ValueError(
                "v0.2 P2 requires use_gpu_resident_operands=True"
            )

        if self.contractor_config.contractor_name != "cutensornet":
            if fallback_to_projector:
                return self.contract_logical_mass_projector(
                    syndrome_list,
                    fixed_logicals=fixed_logicals,
                    open_logicals=open_logicals,
                    optimize=optimize,
                )
            raise RuntimeError(
                "persistent Network mode requires the cutensornet contractor"
            )

        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )

        syndrome_key = tuple(syndrome_list)
        self._fast_cache_syndrome_key = syndrome_key

        # Update the CPU-side syndrome tensors (still used to build/validate the
        # fixed-topology query) and then synchronize only the tiny variable
        # syndrome operands if the P3 GPU cache already exists.
        self.flip_syndromes(syndrome_list)
        if use_gpu_resident_operands:
            self._sync_gpu_syndrome_operands(syndrome_key)

        tn, output_inds, num_base_operands = self._prepare_projector_mass_network(
            fixed, open_list, marginalized
        )
        entry, _created = self._get_or_create_fast_network(
            tn,
            output_inds,
            open_list,
            num_base_operands,
            syndrome_key,
            optimize=optimize,
            autotune_iterations=autotune_iterations,
            memory_limit=memory_limit,
            use_constant_qualifiers=use_constant_qualifiers,
            use_gpu_resident_operands=use_gpu_resident_operands,
        )

        self._last_fast_fallback = None
        try:
            return self._contract_fast_entry(
                entry,
                release_workspace=release_workspace,
                retain_constant_cache=retain_constant_cache,
                retain_scratch_workspace=retain_scratch_workspace,
                scratch_retention_budget=scratch_retention_budget,
            )
        except Exception as exc:
            text = str(exc)
            memoryish = any(
                token in text
                for token in (
                    "INTERNAL_ERROR",
                    "Failed to allocate memory",
                    "OUT_OF_MEMORY",
                    "CUDA_ERROR_OUT_OF_MEMORY",
                )
            )
            if not memoryish:
                raise

            key = self._fast_cache_key(
                open_list,
                use_constant_qualifiers,
                use_gpu_resident_operands,
            )
            bad = self._fast_network_cache.pop(key, None)
            if bad is not None:
                try:
                    bad.network.free()
                except Exception:
                    pass

            # P4 memory-pressure fallback: first drop all retained SCRATCH
            # buffers, even when the P2 constant cache is disabled. If P2 is
            # active, purge its CACHE workspace as well before the safe retry.
            purged_p4_scratch = (
                self.release_fast_scratch_workspaces()
                if retain_scratch_workspace else 0
            )
            purged_p2 = self._purge_p2_workspaces() if retain_constant_cache else 0

            retry_memory_limit = (
                memory_limit if memory_limit is not None else "40%"
            )
            try:
                retry_entry, _ = self._get_or_create_fast_network(
                    tn,
                    output_inds,
                    open_list,
                    num_base_operands,
                    syndrome_key,
                    optimize=optimize,
                    autotune_iterations=0,
                    memory_limit=retry_memory_limit,
                    use_constant_qualifiers=use_constant_qualifiers,
                    use_gpu_resident_operands=use_gpu_resident_operands,
                )
                self._last_fast_fallback = {
                    "kind": "cutensornet_retry",
                    "memory_limit": retry_memory_limit,
                    "purged_p4_scratch_workspaces": purged_p4_scratch,
                    "purged_p2_workspaces": purged_p2,
                    "p4_retention_disabled_on_retry": bool(retain_scratch_workspace),
                    "original_error": text,
                }
                return self._contract_fast_entry(
                    retry_entry,
                    release_workspace=True,
                    retain_constant_cache=retain_constant_cache,
                    retain_scratch_workspace=False,
                    scratch_retention_budget=None,
                )
            except Exception as retry_exc:
                if not fallback_to_projector:
                    raise retry_exc from exc

                self._last_fast_fallback = {
                    "kind": "stateless_projector",
                    "memory_limit": retry_memory_limit,
                    "original_error": text,
                    "retry_error": str(retry_exc),
                }
                return self.contract_logical_mass_projector(
                    syndrome_list,
                    fixed_logicals=fixed_logicals,
                    open_logicals=open_logicals,
                    optimize=optimize,
                )

    def prepare_fast_logical_mass(
        self,
        syndrome: Sequence[float],
        fixed_logicals: Mapping[int, int] | None = None,
        open_logicals: Sequence[int] | None = None,
        *,
        optimize: Any = None,
        autotune_iterations: int = 3,
        memory_limit: int | str | None = None,
        use_constant_qualifiers: bool = False,
        use_gpu_resident_operands: bool = False,
    ) -> dict[str, Any]:
        """Prepare/autotune one persistent topology without executing it."""
        syndrome_list = [float(x) for x in syndrome]
        if self.contractor_config.contractor_name != "cutensornet":
            raise RuntimeError("prepare_fast_logical_mass requires cutensornet")
        if use_constant_qualifiers and not use_gpu_resident_operands:
            raise ValueError("P2 requires use_gpu_resident_operands=True")
        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )
        syndrome_key = tuple(syndrome_list)
        self._fast_cache_syndrome_key = syndrome_key
        self.flip_syndromes(syndrome_list)
        if use_gpu_resident_operands:
            self._sync_gpu_syndrome_operands(syndrome_key)
        tn, output_inds, num_base_operands = self._prepare_projector_mass_network(
            fixed, open_list, marginalized
        )
        entry, created = self._get_or_create_fast_network(
            tn,
            output_inds,
            open_list,
            num_base_operands,
            syndrome_key,
            optimize=optimize,
            autotune_iterations=autotune_iterations,
            memory_limit=memory_limit,
            use_constant_qualifiers=use_constant_qualifiers,
            use_gpu_resident_operands=use_gpu_resident_operands,
        )
        info = entry.optimizer_info
        return {
            "created": created,
            "open_logicals": tuple(open_list),
            "autotuned": entry.autotuned,
            "uses_constant_qualifiers": entry.uses_qualifiers,
            "gpu_resident_operands": entry.gpu_resident_operands,
            "hits": entry.hits,
            "largest_intermediate": getattr(info, "largest_intermediate", None),
            "opt_cost": getattr(info, "opt_cost", None),
            "num_slices": getattr(info, "num_slices", None),
        }

    def last_fast_fallback(self) -> Any | None:
        """Return metadata for the most recent V5.1 fallback/retry, if any."""
        return self._last_fast_fallback

    def fast_cache_info(self) -> list[dict[str, Any]]:
        """Return lightweight metadata for currently prepared fast topologies."""
        rows: list[dict[str, Any]] = []
        for key, entry in self._fast_network_cache.items():
            info = entry.optimizer_info
            net = entry.network
            rows.append({
                "key": key,
                "open_logicals": entry.open_logicals,
                "hits": entry.hits,
                "autotuned": entry.autotuned,
                "uses_constant_qualifiers": entry.uses_qualifiers,
                "gpu_resident_operands": entry.gpu_resident_operands,
                "p2_cache_requested": entry.p2_cache_requested,
                "p2_cache_retained": entry.p2_cache_retained,
                "scratch_only_release_supported": entry.p2_scratch_only_release_supported,
                "p4_retain_scratch_requested": entry.p4_retain_scratch_requested,
                "p4_scratch_retained": entry.p4_scratch_retained,
                "p4_scratch_reuse_hits": entry.p4_scratch_reuse_hits,
                "p4_scratch_evictions": entry.p4_scratch_evictions,
                "constant_operands": len(entry.constant_operand_indices),
                "variable_operands": len(entry.variable_operand_indices),
                "path_source": entry.path_source,
                "largest_intermediate": getattr(info, "largest_intermediate", None),
                "opt_cost": getattr(info, "opt_cost", None),
                "num_slices": getattr(info, "num_slices", None),
                "workspace_scratch_size": getattr(net, "workspace_scratch_size", None),
                "workspace_cache_size": getattr(net, "workspace_cache_size", None),
                "scratch_workspace_live": getattr(net, "workspace_scratch_ptr", None) is not None,
                "cache_workspace_live": getattr(net, "workspace_cache_ptr", None) is not None,
            })
        return rows

    def fast_gpu_operand_info(self) -> dict[str, Any]:
        """Return v0.2/P3 shared-device-operand diagnostics."""
        return {
            "enabled": self._gpu_base_operands is not None,
            "base_operands": 0 if self._gpu_base_operands is None else len(self._gpu_base_operands),
            "base_nbytes": int(self._gpu_base_nbytes),
            "constant_base_operands": len(self._gpu_base_constant_indices),
            "syndrome_base_operands": len(self._gpu_base_syndrome_indices),
            "base_uploads": int(self._gpu_base_uploads),
            "syndrome_updates": int(self._gpu_syndrome_updates),
            "boundary_updates": int(self._gpu_boundary_updates),
        }

    @staticmethod
    def normalized(mass: Any) -> np.ndarray:
        """Convert a mass tensor to a NumPy array and normalize its total to 1."""
        # CUDA-QX currently uses torch on CPU and numpy input with cuTensorNet on
        # GPU. Handle the CPU torch case without making torch a hard dependency.
        if hasattr(mass, "detach"):
            mass = mass.detach()
        if hasattr(mass, "cpu"):
            mass = mass.cpu()
        if hasattr(mass, "get"):
            try:
                mass = mass.get()
            except Exception:
                pass
        arr = np.asarray(mass, dtype=np.float64)
        total = float(arr.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"cannot normalize mass tensor with total={total}")
        return arr / total
