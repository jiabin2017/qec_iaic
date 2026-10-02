"""Multi-logical extension for CUDA-QX's TensorNetworkDecoder.

V5-fast multi-logical extension. Targeted at NVIDIA/cudaqx main commit:
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

    def clear_fast_network_cache(self) -> None:
        """Free all persistent cuTensorNet Network objects held by V5-fast."""
        for entry in list(self._fast_network_cache.values()):
            try:
                entry.network.free()
            except Exception:
                pass
        self._fast_network_cache.clear()
        self._fast_cache_syndrome_key = None
        self._last_fast_fallback = None

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
        if int(version) != 1:
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
        row = self._fast_path_profile.get(tuple(int(i) for i in open_list))
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
    ) -> tuple[Any, ...]:
        # Boundary projectors make fixed vs marginalized states a value-only
        # distinction. Thus one prepared network is reusable for every branch
        # sharing the same ordered set of open logical output legs.  Qualifier
        # state is included because a Network with constant operands has
        # different cross-syndrome validity rules.
        return (
            tuple(int(i) for i in open_list),
            self.contractor_config.contractor_name,
            self.contractor_config.backend,
            self._dtype,
            self.contractor_config.device_id,
            bool(use_constant_qualifiers),
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

    def _make_cutn_qualifiers(
        self,
        num_operands: int,
        num_base_operands: int,
    ) -> Any | None:
        """Mark base tensors constant and boundary projectors variable.

        For one fixed syndrome all tensors in ``full_tn`` are constant across
        BnB branches; only appended rank-1 boundary projectors change. The
        qualifier is a forward-compatible hook for cuTensorNet's constant
        intermediate reuse machinery. If the installed binding exposes a
        differently named field or qualifiers are unavailable, V5-fast simply
        falls back to no qualifiers while retaining stateful Network reuse.
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
            if num_base_operands:
                qualifiers[field][:num_base_operands] = 1
            # Projector operands deliberately remain non-constant.
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
        *,
        optimize: Any = None,
        autotune_iterations: int = 0,
        memory_limit: int | str | None = None,
        use_constant_qualifiers: bool = False,
    ) -> tuple[_FastNetworkEntry, bool]:
        key = self._fast_cache_key(open_list, use_constant_qualifiers)
        operands = list(tn.arrays)
        equation = tn.get_equation(output_inds=output_inds)

        if key in self._fast_network_cache:
            entry = self._fast_network_cache.pop(key)
            # Reinsert at the end for LRU behavior.
            self._fast_network_cache[key] = entry
            if entry.equation != equation or entry.num_operands != len(operands):
                # This should not occur for a fixed open_list, but fail safely.
                try:
                    entry.network.free()
                except Exception:
                    pass
                del self._fast_network_cache[key]
            else:
                entry.network.reset_operands(*operands)
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

        qualifiers = (
            self._make_cutn_qualifiers(len(operands), num_base_operands)
            if use_constant_qualifiers
            else None
        )
        kwargs: dict[str, Any] = {"options": options}
        if qualifiers is not None:
            kwargs["qualifiers"] = qualifiers

        network = Network(equation, *operands, **kwargs)

        # v0.1 / P0: if the caller did not explicitly request an optimizer,
        # install the offline-compiled path+slicing plan for this topology.
        # Passing an explicit path to contract_path() does not run the expensive
        # hyper-path search; it only installs/validates the supplied plan.
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
            # A profile can become stale across cuQuantum/code/topology changes.
            # Correctness is more important than forcing it: fall back once to
            # the native runtime optimizer and mark the source accordingly.
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
    ) -> Any:
        """Fast repeated logical-subtree contraction for GPU/cuTensorNet.

        Compared with V4 ``contract_logical_mass``:

        1. non-open logicals are represented by fixed-shape rank-1 boundary
           projectors instead of topology-changing ``isel``/``sum_reduce``;
        2. one persistent ``cuquantum.tensornet.Network`` is cached per ordered
           ``open_logicals`` tuple;
        3. contraction path, workspace, and optional autotuning are therefore
           amortized over repeated BnB branch queries;
        4. optionally, for a fixed syndrome, base tensors can be marked
           constant via cuTensorNet qualifiers (experimental; off by default).

        v0.1 reuses ordinary prepared Networks across syndrome changes because
        syndrome values alter operands but not tensor topology; cache hits call
        ``reset_operands`` with the newly flipped syndrome tensors.  Only
        Networks created with constant qualifiers are invalidated on a syndrome
        change, since P2 constant-intermediate caching is not enabled by the
        standard v0.1 runner.
        """
        syndrome_list = [float(x) for x in syndrome]
        if len(syndrome_list) != len(self.check_inds):
            raise ValueError(
                f"syndrome length {len(syndrome_list)} does not match "
                f"number of checks {len(self.check_inds)}"
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
                "V5-fast persistent Network mode requires the cutensornet contractor"
            )

        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )

        syndrome_key = tuple(syndrome_list)
        if (
            self._fast_cache_syndrome_key is not None
            and syndrome_key != self._fast_cache_syndrome_key
        ):
            # P1: for the normal v0.1 mode (qualifiers=False), syndrome values
            # change only operand contents, not equation/topology, so prepared
            # Networks and paths remain valid and are reused.  Any experimental
            # qualifier-enabled Networks are conservatively invalidated.
            self._invalidate_constant_fast_networks()
        self._fast_cache_syndrome_key = syndrome_key

        self.flip_syndromes(syndrome_list)
        tn, output_inds, num_base_operands = self._prepare_projector_mass_network(
            fixed, open_list, marginalized
        )
        entry, _created = self._get_or_create_fast_network(
            tn,
            output_inds,
            open_list,
            num_base_operands,
            optimize=optimize,
            autotune_iterations=autotune_iterations,
            memory_limit=memory_limit,
            use_constant_qualifiers=use_constant_qualifiers,
        )

        self._last_fast_fallback = None
        try:
            return entry.network.contract(
                release_workspace=release_workspace
            )
        except Exception as exc:
            # Some cuTensorNet versions can report CUTENSORNET_STATUS_INTERNAL_ERROR
            # for a prepared Network whose chosen workspace/path is too aggressive
            # for the currently available device memory.  Retry once with a lower
            # memory ceiling and with workspace release enabled.
            if "INTERNAL_ERROR" not in str(exc):
                raise

            key = self._fast_cache_key(open_list, use_constant_qualifiers)
            bad = self._fast_network_cache.pop(key, None)
            if bad is not None:
                try:
                    bad.network.free()
                except Exception:
                    pass

            retry_memory_limit = (
                memory_limit if memory_limit is not None else "40%"
            )
            try:
                retry_entry, _ = self._get_or_create_fast_network(
                    tn,
                    output_inds,
                    open_list,
                    num_base_operands,
                    optimize=optimize,
                    autotune_iterations=0,
                    memory_limit=retry_memory_limit,
                    use_constant_qualifiers=use_constant_qualifiers,
                )
                self._last_fast_fallback = {
                    "kind": "cutensornet_retry",
                    "memory_limit": retry_memory_limit,
                    "original_error": str(exc),
                }
                return retry_entry.network.contract(
                    release_workspace=True
                )
            except Exception as retry_exc:
                if not fallback_to_projector:
                    raise retry_exc from exc

                self._last_fast_fallback = {
                    "kind": "stateless_projector",
                    "memory_limit": retry_memory_limit,
                    "original_error": str(exc),
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
    ) -> dict[str, Any]:
        """Prepare/autotune one V5-fast topology without exposing internals."""
        syndrome_list = [float(x) for x in syndrome]
        if self.contractor_config.contractor_name != "cutensornet":
            raise RuntimeError("prepare_fast_logical_mass requires cutensornet")
        fixed, open_list, marginalized = self._validate_logical_request(
            fixed_logicals, open_logicals
        )
        syndrome_key = tuple(syndrome_list)
        if (
            self._fast_cache_syndrome_key is not None
            and syndrome_key != self._fast_cache_syndrome_key
        ):
            self._invalidate_constant_fast_networks()
        self._fast_cache_syndrome_key = syndrome_key
        self.flip_syndromes(syndrome_list)
        tn, output_inds, num_base_operands = self._prepare_projector_mass_network(
            fixed, open_list, marginalized
        )
        entry, created = self._get_or_create_fast_network(
            tn,
            output_inds,
            open_list,
            num_base_operands,
            optimize=optimize,
            autotune_iterations=autotune_iterations,
            memory_limit=memory_limit,
            use_constant_qualifiers=use_constant_qualifiers,
        )
        info = entry.optimizer_info
        return {
            "created": created,
            "open_logicals": tuple(open_list),
            "autotuned": entry.autotuned,
            "uses_constant_qualifiers": entry.uses_qualifiers,
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
            rows.append({
                "key": key,
                "open_logicals": entry.open_logicals,
                "hits": entry.hits,
                "autotuned": entry.autotuned,
                "uses_constant_qualifiers": entry.uses_qualifiers,
                "path_source": entry.path_source,
                "largest_intermediate": getattr(info, "largest_intermediate", None),
                "opt_cost": getattr(info, "opt_cost", None),
                "num_slices": getattr(info, "num_slices", None),
            })
        return rows

    @staticmethod
    def normalized(mass: Any) -> np.ndarray:
        """Convert a mass tensor to a NumPy array and normalize its total to 1."""
        # CUDA-QX currently uses torch on CPU and numpy input with cuTensorNet on
        # GPU. Handle the CPU torch case without making torch a hard dependency.
        if hasattr(mass, "detach"):
            mass = mass.detach()
        if hasattr(mass, "cpu"):
            mass = mass.cpu()
        arr = np.asarray(mass, dtype=np.float64)
        total = float(arr.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"cannot normalize mass tensor with total={total}")
        return arr / total
