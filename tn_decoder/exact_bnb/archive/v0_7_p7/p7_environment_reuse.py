from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np
import opt_einsum as oe


@dataclass
class P7Stats:
    queries: int = 0
    pair_contractions_executed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_insertions: int = 0
    cache_evictions: int = 0
    cache_bytes: int = 0
    cache_peak_bytes: int = 0
    bytes_reused: int = 0
    max_dependency_bits_seen: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "queries": int(self.queries),
            "pair_contractions_executed": int(self.pair_contractions_executed),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_insertions": int(self.cache_insertions),
            "cache_evictions": int(self.cache_evictions),
            "cache_bytes": int(self.cache_bytes),
            "cache_peak_bytes": int(self.cache_peak_bytes),
            "bytes_reused": int(self.bytes_reused),
            "max_dependency_bits_seen": int(self.max_dependency_bits_seen),
        }


def _parse_bytes(value: int | float | str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().lower().replace(" ", "")
    units = {
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    for suffix, scale in units.items():
        if text.endswith(suffix):
            return int(float(text[:-len(suffix)]) * scale)
    return int(float(text))


class MemoizedPairwiseContraction:
    """Execute a fixed pairwise einsum path with assignment-aware memoization.

    Initial operands are split into:
      * base operands: dependency mask 0 inside one syndrome/session
      * logical-boundary operands: one dependency bit per fixed logical

    Each intermediate inherits the OR of its input dependency masks.  The
    cache key is therefore:

        (contraction_step, assignment restricted to dependency_mask)

    If two logical prefixes agree on every bit that a cached intermediate
    depends on, that exact GPU/CPU intermediate can be reused instead of being
    recomputed.  This is P7's shared-environment/subexpression reuse.

    The cache is deliberately per decoding session (one syndrome), so syndrome
    dependent intermediates are never reused across different shots.
    """

    def __init__(
        self,
        *,
        equation: str,
        operand_shapes: Sequence[Sequence[int]],
        path: Sequence[Sequence[int]],
        num_base_operands: int,
        num_variable_operands: int,
        backend: str,
        cache_budget: int | str = "8GB",
        max_entry_bytes: int | str = "2GB",
        max_dependency_bits: int = 6,
    ) -> None:
        self.equation = str(equation)
        self.operand_shapes = [tuple(int(x) for x in s) for s in operand_shapes]
        self.path = [tuple(int(x) for x in step) for step in path]
        self.num_base_operands = int(num_base_operands)
        self.num_variable_operands = int(num_variable_operands)
        self.backend = str(backend)
        self.cache_budget = _parse_bytes(cache_budget)
        self.max_entry_bytes = _parse_bytes(max_entry_bytes)
        self.max_dependency_bits = int(max_dependency_bits)

        if self.num_base_operands + self.num_variable_operands != len(self.operand_shapes):
            raise ValueError("base + variable operand counts do not match operand_shapes")
        if self.max_dependency_bits < 0:
            raise ValueError("max_dependency_bits must be >= 0")

        _, info = oe.contract_path(
            self.equation,
            *self.operand_shapes,
            shapes=True,
            optimize=self.path,
        )
        self.contraction_list = list(info.contraction_list)

        self.initial_masks = [0] * self.num_base_operands + [
            1 << i for i in range(self.num_variable_operands)
        ]
        self.full_mask = (1 << self.num_variable_operands) - 1
        self.cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self.stats = P7Stats()

    @staticmethod
    def _nbytes(x: Any) -> int:
        n = getattr(x, "nbytes", None)
        if n is not None:
            return int(n)
        return int(np.asarray(x).nbytes)

    def clear(self) -> None:
        self.cache.clear()
        self.stats.cache_bytes = 0

    def _evict_to_fit(self, needed: int) -> None:
        while self.cache and self.stats.cache_bytes + needed > self.cache_budget:
            _, arr = self.cache.popitem(last=False)
            self.stats.cache_bytes -= self._nbytes(arr)
            self.stats.cache_evictions += 1

    def _cache_get(self, key: tuple[int, int]) -> Any | None:
        arr = self.cache.pop(key, None)
        if arr is None:
            return None
        self.cache[key] = arr
        self.stats.cache_hits += 1
        self.stats.bytes_reused += self._nbytes(arr)
        return arr

    def _cache_put(self, key: tuple[int, int], arr: Any, dep_mask: int) -> None:
        dep_bits = int(dep_mask).bit_count()
        self.stats.max_dependency_bits_seen = max(
            self.stats.max_dependency_bits_seen, dep_bits
        )
        # Full-assignment intermediates cannot help a different prefix.
        if dep_mask == self.full_mask:
            return
        if dep_bits > self.max_dependency_bits:
            return
        nbytes = self._nbytes(arr)
        if nbytes > self.max_entry_bytes or nbytes > self.cache_budget:
            return
        old = self.cache.pop(key, None)
        if old is not None:
            self.stats.cache_bytes -= self._nbytes(old)
        self._evict_to_fit(nbytes)
        self.cache[key] = arr
        self.stats.cache_bytes += nbytes
        self.stats.cache_peak_bytes = max(
            self.stats.cache_peak_bytes, self.stats.cache_bytes
        )
        self.stats.cache_insertions += 1

    def contract(
        self,
        *,
        base_operands: Sequence[Any],
        variable_operands: Sequence[Any],
        assignment_code: int,
    ) -> Any:
        if len(base_operands) != self.num_base_operands:
            raise ValueError("wrong number of base_operands")
        if len(variable_operands) != self.num_variable_operands:
            raise ValueError("wrong number of variable_operands")

        operands = list(base_operands) + list(variable_operands)
        masks = list(self.initial_masks)
        self.stats.queries += 1

        for step_id, contraction in enumerate(self.contraction_list):
            inds, _idx_rm, einsum_str, _remaining, _blas = contraction
            # opt_einsum provides a pop-safe ordering (normally descending).
            tmp_operands = [operands.pop(int(i)) for i in inds]
            tmp_masks = [masks.pop(int(i)) for i in inds]
            dep_mask = 0
            for m in tmp_masks:
                dep_mask |= int(m)

            signature = int(assignment_code) & dep_mask
            key = (int(step_id), int(signature))
            dep_bits = int(dep_mask).bit_count()
            self.stats.max_dependency_bits_seen = max(
                self.stats.max_dependency_bits_seen, dep_bits
            )

            cacheable = (
                dep_mask != self.full_mask
                and dep_bits <= self.max_dependency_bits
            )
            result = self._cache_get(key) if cacheable else None
            if result is None:
                if cacheable:
                    self.stats.cache_misses += 1
                result = oe.contract(
                    einsum_str,
                    *tmp_operands,
                    optimize=False,
                    backend=self.backend,
                )
                self.stats.pair_contractions_executed += 1
                if cacheable:
                    self._cache_put(key, result, dep_mask)

            operands.append(result)
            masks.append(dep_mask)

        if len(operands) != 1:
            raise RuntimeError(f"pairwise executor ended with {len(operands)} operands")
        return operands[0]


class LogicalTailEnvironmentSession:
    """One-syndrome P7 session for repeated BB72 tail contractions."""

    def __init__(self, engine: "LogicalTailEnvironmentEngine", syndrome: Sequence[float]):
        self.engine = engine
        self.decoder = engine.decoder
        self.syndrome = tuple(float(x) for x in syndrome)

        dec = self.decoder
        dec.flip_syndromes(list(self.syndrome))
        # P3 shared GPU buffers are reused, but the P7 memoized intermediates
        # below are strictly local to this syndrome/session.
        base_gpu = dec._ensure_gpu_base_operands(self.syndrome)
        dec._sync_gpu_syndrome_operands(self.syndrome)
        self.base_operands = list(base_gpu)

        import cupy as cp
        dtype = np.dtype(dec._dtype)
        with cp.cuda.Device(int(dec.contractor_config.device_id)):
            self.proj0 = cp.asarray(np.asarray([1.0, 0.0], dtype=dtype))
            self.proj1 = cp.asarray(np.asarray([0.0, 1.0], dtype=dtype))

        self.executor = MemoizedPairwiseContraction(
            equation=engine.equation,
            operand_shapes=engine.operand_shapes,
            path=engine.path,
            num_base_operands=engine.num_base_operands,
            num_variable_operands=len(engine.boundary_logicals),
            backend="cupy",
            cache_budget=engine.cache_budget,
            max_entry_bytes=engine.max_entry_bytes,
            max_dependency_bits=engine.max_dependency_bits,
        )

    @property
    def stats(self) -> P7Stats:
        return self.executor.stats

    def contract(self, fixed_logicals: Mapping[int, int]) -> Any:
        fixed = {int(k): int(v) for k, v in fixed_logicals.items()}
        expected = tuple(self.engine.boundary_logicals)
        if tuple(sorted(fixed)) != expected:
            raise ValueError(
                "P7 tail session requires exactly the fixed logical set "
                f"{expected}, got {tuple(sorted(fixed))}"
            )

        variable_operands = []
        assignment_code = 0
        for pos, logical in enumerate(self.engine.boundary_logicals):
            bit = int(fixed[logical])
            if bit not in (0, 1):
                raise ValueError(f"logical {logical} has invalid bit {bit}")
            if bit:
                assignment_code |= 1 << pos
                variable_operands.append(self.proj1)
            else:
                variable_operands.append(self.proj0)

        return self.executor.contract(
            base_operands=self.base_operands,
            variable_operands=variable_operands,
            assignment_code=assignment_code,
        )


class LogicalTailEnvironmentEngine:
    """Structural P7 plan shared across decoding sessions.

    v0.7 intentionally targets the validated BB72 b9 tail topology:
        fixed logicals 0..8, open logicals 9..11.

    The P5 compiled path is reused.  The current implementation requires that
    this topology has no cuTensorNet slicing; the validated b9 tail profile has
    zero slices.  Root [0..8] contractions remain on the P0-P5 cuTensorNet
    fast path.
    """

    def __init__(
        self,
        decoder: Any,
        *,
        open_logicals: Sequence[int] = (9, 10, 11),
        cache_budget: int | str = "8GB",
        max_entry_bytes: int | str = "2GB",
        max_dependency_bits: int = 6,
    ) -> None:
        self.decoder = decoder
        self.open_logicals = tuple(int(x) for x in open_logicals)
        self.cache_budget = cache_budget
        self.max_entry_bytes = max_entry_bytes
        self.max_dependency_bits = int(max_dependency_bits)

        dec = decoder
        if dec.contractor_config.contractor_name != "cutensornet":
            raise RuntimeError("P7 BB72 environment engine expects cutensornet")
        if not getattr(dec, "_fast_path_profile", None):
            raise RuntimeError("load the P5 compiled path profile before creating P7")

        profile = dec._fast_path_profile.get(self.open_logicals)
        if profile is None:
            raise KeyError(
                f"compiled profile has no topology {self.open_logicals}"
            )
        slices = tuple(profile.get("slices", ()))
        if slices:
            raise RuntimeError(
                "v0.7 P7 pairwise environment reuse currently supports only "
                "unsliced topologies; the validated BB72 tail should have zero slices"
            )
        self.path = [tuple(int(v) for v in pair) for pair in profile["path"]]

        open_set = set(self.open_logicals)
        self.boundary_logicals = tuple(
            i for i in range(dec.num_logicals) if i not in open_set
        )
        fixed0 = {i: 0 for i in self.boundary_logicals}
        fixed, open_list, marginalized = dec._validate_logical_request(
            fixed0, self.open_logicals
        )
        if marginalized:
            raise RuntimeError(
                "P7 tail engine expects every non-open logical to be fixed"
            )

        # Equation and operand shapes are topology-only.  Use the current
        # syndrome values merely to instantiate the Quimb view.
        tn, output_inds, num_base = dec._prepare_projector_mass_network(
            fixed, open_list, marginalized
        )
        self.equation = tn.get_equation(output_inds=output_inds)
        self.operand_shapes = [tuple(int(x) for x in a.shape) for a in tn.arrays]
        self.num_base_operands = int(num_base)
        nvar = len(self.operand_shapes) - self.num_base_operands
        if nvar != len(self.boundary_logicals):
            raise RuntimeError(
                f"expected {len(self.boundary_logicals)} boundary operands, got {nvar}"
            )

        # Validate the inherited path immediately without doing any GPU work.
        _, info = oe.contract_path(
            self.equation,
            *self.operand_shapes,
            shapes=True,
            optimize=self.path,
        )
        self.num_pair_steps = len(info.contraction_list)

    def new_session(self, syndrome: Sequence[float]) -> LogicalTailEnvironmentSession:
        return LogicalTailEnvironmentSession(self, syndrome)

    def info(self) -> dict[str, Any]:
        return {
            "open_logicals": list(self.open_logicals),
            "boundary_logicals": list(self.boundary_logicals),
            "num_base_operands": int(self.num_base_operands),
            "num_variable_operands": int(len(self.boundary_logicals)),
            "num_pair_steps": int(self.num_pair_steps),
            "cache_budget_bytes": int(_parse_bytes(self.cache_budget)),
            "max_entry_bytes": int(_parse_bytes(self.max_entry_bytes)),
            "max_dependency_bits": int(self.max_dependency_bits),
            "path_source": "P5 compiled b9 tail path",
            "slices": 0,
        }
