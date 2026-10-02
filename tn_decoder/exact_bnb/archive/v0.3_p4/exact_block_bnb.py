from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, Any
import math
import time
import numpy as np


Oracle = Callable[[Mapping[int, int], Sequence[int]], Any]


@dataclass
class BnBStats:
    oracle_calls: int = 0
    expanded_internal_nodes: int = 0
    leaves_evaluated: int = 0
    best_updates: int = 0
    pruned_subtrees: int = 0
    pruned_leaves: int = 0
    oracle_seconds: float = 0.0
    wall_seconds: float = 0.0
    clipped_negative_entries: int = 0
    most_negative_raw_mass: float = 0.0
    max_local_guard_abs: float = 0.0


@dataclass
class BnBResult:
    best_lambda: tuple[int, ...]
    best_mass: float
    root_mass: float
    best_probability: float
    stats: BnBStats
    tree_lines: list[str] = field(default_factory=list)


class ExactBlockBnB:
    """Exact block branch-and-bound over binary logical sectors.

    The oracle must return exact subtree masses for a partial assignment:

        oracle(fixed_logicals, open_logicals) -> shape (2,)*len(open_logicals)

    All logical bits that are neither fixed nor open are assumed marginalized.

    For a partial assignment A, the returned child mass W(A') obeys

        max_{lambda extends A'} Z_lambda <= W(A'),

    because W(A') is the sum of all non-negative leaf masses in that subtree.
    Therefore a subtree may be pruned once W(A') cannot beat the incumbent leaf.

    Floating-point contractions are handled conservatively: pruning uses

        W(A') + numerical_guard <= incumbent,

    so a near tie is explored instead of risking a false prune.
    """

    def __init__(
        self,
        *,
        k: int,
        block_size: int,
        oracle: Oracle,
        logical_order: Sequence[int] | None = None,
        guard_rtol: float = 1e-12,
        guard_atol: float = 0.0,
        negative_guard_factor: float = 16.0,
        negative_abort_rtol: float = 1e-6,
        print_tree: bool = True,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.k = int(k)
        self.block_size = int(block_size)
        self.oracle = oracle
        self.order = tuple(range(k) if logical_order is None else logical_order)
        if len(self.order) != k or set(self.order) != set(range(k)):
            raise ValueError("logical_order must be a permutation of range(k)")
        self.guard_rtol = float(guard_rtol)
        self.guard_atol = float(guard_atol)
        self.negative_guard_factor = float(negative_guard_factor)
        self.negative_abort_rtol = float(negative_abort_rtol)
        self.print_tree = bool(print_tree)

        self.best_mass = -math.inf
        self.best_lambda: tuple[int, ...] | None = None
        self.root_mass: float | None = None
        self.guard_abs: float = 0.0
        self.stats = BnBStats()
        self.tree_lines: list[str] = []

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "get"):
            try:
                x = x.get()
            except Exception:
                pass
        return np.asarray(x, dtype=np.float64)

    def _call_oracle(
        self,
        fixed: Mapping[int, int],
        open_logicals: Sequence[int],
    ) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.oracle(fixed, open_logicals)
        self.stats.oracle_seconds += time.perf_counter() - t0
        self.stats.oracle_calls += 1

        arr = self._to_numpy(out)
        expected_shape = (2,) * len(open_logicals)
        if arr.shape != expected_shape:
            raise ValueError(
                f"oracle returned shape {arr.shape}, expected {expected_shape} "
                f"for open_logicals={list(open_logicals)}"
            )
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError("oracle returned NaN/Inf subtree masses")
        return arr

    def _sanitize(self, arr: np.ndarray) -> tuple[np.ndarray, float]:
        """Clip tiny negative cancellation residuals conservatively.

        Physical subtree masses are non-negative.  A negative contraction
        result therefore directly reveals a numerical cancellation scale for
        that oracle call.  We convert the largest observed negative magnitude
        into a *local* pruning guard rather than repeatedly tuning one global
        threshold by hand.

        Returns
        -------
        sanitized, local_guard_abs
            Negative entries are clipped to zero, while local_guard_abs is used
            in every pruning decision for this set of children.
        """
        arr = np.asarray(arr, dtype=np.float64).copy()

        min_raw = float(arr.min(initial=0.0))
        neg_scale = max(0.0, -min_raw)

        abort_tol = max(
            self.guard_atol,
            self.negative_abort_rtol * abs(self.root_mass or 0.0),
        )
        if neg_scale > abort_tol:
            idx = np.unravel_index(int(np.argmin(arr)), arr.shape)
            raise FloatingPointError(
                f"negative subtree mass {arr[idx]} at {idx} exceeds "
                f"abort tolerance {abort_tol}; root mass={self.root_mass}"
            )

        local_guard = max(
            self.guard_abs,
            self.negative_guard_factor * neg_scale,
        )
        self.stats.max_local_guard_abs = max(
            self.stats.max_local_guard_abs,
            local_guard,
        )

        neg_mask = arr < 0.0
        if np.any(neg_mask):
            vals = arr[neg_mask]
            self.stats.clipped_negative_entries += int(vals.size)
            self.stats.most_negative_raw_mass = min(
                self.stats.most_negative_raw_mass,
                float(vals.min()),
            )
            arr[neg_mask] = 0.0

        return arr, local_guard

    def _prefix_string(self, fixed: Mapping[int, int]) -> str:
        # Print in the chosen search order, only for currently fixed bits.
        chars = []
        for q in self.order:
            if q in fixed:
                chars.append(str(int(fixed[q])))
            else:
                break
        return "".join(chars)

    def _full_lambda_tuple(self, fixed: Mapping[int, int]) -> tuple[int, ...]:
        return tuple(int(fixed[i]) for i in range(self.k))

    def _remaining_leaf_count(self, n_fixed: int) -> int:
        return 1 << (self.k - n_fixed)

    def _log(self, depth: int, text: str) -> None:
        if self.print_tree:
            self.tree_lines.append("  " * depth + text)

    def _update_incumbent(self, fixed: Mapping[int, int], mass: float, depth: int) -> None:
        self.stats.leaves_evaluated += 1
        lam = self._full_lambda_tuple(fixed)
        if mass > self.best_mass:
            self.best_mass = float(mass)
            self.best_lambda = lam
            self.stats.best_updates += 1
            p = mass / self.root_mass if self.root_mass else float("nan")
            self._log(
                depth,
                f"leaf {''.join(map(str, lam))}  P={p:.12g}  UPDATE B",
            )
        else:
            p = mass / self.root_mass if self.root_mass else float("nan")
            self._log(depth, f"leaf {''.join(map(str, lam))}  P={p:.12g}")

    def _dfs(self, fixed: dict[int, int], depth: int) -> None:
        n_fixed = len(fixed)
        if n_fixed == self.k:
            raise RuntimeError("internal error: complete leaf should be handled by parent")

        block = list(self.order[n_fixed:min(n_fixed + self.block_size, self.k)])
        arr = self._call_oracle(fixed, block)
        self.stats.expanded_internal_nodes += 1

        # For the root, establish the total probability mass and numerical guard
        # before checking tiny negative cancellation residuals.
        if self.root_mass is None:
            root_total = float(arr.sum())
            if not np.isfinite(root_total) or root_total <= 0.0:
                raise FloatingPointError(f"invalid root mass {root_total}")
            self.root_mass = root_total
            self.guard_abs = max(
                self.guard_atol,
                self.guard_rtol * abs(root_total),
            )

        arr, local_guard_abs = self._sanitize(arr)

        children: list[tuple[float, tuple[int, ...]]] = []
        for idx in np.ndindex(arr.shape):
            children.append((float(arr[idx]), tuple(int(v) for v in idx)))
        children.sort(key=lambda x: x[0], reverse=True)

        child_n_fixed = n_fixed + len(block)
        leaves_per_child = self._remaining_leaf_count(child_n_fixed)

        for rank, (bound, bits) in enumerate(children, start=1):
            child_fixed = dict(fixed)
            for q, bit in zip(block, bits):
                child_fixed[q] = bit

            prefix = self._prefix_string(child_fixed)
            p_bound = bound / self.root_mass

            # Conservative exact pruning. Because children are sorted descending,
            # once this child is prunable, every remaining sibling is too.
            if self.best_lambda is not None and bound + local_guard_abs <= self.best_mass:
                remaining = len(children) - rank + 1
                self.stats.pruned_subtrees += remaining
                self.stats.pruned_leaves += remaining * leaves_per_child

                self._log(
                    depth,
                    f"{prefix}  W={p_bound:.12g}  PRUNE "
                    f"(B={self.best_mass / self.root_mass:.12g})",
                )
                # We already know all remaining masses from this one oracle call;
                # print them too so the search tree is explicit.
                for later_bound, later_bits in children[rank:]:
                    later_fixed = dict(fixed)
                    for q, bit in zip(block, later_bits):
                        later_fixed[q] = bit
                    later_prefix = self._prefix_string(later_fixed)
                    self._log(
                        depth,
                        f"{later_prefix}  W={later_bound / self.root_mass:.12g}  PRUNE",
                    )
                break

            if child_n_fixed == self.k:
                self._update_incumbent(child_fixed, bound, depth)
            else:
                self._log(depth, f"{prefix}  W={p_bound:.12g}  EXPLORE")
                self._dfs(child_fixed, depth + 1)

    def search(self) -> BnBResult:
        t0 = time.perf_counter()
        self._log(0, "root")
        self._dfs({}, 1)
        self.stats.wall_seconds = time.perf_counter() - t0

        if self.best_lambda is None or self.root_mass is None:
            raise RuntimeError("search finished without a leaf")

        return BnBResult(
            best_lambda=self.best_lambda,
            best_mass=self.best_mass,
            root_mass=self.root_mass,
            best_probability=self.best_mass / self.root_mass,
            stats=self.stats,
            tree_lines=self.tree_lines,
        )
