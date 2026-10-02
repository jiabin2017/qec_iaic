from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, Any
import math
import time
import numpy as np


Oracle = Callable[[Mapping[int, int], Sequence[int]], Any]
BatchOracle = Callable[[Sequence[Mapping[int, int]], Sequence[int]], Any]


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
    # P6 diagnostics. oracle_calls counts physical oracle executions, so one
    # batched contraction counts as one call even if it evaluates several
    # logical parent prefixes.
    batch_oracle_calls: int = 0
    batched_parent_queries: int = 0
    max_parent_batch_size: int = 1
    wasted_batched_parent_queries: int = 0


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
        batch_oracle: BatchOracle | None = None,
        parent_batch_size: int = 1,
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
        self.batch_oracle = batch_oracle
        self.parent_batch_size = int(parent_batch_size)
        if self.parent_batch_size < 1:
            raise ValueError("parent_batch_size must be >= 1")
        if self.parent_batch_size > 1 and self.batch_oracle is None:
            raise ValueError(
                "parent_batch_size > 1 requires batch_oracle; "
                "use parent_batch_size=1 for the legacy sequential search"
            )
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

    def _call_batch_oracle(
        self,
        fixed_batch: Sequence[Mapping[int, int]],
        open_logicals: Sequence[int],
    ) -> np.ndarray:
        if self.batch_oracle is None:
            raise RuntimeError("batch oracle is not configured")
        batch = [dict(x) for x in fixed_batch]
        if not batch:
            raise ValueError("fixed_batch must be non-empty")

        t0 = time.perf_counter()
        out = self.batch_oracle(batch, open_logicals)
        self.stats.oracle_seconds += time.perf_counter() - t0
        self.stats.oracle_calls += 1
        self.stats.batch_oracle_calls += 1
        self.stats.batched_parent_queries += len(batch)
        self.stats.max_parent_batch_size = max(
            self.stats.max_parent_batch_size, len(batch)
        )

        arr = self._to_numpy(out)
        expected_shape = (len(batch),) + (2,) * len(open_logicals)
        if arr.shape != expected_shape:
            raise ValueError(
                f"batch oracle returned shape {arr.shape}, expected {expected_shape} "
                f"for batch={len(batch)}, open_logicals={list(open_logicals)}"
            )
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError("batch oracle returned NaN/Inf subtree masses")
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

    def _prune_remaining_children(
        self,
        *,
        children: Sequence[tuple[float, tuple[int, ...]]],
        start_index: int,
        fixed: Mapping[int, int],
        block: Sequence[int],
        depth: int,
        leaves_per_child: int,
    ) -> None:
        """Account/log every sibling from ``start_index`` onward as pruned."""
        remaining = len(children) - int(start_index)
        if remaining <= 0:
            return
        self.stats.pruned_subtrees += remaining
        self.stats.pruned_leaves += remaining * int(leaves_per_child)
        for later_bound, later_bits in children[start_index:]:
            later_fixed = dict(fixed)
            for q, bit in zip(block, later_bits):
                later_fixed[q] = bit
            later_prefix = self._prefix_string(later_fixed)
            self._log(
                depth,
                f"{later_prefix}  W={later_bound / self.root_mass:.12g}  PRUNE",
            )

    def _process_final_parent_children_p6(
        self,
        *,
        fixed: dict[int, int],
        depth: int,
        block: Sequence[int],
        children: Sequence[tuple[float, tuple[int, ...]]],
        parent_guard_abs: float,
        child_n_fixed: int,
        leaves_per_child: int,
    ) -> None:
        """P6 micro-batch final-level parent expansions.

        The first promising parent is deliberately expanded *sequentially* to
        establish an incumbent. This preserves the excellent easy-syndrome
        behavior of v0.5: when that first leaf dominates, all remaining parent
        prefixes are pruned and no batch is launched.

        Once an incumbent exists, up to ``parent_batch_size`` consecutive
        surviving parents (which all share the same next-block topology) are
        evaluated in one batch contraction. If the incumbent improves while
        consuming the batch, later precomputed parents can become unnecessary;
        those are counted as ``wasted_batched_parent_queries`` but do not affect
        exactness.
        """
        i = 0
        next_block = list(
            self.order[child_n_fixed:min(child_n_fixed + self.block_size, self.k)]
        )
        if child_n_fixed + len(next_block) != self.k:
            raise RuntimeError("P6 helper is only valid one block before leaves")

        while i < len(children):
            bound, bits = children[i]

            if self.best_lambda is not None and bound + parent_guard_abs <= self.best_mass:
                self._prune_remaining_children(
                    children=children,
                    start_index=i,
                    fixed=fixed,
                    block=block,
                    depth=depth,
                    leaves_per_child=leaves_per_child,
                )
                return

            child_fixed = dict(fixed)
            for q, bit in zip(block, bits):
                child_fixed[q] = bit
            prefix = self._prefix_string(child_fixed)

            # Bootstrap one parent sequentially before attempting a batch. This
            # is what keeps easy cases at exactly the v0.5 amount of TN work.
            if self.best_lambda is None or self.parent_batch_size <= 1:
                self._log(
                    depth,
                    f"{prefix}  W={bound / self.root_mass:.12g}  EXPLORE",
                )
                self._dfs(child_fixed, depth + 1)
                i += 1
                continue

            batch_rows: list[tuple[float, tuple[int, ...], dict[int, int]]] = []
            j = i
            incumbent_before_batch = self.best_mass
            while j < len(children) and len(batch_rows) < self.parent_batch_size:
                bnd, bts = children[j]
                if bnd + parent_guard_abs <= incumbent_before_batch:
                    break
                cf = dict(fixed)
                for q, bit in zip(block, bts):
                    cf[q] = bit
                batch_rows.append((bnd, bts, cf))
                j += 1

            if len(batch_rows) <= 1:
                self._log(
                    depth,
                    f"{prefix}  W={bound / self.root_mass:.12g}  EXPLORE",
                )
                self._dfs(child_fixed, depth + 1)
                i += 1
                continue

            batch_out = self._call_batch_oracle(
                [row[2] for row in batch_rows], next_block
            )
            # Every row in batch_out required contraction work, even if an
            # earlier row subsequently improves B enough to make a later row
            # logically unnecessary.
            self.stats.expanded_internal_nodes += len(batch_rows)

            stop_all = False
            for bi, (parent_bound, _parent_bits, parent_fixed) in enumerate(batch_rows):
                parent_prefix = self._prefix_string(parent_fixed)
                if (
                    self.best_lambda is not None
                    and parent_bound + parent_guard_abs <= self.best_mass
                ):
                    wasted = len(batch_rows) - bi
                    self.stats.wasted_batched_parent_queries += wasted
                    self._prune_remaining_children(
                        children=children,
                        start_index=i + bi,
                        fixed=fixed,
                        block=block,
                        depth=depth,
                        leaves_per_child=leaves_per_child,
                    )
                    stop_all = True
                    break

                self._log(
                    depth,
                    f"{parent_prefix}  W={parent_bound / self.root_mass:.12g}  "
                    f"EXPLORE_BATCH[{len(batch_rows)}]",
                )

                arr = np.asarray(batch_out[bi], dtype=np.float64)
                arr, local_guard_abs = self._sanitize(arr)
                leaf_children = [
                    (float(arr[idx]), tuple(int(v) for v in idx))
                    for idx in np.ndindex(arr.shape)
                ]
                leaf_children.sort(key=lambda x: x[0], reverse=True)

                for rank, (leaf_mass, leaf_bits) in enumerate(leaf_children, start=1):
                    leaf_fixed = dict(parent_fixed)
                    for q, bit in zip(next_block, leaf_bits):
                        leaf_fixed[q] = bit

                    if (
                        self.best_lambda is not None
                        and leaf_mass + local_guard_abs <= self.best_mass
                    ):
                        remaining = len(leaf_children) - rank + 1
                        self.stats.pruned_subtrees += remaining
                        self.stats.pruned_leaves += remaining
                        leaf_prefix = self._prefix_string(leaf_fixed)
                        self._log(
                            depth + 1,
                            f"{leaf_prefix}  W={leaf_mass / self.root_mass:.12g}  "
                            f"PRUNE (B={self.best_mass / self.root_mass:.12g})",
                        )
                        for later_mass, later_bits in leaf_children[rank:]:
                            later_fixed = dict(parent_fixed)
                            for q, bit in zip(next_block, later_bits):
                                later_fixed[q] = bit
                            self._log(
                                depth + 1,
                                f"{self._prefix_string(later_fixed)}  "
                                f"W={later_mass / self.root_mass:.12g}  PRUNE",
                            )
                        break

                    self._update_incumbent(leaf_fixed, leaf_mass, depth + 1)

            if stop_all:
                return
            i += len(batch_rows)

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

        # P6 targets exactly the repeated-parent situation that dominates the
        # b=9 BB72 hard case: several first-block prefixes survive, and each
        # needs the same final-block topology. Keep the legacy DFS for all
        # other levels so parent_batch_size=1 is bit-for-bit the v0.5 search.
        if (
            self.batch_oracle is not None
            and self.parent_batch_size > 1
            and child_n_fixed < self.k
            and child_n_fixed + self.block_size >= self.k
        ):
            self._process_final_parent_children_p6(
                fixed=fixed,
                depth=depth,
                block=block,
                children=children,
                parent_guard_abs=local_guard_abs,
                child_n_fixed=child_n_fixed,
                leaves_per_child=leaves_per_child,
            )
            return

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
