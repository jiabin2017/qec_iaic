from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import re
import numpy as np


@dataclass(frozen=True)
class P5Config:
    block_size: int
    logical_order: tuple[int, ...]

    @property
    def topologies(self) -> tuple[tuple[int, ...], ...]:
        return block_topologies(self.logical_order, self.block_size)


def block_topologies(logical_order: Sequence[int], block_size: int) -> tuple[tuple[int, ...], ...]:
    order = tuple(int(i) for i in logical_order)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not order:
        raise ValueError("logical_order must be non-empty")
    return tuple(
        tuple(order[i : i + block_size])
        for i in range(0, len(order), block_size)
    )



def canonical_topology(open_logicals: Sequence[int]) -> tuple[int, ...]:
    """Canonical key for a block topology, ignoring within-block output order.

    Two open-logical tuples that contain the same logical indices, e.g.
    ``(6,7,8,9,10,11)`` and ``(11,10,9,8,7,6)``, have the same tensor
    connectivity and operand shapes; only the order of the output axes differs.
    A cuTensorNet contraction path is therefore shared under the sorted key.
    """
    vals = tuple(int(v) for v in open_logicals)
    if len(vals) != len(set(vals)):
        raise ValueError(f"open_logicals contains duplicates: {vals}")
    return tuple(sorted(vals))


def canonical_block_signature(
    logical_order: Sequence[int],
    block_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Ordered sequence of canonical block sets for one BnB configuration.

    Within-block permutations are removed, while block order is preserved because
    visiting block A before B can change BnB latency even when the same two sets
    are used.
    """
    return tuple(
        canonical_topology(block)
        for block in block_topologies(logical_order, block_size)
    )

def validate_logical_order(order: Sequence[int], k: int) -> tuple[int, ...]:
    out = tuple(int(i) for i in order)
    if len(out) != int(k) or set(out) != set(range(int(k))):
        raise ValueError(f"logical order must be a permutation of 0..{int(k)-1}; got {out}")
    return out


def _support_weights(logical_obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(logical_obs, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("logical_obs must be 2D")
    return arr.sum(axis=1).astype(np.int64)


def _overlap_matrix(logical_obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(logical_obs, dtype=np.uint8)
    return (arr @ arr.T).astype(np.int64)


def _greedy_overlap_order(logical_obs: np.ndarray, block_size: int, *, maximize: bool) -> tuple[int, ...]:
    """Build block-aware logical groups using physical-support overlap.

    This is an offline topology heuristic only.  It does not look at a syndrome
    and therefore does not add online decoder work or bias the exact BnB result.
    Within each block, start from the largest-support remaining logical row and
    then add rows with maximal (or minimal) total overlap to the rows already in
    that block.  Ties are deterministic.
    """
    arr = np.asarray(logical_obs, dtype=np.uint8)
    k = int(arr.shape[0])
    weights = _support_weights(arr)
    overlap = _overlap_matrix(arr)
    remaining = set(range(k))
    order: list[int] = []

    while remaining:
        seed = min(remaining, key=lambda i: (-int(weights[i]), int(i)))
        block = [seed]
        remaining.remove(seed)

        while remaining and len(block) < int(block_size):
            def key(i: int):
                score = int(sum(int(overlap[i, j]) for j in block))
                if maximize:
                    return (-score, -int(weights[i]), int(i))
                return (score, -int(weights[i]), int(i))

            nxt = min(remaining, key=key)
            block.append(nxt)
            remaining.remove(nxt)

        order.extend(block)

    return tuple(order)


def named_logical_order(
    name: str,
    *,
    k: int,
    logical_obs: np.ndarray | None = None,
    block_size: int = 1,
    seed: int = 0,
) -> tuple[int, ...]:
    """Resolve a deterministic P5 logical-order heuristic."""
    key = str(name).strip().lower().replace("_", "-")
    if key in {"identity", "natural", "default"}:
        return tuple(range(k))
    if key in {"reverse", "reversed"}:
        return tuple(reversed(range(k)))
    if key in {"random", "shuffle"}:
        rng = np.random.default_rng(int(seed))
        return tuple(int(v) for v in rng.permutation(k))

    if logical_obs is None:
        raise ValueError(f"order strategy '{name}' requires logical_obs")
    arr = np.asarray(logical_obs, dtype=np.uint8)
    if arr.shape[0] != k:
        raise ValueError("logical_obs row count does not match k")
    weights = _support_weights(arr)

    if key in {"weight-asc", "support-asc"}:
        return tuple(sorted(range(k), key=lambda i: (int(weights[i]), int(i))))
    if key in {"weight-desc", "support-desc"}:
        return tuple(sorted(range(k), key=lambda i: (-int(weights[i]), int(i))))
    if key in {"overlap-max", "cluster", "cluster-overlap"}:
        return _greedy_overlap_order(arr, block_size, maximize=True)
    if key in {"overlap-min", "spread", "anti-overlap"}:
        return _greedy_overlap_order(arr, block_size, maximize=False)

    raise ValueError(
        f"unknown logical-order strategy '{name}'. Supported names: "
        "identity, reverse, weight-asc, weight-desc, overlap-max, overlap-min, random; "
        "or pass an explicit comma-separated permutation."
    )


def parse_logical_order(
    spec: str | Sequence[int] | None,
    *,
    k: int,
    logical_obs: np.ndarray | None = None,
    block_size: int = 1,
    seed: int = 0,
) -> tuple[int, ...]:
    if spec is None:
        return tuple(range(k))
    if not isinstance(spec, str):
        return validate_logical_order(spec, k)

    text = spec.strip()
    if not text:
        return tuple(range(k))

    # Explicit permutations accept commas, colons, semicolons, or whitespace.
    if re.search(r"\d", text) and any(ch in text for ch in ",:; "):
        toks = [tok for tok in re.split(r"[,;:\s]+", text) if tok]
        try:
            vals = tuple(int(tok) for tok in toks)
        except ValueError:
            vals = tuple()
        if vals:
            return validate_logical_order(vals, k)

    return validate_logical_order(
        named_logical_order(
            text,
            k=k,
            logical_obs=logical_obs,
            block_size=block_size,
            seed=seed,
        ),
        k,
    )


def order_label(order: Sequence[int]) -> str:
    return "-".join(str(int(i)) for i in order)


def parse_block_sizes(spec: str | Iterable[int], *, k: int) -> tuple[int, ...]:
    if isinstance(spec, str):
        toks = [tok for tok in re.split(r"[,;:\s]+", spec.strip()) if tok]
        vals = [int(tok) for tok in toks]
    else:
        vals = [int(v) for v in spec]
    out: list[int] = []
    for b in vals:
        if b < 1 or b > int(k):
            raise ValueError(f"block size must be between 1 and {k}; got {b}")
        if b not in out:
            out.append(b)
    if not out:
        raise ValueError("no block sizes supplied")
    return tuple(out)
