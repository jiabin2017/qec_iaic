from __future__ import annotations

import itertools
import numpy as np

from exact_block_bnb import ExactBlockBnB


def make_oracle(leaf_masses):
    leaf_masses = {
        tuple(int(x) for x in bits): float(m)
        for bits, m in leaf_masses.items()
    }
    k = len(next(iter(leaf_masses)))

    def oracle(fixed_logicals, open_logicals):
        open_logicals = list(open_logicals)
        out = np.zeros((2,) * len(open_logicals), dtype=np.float64)

        for open_bits in itertools.product((0, 1), repeat=len(open_logicals)):
            partial = dict(fixed_logicals)
            partial.update(dict(zip(open_logicals, open_bits)))

            total = 0.0
            for lam, mass in leaf_masses.items():
                if all(lam[q] == bit for q, bit in partial.items()):
                    total += mass
            out[open_bits] = total
        return out

    return oracle


def test_exact_block_bnb_finds_global_ml():
    masses = {
        (0, 0, 0, 0): 0.01,
        (0, 0, 0, 1): 0.02,
        (0, 0, 1, 0): 0.03,
        (0, 0, 1, 1): 0.04,
        (0, 1, 0, 0): 0.05,
        (0, 1, 0, 1): 0.06,
        (0, 1, 1, 0): 0.07,
        (0, 1, 1, 1): 0.08,
        (1, 0, 0, 0): 0.09,
        (1, 0, 0, 1): 0.10,
        (1, 0, 1, 0): 0.11,
        (1, 0, 1, 1): 0.12,
        (1, 1, 0, 0): 0.13,
        (1, 1, 0, 1): 0.14,
        (1, 1, 1, 0): 0.15,
        (1, 1, 1, 1): 0.90,
    }
    result = ExactBlockBnB(
        k=4,
        block_size=2,
        oracle=make_oracle(masses),
        logical_order=[0, 1, 2, 3],
        print_tree=False,
    ).search()

    assert result.best_lambda == (1, 1, 1, 1)
    assert np.isclose(result.best_mass, 0.90)
    assert np.isclose(result.root_mass, sum(masses.values()))
    assert np.isclose(result.best_probability, 0.90 / sum(masses.values()))


def test_exact_block_bnb_accepts_degenerate_ml():
    masses = {
        bits: 0.01 for bits in itertools.product((0, 1), repeat=3)
    }
    masses[(0, 1, 1)] = 0.5
    masses[(1, 0, 0)] = 0.5

    result = ExactBlockBnB(
        k=3,
        block_size=2,
        oracle=make_oracle(masses),
        print_tree=False,
    ).search()

    assert result.best_lambda in {(0, 1, 1), (1, 0, 0)}
    assert np.isclose(result.best_mass, 0.5)


def test_tiny_negative_oracle_residual_is_guarded():
    masses = {
        bits: 0.01 for bits in itertools.product((0, 1), repeat=3)
    }
    masses[(1, 1, 1)] = 0.8
    base_oracle = make_oracle(masses)

    def oracle(fixed, open_logicals):
        arr = base_oracle(fixed, open_logicals)
        # Introduce a tiny negative residual only where the exact subtree mass
        # would be zero-like for the purpose of exercising sanitization.
        if arr.size > 1:
            arr = arr.copy()
            arr.flat[0] -= 1e-15
        return arr

    result = ExactBlockBnB(
        k=3,
        block_size=2,
        oracle=oracle,
        guard_rtol=1e-12,
        print_tree=False,
    ).search()

    assert result.best_lambda == (1, 1, 1)
