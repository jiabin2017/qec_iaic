import numpy as np

from exact_block_bnb import ExactBlockBnB


def make_exact_oracle(full):
    full = np.asarray(full, dtype=np.float64)
    k = full.ndim

    def oracle(fixed, open_logicals):
        out = np.zeros((2,) * len(open_logicals), dtype=np.float64)
        for out_idx in np.ndindex(out.shape):
            total = 0.0
            for leaf in np.ndindex(full.shape):
                ok = all(leaf[q] == bit for q, bit in fixed.items())
                if not ok:
                    continue
                ok = all(leaf[q] == bit for q, bit in zip(open_logicals, out_idx))
                if ok:
                    total += full[leaf]
            out[out_idx] = total
        return out

    return oracle


def test_exact_bnb_matches_exhaustive_argmax():
    rng = np.random.default_rng(7)
    full = rng.random((2,) * 9)
    # Create a clear ML leaf but keep a nontrivial distribution.
    winner = (1, 0, 1, 1, 0, 0, 1, 0, 1)
    full[winner] = full.max() + 10.0

    searcher = ExactBlockBnB(
        k=9,
        block_size=3,
        oracle=make_exact_oracle(full),
        guard_rtol=0.0,
        print_tree=False,
    )
    result = searcher.search()
    brute = tuple(np.unravel_index(np.argmax(full), full.shape))

    assert result.best_lambda == brute
    assert np.isclose(result.best_mass, full[brute])
    assert np.isclose(result.root_mass, full.sum())
    assert result.stats.pruned_leaves > 0


def test_exact_bnb_easy_distribution_prunes_heavily():
    full = np.full((2,) * 12, 1e-12, dtype=np.float64)
    full[(0,) * 12] = 1.0

    searcher = ExactBlockBnB(
        k=12,
        block_size=3,
        oracle=make_exact_oracle(full),
        guard_rtol=0.0,
        print_tree=False,
    )
    result = searcher.search()

    assert result.best_lambda == (0,) * 12
    assert result.stats.leaves_evaluated == 1
    assert result.stats.oracle_calls == 4
    assert result.stats.pruned_leaves == 4095


def test_exact_bnb_nontrivial_order_and_partial_last_block():
    rng = np.random.default_rng(11)
    full = rng.random((2,) * 7)
    winner = (1, 1, 0, 1, 0, 1, 0)
    full[winner] = full.max() + 20.0
    order = [6, 2, 0, 5, 1, 4, 3]

    searcher = ExactBlockBnB(
        k=7,
        block_size=3,
        oracle=make_exact_oracle(full),
        logical_order=order,
        guard_rtol=0.0,
        print_tree=False,
    )
    result = searcher.search()
    brute = tuple(np.unravel_index(np.argmax(full), full.shape))

    assert result.best_lambda == brute
    assert np.isclose(result.best_mass, full[brute])
    assert result.stats.oracle_calls >= 3
