import numpy as np

from exact_block_bnb import ExactBlockBnB


def _fake_oracle_factory(root, tails):
    calls = {"single": 0, "batch": 0, "batch_sizes": []}

    def oracle(fixed, open_logicals):
        calls["single"] += 1
        if not fixed:
            return np.asarray(root, dtype=np.float64)
        key = tuple(int(fixed[i]) for i in (0, 1))
        return np.asarray(tails[key], dtype=np.float64)

    def batch_oracle(fixed_batch, open_logicals):
        calls["batch"] += 1
        calls["batch_sizes"].append(len(fixed_batch))
        rows = []
        for fixed in fixed_batch:
            key = tuple(int(fixed[i]) for i in (0, 1))
            rows.append(np.asarray(tails[key], dtype=np.float64))
        return np.stack(rows, axis=0)

    return oracle, batch_oracle, calls


def test_p6_easy_bootstrap_avoids_batch():
    root = [[0.90, 0.05], [0.03, 0.02]]
    tails = {
        (0, 0): [[0.85, 0.02], [0.02, 0.01]],
        (0, 1): [[0.03, 0.01], [0.005, 0.005]],
        (1, 0): [[0.02, 0.005], [0.003, 0.002]],
        (1, 1): [[0.01, 0.005], [0.003, 0.002]],
    }
    oracle, batch_oracle, calls = _fake_oracle_factory(root, tails)
    res = ExactBlockBnB(
        k=4,
        block_size=2,
        oracle=oracle,
        batch_oracle=batch_oracle,
        parent_batch_size=4,
        print_tree=False,
    ).search()
    assert res.best_lambda == (0, 0, 0, 0)
    assert np.isclose(res.best_mass, 0.85)
    assert calls["batch"] == 0
    assert res.stats.oracle_calls == 2  # root + bootstrap tail


def test_p6_hard_batches_surviving_final_parents():
    root = [[0.40, 0.35], [0.25, 0.00]]
    tails = {
        (0, 0): [[0.15, 0.10], [0.08, 0.07]],  # sum .40
        (0, 1): [[0.20, 0.08], [0.04, 0.03]],  # sum .35
        (1, 0): [[0.25, 0.00], [0.00, 0.00]],  # sum .25
        (1, 1): [[0.00, 0.00], [0.00, 0.00]],
    }
    oracle_s, _batch_s, calls_s = _fake_oracle_factory(root, tails)
    seq = ExactBlockBnB(
        k=4,
        block_size=2,
        oracle=oracle_s,
        print_tree=False,
    ).search()

    oracle_b, batch_b, calls_b = _fake_oracle_factory(root, tails)
    bat = ExactBlockBnB(
        k=4,
        block_size=2,
        oracle=oracle_b,
        batch_oracle=batch_b,
        parent_batch_size=2,
        print_tree=False,
    ).search()

    assert bat.best_lambda == seq.best_lambda == (1, 0, 0, 0)
    assert np.isclose(bat.best_mass, seq.best_mass)
    assert calls_s["single"] == 4  # root + 3 final parents
    assert calls_b["single"] == 2  # root + bootstrap first parent
    assert calls_b["batch"] == 1
    assert calls_b["batch_sizes"] == [2]
    assert bat.stats.oracle_calls == 3
    assert bat.stats.batch_oracle_calls == 1
    assert bat.stats.batched_parent_queries == 2
    assert bat.stats.max_parent_batch_size == 2
