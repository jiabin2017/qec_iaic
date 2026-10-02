import numpy as np
from p5_utils import block_topologies, parse_logical_order, parse_block_sizes


def test_block_topologies_nondivisible():
    assert block_topologies(range(5), 2) == ((0, 1), (2, 3), (4,))


def test_parse_explicit_order():
    assert parse_logical_order("2,0,1", k=3) == (2, 0, 1)


def test_named_orders_are_permutations():
    L = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
    ], dtype=np.uint8)
    for name in ["identity", "reverse", "weight-asc", "weight-desc", "overlap-max", "overlap-min", "random"]:
        order = parse_logical_order(name, k=4, logical_obs=L, block_size=2, seed=7)
        assert sorted(order) == [0, 1, 2, 3]


def test_block_size_parser():
    assert parse_block_sizes("1,2,3,6", k=12) == (1, 2, 3, 6)
