from p5_utils import canonical_topology, canonical_block_signature


def test_canonical_topology_ignores_within_block_order():
    assert canonical_topology((6, 7, 8, 9, 10, 11)) == (6, 7, 8, 9, 10, 11)
    assert canonical_topology((11, 10, 9, 8, 7, 6)) == (6, 7, 8, 9, 10, 11)


def test_signature_preserves_block_order():
    identity = tuple(range(12))
    reverse = tuple(reversed(range(12)))
    assert canonical_block_signature(identity, 6) == (
        (0, 1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10, 11),
    )
    assert canonical_block_signature(reverse, 6) == (
        (6, 7, 8, 9, 10, 11),
        (0, 1, 2, 3, 4, 5),
    )
