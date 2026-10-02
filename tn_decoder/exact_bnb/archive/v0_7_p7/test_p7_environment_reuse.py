import numpy as np
import opt_einsum as oe

from p7_environment_reuse import MemoizedPairwiseContraction


def test_memoized_pairwise_matches_exact_and_reuses():
    # base A plus three variable rank-1 boundary vectors.
    # The path is chosen so one intermediate depends only on variables 0 and 1;
    # a second query that keeps those bits but flips variable 2 should reuse it.
    equation = "abc,a,b,c->"
    A = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 1.0
    shapes = [A.shape, (2,), (2,), (2,)]
    path, _ = oe.contract_path(
        equation, *shapes, shapes=True, optimize=[(1, 2), (0, 2), (0, 1)]
    )

    exe = MemoizedPairwiseContraction(
        equation=equation,
        operand_shapes=shapes,
        path=path,
        num_base_operands=1,
        num_variable_operands=3,
        backend="numpy",
        cache_budget="16MB",
        max_entry_bytes="16MB",
        max_dependency_bits=2,
    )

    p0 = np.array([1.0, 0.0])
    p1 = np.array([0.0, 1.0])

    vars_a = [p0, p1, p0]   # assignment 010
    code_a = 0b010
    out_a = exe.contract(base_operands=[A], variable_operands=vars_a, assignment_code=code_a)
    ref_a = oe.contract(equation, A, *vars_a)
    assert np.allclose(out_a, ref_a)

    hits_before = exe.stats.cache_hits
    vars_b = [p0, p1, p1]   # assignment 110: first two variable bits unchanged
    code_b = 0b110
    out_b = exe.contract(base_operands=[A], variable_operands=vars_b, assignment_code=code_b)
    ref_b = oe.contract(equation, A, *vars_b)
    assert np.allclose(out_b, ref_b)
    assert exe.stats.cache_hits > hits_before


def test_cache_is_assignment_sensitive():
    equation = "ab,a,b->"
    A = np.array([[1.0, 2.0], [3.0, 5.0]])
    shapes = [A.shape, (2,), (2,)]
    path, _ = oe.contract_path(
        equation, *shapes, shapes=True, optimize=[(0, 1), (0, 1)]
    )
    exe = MemoizedPairwiseContraction(
        equation=equation,
        operand_shapes=shapes,
        path=path,
        num_base_operands=1,
        num_variable_operands=2,
        backend="numpy",
        cache_budget="1MB",
        max_entry_bytes="1MB",
        max_dependency_bits=1,
    )
    p0 = np.array([1.0, 0.0])
    p1 = np.array([0.0, 1.0])
    r00 = exe.contract(base_operands=[A], variable_operands=[p0, p0], assignment_code=0b00)
    r10 = exe.contract(base_operands=[A], variable_operands=[p1, p0], assignment_code=0b01)
    assert np.allclose(r00, A[0, 0])
    assert np.allclose(r10, A[1, 0])
    assert not np.allclose(r00, r10)
