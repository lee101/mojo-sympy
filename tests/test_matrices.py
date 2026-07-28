import random

import pytest
import sympy as sp

import mojosympy as ms


def random_matrix(rows, cols, seed):
    rng = random.Random(seed)
    return [[rng.randint(-7, 7) for _ in range(cols)] for _ in range(rows)]


@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (8, 8)])
def test_matrix_add_sub_transpose_parity(shape):
    rows, cols = shape
    a = random_matrix(rows, cols, rows + cols)
    b = random_matrix(rows, cols, rows * cols + 3)
    ours_a, ours_b = ms.Matrix(a), ms.Matrix(b)
    theirs_a, theirs_b = sp.Matrix(a), sp.Matrix(b)
    assert ours_a + ours_b == theirs_a + theirs_b
    assert ours_a - ours_b == theirs_a - theirs_b
    assert ours_a.T == theirs_a.T


@pytest.mark.parametrize("shape", [(2, 3, 4), (7, 5, 9), (16, 16, 16)])
def test_integer_matrix_multiply_parity(shape):
    rows, inner, cols = shape
    a = random_matrix(rows, inner, rows + 10)
    b = random_matrix(inner, cols, cols + 20)
    assert ms.Matrix(a) * ms.Matrix(b) == sp.Matrix(a) * sp.Matrix(b)
    assert ms.Matrix(a) @ ms.Matrix(b) == sp.Matrix(a) @ sp.Matrix(b)


def test_rational_matrix_arithmetic_parity():
    a = [[sp.Rational(i + 1, i + 2) for i in range(4)] for _ in range(3)]
    b = [[sp.Rational(i + j + 1, i + j + 3) for j in range(2)] for i in range(4)]
    ours = ms.Matrix(a) * ms.Matrix(b)
    theirs = sp.Matrix(a) * sp.Matrix(b)
    assert ours == theirs


def test_float_matrix_multiply_parity():
    a = [[sp.Float("1.25"), sp.Float("-2.5")], [sp.Float("0.5"), sp.Float("3.0")]]
    b = [[sp.Float("4.0"), sp.Float("0.25")], [sp.Float("-1.5"), sp.Float("2.0")]]
    ours = ms.Matrix(a) * ms.Matrix(b)
    theirs = sp.Matrix(a) * sp.Matrix(b)
    for actual, expected in zip(ours, theirs):
        assert float(actual) == pytest.approx(float(expected), abs=1e-14)


def test_exact_matrix_overflow_does_not_narrow_to_float():
    huge = 1 << 100
    ours = ms.Matrix([[huge, 1]]) * ms.Matrix([[1], [huge]])
    assert ours == sp.Matrix([[huge, 1]]) * sp.Matrix([[1], [huge]])
    assert ours[0, 0].is_Integer


def test_high_precision_float_matrix_does_not_narrow_to_float64():
    value = sp.Float("1.000000000000000000000000000001", 100)
    ours = ms.Matrix([[value]]) * ms.Matrix([[value]])
    expected = sp.Matrix([[value]]) * sp.Matrix([[value]])
    assert ours == expected
    assert ours[0, 0]._prec == expected[0, 0]._prec


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6])
def test_determinant_parity(size):
    data = random_matrix(size, size, 100 + size)
    assert ms.Matrix(data).det() == sp.Matrix(data).det()


def test_determinant_row_swap_and_singular():
    swap = [[0, 2, 1], [3, 4, 5], [6, 7, 8]]
    singular = [[1, 2, 3], [2, 4, 6], [3, 5, 7]]
    assert ms.Matrix(swap).det() == sp.Matrix(swap).det()
    assert ms.Matrix(singular).det() == 0


def test_rational_determinant_parity():
    data = [
        [sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 5)],
        [sp.Rational(5, 7), sp.Rational(-1, 3), sp.Rational(4, 9)],
        [sp.Rational(2, 11), sp.Rational(7, 8), sp.Rational(1, 6)],
    ]
    assert ms.Matrix(data).det() == sp.Matrix(data).det()


def test_inverse_rref_rank_and_solve_delegate_with_wrapped_results():
    data = [[3, 1, 2], [1, 4, 0], [2, 0, 5]]
    ours = ms.Matrix(data)
    theirs = sp.Matrix(data)
    assert ours.inv() == theirs.inv()
    assert ours.rref() == theirs.rref()
    assert ours.rank() == theirs.rank()
    rhs = ms.Matrix([1, 2, 3])
    assert ours.solve(rhs) == theirs.solve(sp.Matrix([1, 2, 3]))


def test_symbolic_matrix_fallback():
    x = sp.symbols("x")
    a = ms.Matrix([[x, 1], [2, x]])
    b = ms.Matrix([[1, x], [x, 3]])
    assert a * b == sp.Matrix([[x, 1], [2, x]]) * sp.Matrix([[1, x], [x, 3]])
    assert a.det() == x**2 - 2


def test_matrix_constructors_indexing_and_mutation():
    assert ms.zeros(2, 3) == sp.zeros(2, 3)
    assert ms.ones(2, 2) == sp.ones(2, 2)
    assert ms.eye(4) == sp.eye(4)
    assert ms.diag(1, 2, 3) == sp.diag(1, 2, 3)
    matrix = ms.Matrix(2, 2, [1, 2, 3, 4])
    matrix[0, 1] = 9
    assert matrix.tolist() == [[1, 9], [3, 4]]
