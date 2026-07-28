import random

import pytest
import sympy as sp

import mojosympy as ms


x = sp.symbols("x")


def native(poly):
    return sp.Poly(poly.as_expr(), *poly.gens)


@pytest.mark.parametrize("seed", range(5))
def test_integer_add_sub_mul_parity(seed):
    rng = random.Random(seed)
    a = [rng.randint(-20, 20) for _ in range(18)]
    b = [rng.randint(-20, 20) for _ in range(13)]
    ours_a = ms.Poly.from_list(a, x)
    ours_b = ms.Poly.from_list(b, x)
    theirs_a = sp.Poly.from_list(a, x)
    theirs_b = sp.Poly.from_list(b, x)
    assert native(ours_a + ours_b) == theirs_a + theirs_b
    assert native(ours_a - ours_b) == theirs_a - theirs_b
    assert native(ours_a * ours_b) == theirs_a * theirs_b


def test_rational_polynomial_arithmetic():
    ours_a = ms.Poly(x**3 / 3 - x / 7 + sp.Rational(2, 5), x)
    ours_b = ms.Poly(x**2 / 11 + x / 2 - sp.Rational(5, 9), x)
    theirs_a = sp.Poly(ours_a.as_expr(), x)
    theirs_b = sp.Poly(ours_b.as_expr(), x)
    assert native(ours_a + ours_b) == theirs_a + theirs_b
    assert native(ours_a - ours_b) == theirs_a - theirs_b
    assert native(ours_a * ours_b) == theirs_a * theirs_b


@pytest.mark.parametrize("exponent", [0, 1, 2, 5, 12])
def test_power_parity(exponent):
    ours = ms.Poly(x**3 - 2 * x + 1, x) ** exponent
    theirs = sp.Poly(x**3 - 2 * x + 1, x) ** exponent
    assert native(ours) == theirs


def test_power_simd_tail_parity():
    coefficients = [1, -2, 0, 3, 1, -1]
    ours = ms.Poly.from_list(coefficients, x) ** 7
    theirs = sp.Poly.from_list(coefficients, x) ** 7
    assert native(ours) == theirs


def test_power_parallel_threshold_parity():
    coefficients = [1 if i % 17 == 0 else 0 for i in range(2048)]
    ours = ms.Poly.from_list(coefficients, x) ** 2
    theirs = sp.Poly.from_list(coefficients, x) ** 2
    assert native(ours) == theirs


@pytest.mark.parametrize("order", [0, 1, 2, 5, 9])
def test_derivative_parity(order):
    ours = ms.Poly(3 * x**7 - 2 * x**4 + x - 8, x).diff(order)
    theirs = sp.Poly(3 * x**7 - 2 * x**4 + x - 8, x).diff((x, order))
    assert native(ours) == theirs


@pytest.mark.parametrize("value", [-7, -1, 0, 2, 13])
def test_evaluation_parity(value):
    ours = ms.Poly(x**8 - 4 * x**5 + 9 * x**2 - 3, x)
    theirs = sp.Poly(ours.as_expr(), x)
    assert ours.eval(value) == theirs.eval(value)
    assert ours(value) == theirs(value)


def test_rational_polynomial_integer_evaluation_parity():
    ours = ms.Poly(x**3 / 7 - x / 3 + sp.Rational(2, 5), x)
    theirs = sp.Poly(ours.as_expr(), x)
    assert ours.eval(11) == theirs.eval(11)


def test_large_integer_falls_back_without_overflow():
    huge = 1 << 100
    ours = ms.Poly(huge * x + 1, x) * ms.Poly(x - huge, x)
    theirs = sp.Poly(huge * x + 1, x) * sp.Poly(x - huge, x)
    assert native(ours) == theirs


def test_delegated_division_gcd_and_content():
    a = ms.Poly(x**6 - 1, x)
    b = ms.Poly(x**2 - 1, x)
    quotient, remainder = a.div(b)
    expected_q, expected_r = sp.Poly(x**6 - 1, x).div(sp.Poly(x**2 - 1, x))
    assert native(quotient) == expected_q
    assert native(remainder) == expected_r
    assert native(a.gcd(b)) == sp.Poly(x**2 - 1, x)
    assert a.content() == 1


def test_poly_constructor_and_properties_match():
    ours = ms.Poly([1, 0, -2, 3], x)
    theirs = sp.Poly([1, 0, -2, 3], x)
    assert ours.all_coeffs() == theirs.all_coeffs()
    assert ours.degree() == theirs.degree()
    assert ours.LC() == theirs.LC()
    assert ours.TC() == theirs.TC()
    assert ours.domain == theirs.domain
