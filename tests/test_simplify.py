import pytest
import sympy as sp

import mojosympy as ms
from mojosympy.simplify import _expand_default


x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "expr",
    [
        (x + 1) ** 12,
        (2 * x - 3) ** 7,
        (x**2 + x + 1) * (x**3 - 2 * x + 4),
        (x / 2 + sp.Rational(1, 3)) ** 6,
    ],
)
def test_expand_univariate_parity(expr):
    assert ms.expand(expr) == sp.expand(expr)


def test_expand_reuses_immutable_expression_result():
    expr = (x + 1) ** 9 * (x - 2) ** 7
    _expand_default.cache_clear()
    expected = ms.expand(expr)
    assert ms.expand(expr) is expected
    assert _expand_default.cache_info().hits == 1


def test_expand_multivariate_fallback_parity():
    expr = (x + 2 * y - 1) ** 7
    assert ms.expand(expr) == sp.expand(expr)


def test_expand_transcendental_fallback_parity():
    expr = sp.exp(x + 1) * (x + 2) ** 3
    assert ms.expand(expr) == sp.expand(expr)


@pytest.mark.parametrize(
    "expr",
    [
        (x + 1) ** 2 - x**2 - 2 * x - 1,
        (x - 2) * (x + 2) - x**2 + 4,
        x + x - 2 * x,
    ],
)
def test_simplify_polynomial_parity(expr):
    assert ms.simplify(expr) == sp.simplify(expr)


def test_standard_simplification_names():
    expr = (x**2 - 1) / (x - 1)
    assert ms.cancel(expr) == sp.cancel(expr)
    assert ms.factor(x**4 - 1) == sp.factor(x**4 - 1)
    assert ms.together(1 / x + 1 / (x + 1)) == sp.together(
        1 / x + 1 / (x + 1)
    )
    assert ms.collect(x * y + x, x) == sp.collect(x * y + x, x)
    assert ms.apart(1 / (x**2 - 1), x) == sp.apart(1 / (x**2 - 1), x)
    assert ms.powsimp(x**y * x**2) == sp.powsimp(x**y * x**2)
    assert ms.trigsimp(sp.sin(x) ** 2 + sp.cos(x) ** 2) == 1
    radical = 1 / (1 + sp.sqrt(2))
    assert ms.radsimp(radical) == sp.radsimp(radical)


def test_upstream_symbolic_namespace_is_available():
    assert ms.diff(ms.sin(x) ** 2, x) == sp.diff(sp.sin(x) ** 2, x)
    assert ms.integrate(x**3, x) == sp.integrate(x**3, x)
