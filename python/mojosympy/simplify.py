from __future__ import annotations

from functools import lru_cache

import sympy as _sp

from .polys import Poly


class _UnsupportedPolynomial(Exception):
    pass


def _tree_poly(expr, generator) -> Poly:
    if expr == generator:
        return Poly(generator, generator)
    if expr.is_Rational:
        return Poly(expr, generator, domain="QQ")
    if expr.is_Add:
        result = Poly(0, generator, domain="QQ")
        for arg in expr.args:
            result += _tree_poly(arg, generator)
        return result
    if expr.is_Mul:
        result = Poly(1, generator, domain="QQ")
        for arg in expr.args:
            result *= _tree_poly(arg, generator)
        return result
    if expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        return _tree_poly(expr.base, generator) ** int(expr.exp)
    raise _UnsupportedPolynomial


@lru_cache(maxsize=256)
def _expand_default(expr):
    symbols = expr.free_symbols
    if len(symbols) == 1:
        try:
            return _tree_poly(expr, next(iter(symbols))).as_expr()
        except (_UnsupportedPolynomial, TypeError, ValueError):
            pass
    return _sp.expand(expr)


def expand(e, deep=True, modulus=None, power_base=True, power_exp=True, mul=True, log=True, multinomial=True, basic=True, **hints):
    expr = _sp.sympify(e)
    if (
        deep
        and modulus is None
        and power_base
        and power_exp
        and mul
        and log
        and multinomial
        and basic
        and not hints
    ):
        return _expand_default(expr)
    symbols = expr.free_symbols
    use_default_algebraic_hints = (
        deep
        and modulus is None
        and power_base
        and power_exp
        and mul
        and multinomial
        and basic
        and not hints
    )
    if len(symbols) == 1 and use_default_algebraic_hints:
        try:
            return _tree_poly(expr, next(iter(symbols))).as_expr()
        except (_UnsupportedPolynomial, TypeError, ValueError):
            pass
    return _sp.expand(
        expr,
        deep=deep,
        modulus=modulus,
        power_base=power_base,
        power_exp=power_exp,
        mul=mul,
        log=log,
        multinomial=multinomial,
        basic=basic,
        **hints,
    )


def simplify(expr, ratio=1.7, measure=_sp.count_ops, rational=False, inverse=False, doit=True, **kwargs):
    value = _sp.sympify(expr)
    expanded = expand(value)
    if expanded == 0 or measure(expanded) <= measure(value):
        return expanded
    return _sp.simplify(
        value,
        ratio=ratio,
        measure=measure,
        rational=rational,
        inverse=inverse,
        doit=doit,
        **kwargs,
    )


factor = _sp.factor
cancel = _sp.cancel
collect = _sp.collect
together = _sp.together
apart = _sp.apart
powsimp = _sp.powsimp
trigsimp = _sp.trigsimp
radsimp = _sp.radsimp
