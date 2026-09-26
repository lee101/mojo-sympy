from __future__ import annotations

from functools import lru_cache, reduce
from math import gcd
from typing import Any

import numpy as np
import sympy as _sp
from sympy.polys.domains import ZZ
from sympy.polys.polyclasses import DMP

from ._lib import addr, lib

I64_MAX = (1 << 63) - 1


def _lcm(a: int, b: int) -> int:
    return abs(a // gcd(a, b) * b)


def _bounded(value: int) -> bool:
    return abs(value) <= I64_MAX


@lru_cache(maxsize=512)
def _encoded(poly: _sp.Poly) -> tuple[np.ndarray, int]:
    coeffs = list(reversed(poly.all_coeffs()))
    if not coeffs:
        coeffs = [_sp.Integer(0)]
    if not all(value.is_Rational for value in coeffs):
        raise TypeError("the exact kernels require rational coefficients")
    denominator = reduce(_lcm, (int(value.q) for value in coeffs), 1)
    integers = [int(value * denominator) for value in coeffs]
    if not all(_bounded(value) for value in integers):
        raise OverflowError("coefficient does not fit int64")
    return np.ascontiguousarray(integers, dtype=np.int64), denominator


def _from_encoded(values: np.ndarray, denominator: int, gens: tuple) -> "Poly":
    integers = [int(value) for value in values]
    while len(integers) > 1 and integers[-1] == 0:
        integers.pop()
    if denominator == 1:
        rep = DMP.new(list(reversed(integers)), ZZ, 0)
        return Poly(_sp.Poly.new(rep, *gens))
    coeffs = [_sp.Rational(value, denominator) for value in reversed(integers)]
    return Poly(_sp.Poly.from_list(coeffs, *gens))


def _unwrap(value):
    if isinstance(value, Poly):
        return value._poly
    if isinstance(value, MatrixLike):
        return value._matrix
    return value


def _wrap(value):
    if isinstance(value, _sp.Poly):
        return Poly(value)
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


class MatrixLike:
    pass


class Poly:
    """SymPy-compatible univariate polynomial with exact Mojo fast paths."""

    __slots__ = ("_poly",)

    def __init__(self, rep, *gens, **args):
        if isinstance(rep, Poly):
            self._poly = rep._poly
        elif isinstance(rep, _sp.Poly) and not gens and not args:
            self._poly = rep
        else:
            self._poly = _sp.Poly(rep, *gens, **args)

    @classmethod
    def from_list(cls, rep, *gens, **args):
        return cls(_sp.Poly.from_list(rep, *gens, **args))

    @classmethod
    def from_dict(cls, rep, *gens, **args):
        return cls(_sp.Poly.from_dict(rep, *gens, **args))

    @property
    def gens(self):
        return self._poly.gens

    @property
    def domain(self):
        return self._poly.domain

    @property
    def rep(self):
        return self._poly.rep

    @property
    def is_zero(self):
        return self._poly.is_zero

    def as_expr(self, *gens):
        return self._poly.as_expr(*gens)

    def all_coeffs(self):
        return self._poly.all_coeffs()

    def degree(self, gen=0):
        return self._poly.degree(gen)

    def LC(self, order=None):
        return self._poly.LC(order)

    def TC(self):
        return self._poly.TC()

    def _coerce(self, other) -> _sp.Poly:
        if isinstance(other, Poly):
            return other._poly
        if isinstance(other, _sp.Poly):
            return other
        return _sp.Poly(other, *self.gens, domain=self.domain)

    def _same_univariate(self, other: _sp.Poly) -> bool:
        return len(self.gens) == 1 and self.gens == other.gens

    def _binary(self, other, operation: str):
        rhs = self._coerce(other)
        fallback = getattr(self._poly, f"__{operation}__")
        if not self._same_univariate(rhs):
            return _wrap(fallback(rhs))
        try:
            a, da = _encoded(self._poly)
            b, db = _encoded(rhs)
            if operation in ("add", "sub"):
                denominator = _lcm(da, db)
                scale_a = denominator // da
                scale_b = denominator // db
                max_a = int(np.max(np.abs(a), initial=0)) * scale_a
                max_b = int(np.max(np.abs(b), initial=0)) * scale_b
                bound = max_a + max_b
                if bound > I64_MAX:
                    raise OverflowError
                aa = np.ascontiguousarray(a * scale_a, dtype=np.int64)
                bb = np.ascontiguousarray(b * scale_b, dtype=np.int64)
                result = np.empty(max(len(aa), len(bb)), dtype=np.int64)
                fn = lib().msp_poly_add if operation == "add" else lib().msp_poly_sub
                fn(addr(aa), addr(bb), addr(result), len(aa), len(bb))
            else:
                denominator = da * db
                bound = (
                    min(len(a), len(b))
                    * int(np.max(np.abs(a), initial=0))
                    * int(np.max(np.abs(b), initial=0))
                )
                if bound > I64_MAX:
                    raise OverflowError
                result = np.empty(len(a) + len(b) - 1, dtype=np.int64)
                lib().msp_poly_mul(addr(a), addr(b), addr(result), len(a), len(b))
            return _from_encoded(result, denominator, self.gens)
        except (TypeError, OverflowError):
            return _wrap(fallback(rhs))

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return Poly(other, *self.gens) - self

    def __mul__(self, other):
        return self._binary(other, "mul")

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, _sp.Integer)) or int(exponent) < 0:
            return _wrap(self._poly ** exponent)
        exponent = int(exponent)
        if len(self.gens) != 1:
            return _wrap(self._poly**exponent)
        try:
            source, denominator = _encoded(self._poly)
            max_coeff = int(np.max(np.abs(source), initial=0))
            if exponent == 0:
                bound = 1
            else:
                bound = pow(max(1, len(source)), exponent - 1) * pow(
                    max_coeff, exponent
                )
            result_denominator = pow(denominator, exponent)
            if bound > I64_MAX:
                raise OverflowError
            capacity = max(1, (len(source) - 1) * exponent + 1)
            buffers = np.empty(capacity * 2, dtype=np.int64)
            result = buffers[:capacity]
            work = buffers[capacity:]
            lib().msp_poly_pow(
                addr(source),
                addr(result),
                addr(work),
                len(source),
                exponent,
                capacity,
            )
            return _from_encoded(result, result_denominator, self.gens)
        except (TypeError, OverflowError):
            return _wrap(self._poly**exponent)

    def diff(self, *specs, **kwargs):
        order = 1
        if specs:
            if len(specs) == 1 and isinstance(specs[0], (int, _sp.Integer)):
                order = int(specs[0])
            else:
                return _wrap(self._poly.diff(*specs, **kwargs))
        if kwargs or order < 0 or len(self.gens) != 1:
            return _wrap(self._poly.diff(*specs, **kwargs))
        try:
            source, denominator = _encoded(self._poly)
            factor = 1
            degree = len(source) - 1
            for value in range(order):
                factor *= max(1, degree - value)
            if int(np.max(np.abs(source), initial=0)) * factor > I64_MAX:
                raise OverflowError
            result = np.empty(max(1, len(source) - order), dtype=np.int64)
            size = lib().msp_poly_derivative(
                addr(source), addr(result), len(source), order
            )
            return _from_encoded(result[:size], denominator, self.gens)
        except (TypeError, OverflowError):
            return _wrap(self._poly.diff(*specs, **kwargs))

    def eval(self, x, a=None, auto=True):
        if a is not None:
            return self._poly.eval(x, a, auto=auto)
        if isinstance(x, (int, _sp.Integer)) and len(self.gens) == 1:
            try:
                source, denominator = _encoded(self._poly)
                xv = int(x)
                bound = 0
                for value in reversed(source):
                    bound = bound * abs(xv) + abs(int(value))
                    if bound > I64_MAX:
                        raise OverflowError
                result = lib().msp_poly_eval_i64(addr(source), len(source), xv)
                return _sp.Rational(result, denominator)
            except (TypeError, OverflowError):
                pass
        return self._poly.eval(x, auto=auto)

    def __call__(self, *values):
        if len(values) == 1:
            return self.eval(values[0])
        return self._poly(*values)

    def __neg__(self):
        return Poly(-self._poly)

    def __eq__(self, other):
        return self._poly == _unwrap(other)

    def __hash__(self):
        return hash(self._poly)

    def __repr__(self):
        return repr(self._poly)

    def __str__(self):
        return str(self._poly)

    def _sympy_(self):
        return self.as_expr()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._poly, name)
        if not callable(attribute):
            return _wrap(attribute)

        def delegated(*args, **kwargs):
            return _wrap(
                attribute(
                    *(_unwrap(value) for value in args),
                    **{key: _unwrap(value) for key, value in kwargs.items()},
                )
            )

        return delegated
