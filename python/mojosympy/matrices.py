from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Any

import numpy as np
import sympy as _sp

from ._lib import addr, lib
from .polys import I64_MAX, MatrixLike, Poly


def _lcm(a: int, b: int) -> int:
    return abs(a // gcd(a, b) * b)


def _unwrap(value):
    if isinstance(value, Matrix):
        return value._matrix
    if isinstance(value, Poly):
        return value._poly
    return value


def _wrap(value):
    if isinstance(value, _sp.MatrixBase):
        return Matrix(value)
    if isinstance(value, _sp.Poly):
        return Poly(value)
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _encoded(matrix: _sp.MatrixBase) -> tuple[np.ndarray, int]:
    values = list(matrix)
    if not all(value.is_Rational for value in values):
        raise TypeError("the exact kernels require rational entries")
    denominator = reduce(_lcm, (int(value.q) for value in values), 1)
    integers = [int(value * denominator) for value in values]
    if any(abs(value) > I64_MAX for value in integers):
        raise OverflowError
    return (
        np.ascontiguousarray(integers, dtype=np.int64).reshape(matrix.shape),
        denominator,
    )


def _float_encoded(matrix: _sp.MatrixBase) -> np.ndarray:
    values = list(matrix)
    if not all(
        isinstance(value, _sp.Float)
        and value.is_real is True
        and value._prec <= 53
        for value in values
    ):
        raise TypeError("the floating kernel requires float64-compatible entries")
    result = np.ascontiguousarray([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise TypeError("the floating kernel requires finite entries")
    return result.reshape(matrix.shape)


def _from_encoded(values: np.ndarray, denominator: int) -> "Matrix":
    rows, cols = values.shape
    data = [_sp.Rational(int(value), denominator) for value in values.ravel()]
    return Matrix(rows, cols, data)


class Matrix(MatrixLike):
    """Mutable dense SymPy matrix with exact Mojo arithmetic fast paths."""

    __slots__ = ("_matrix",)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Matrix):
            self._matrix = args[0]._matrix.copy()
        elif len(args) == 1 and isinstance(args[0], _sp.MatrixBase):
            self._matrix = _sp.Matrix(args[0])
        else:
            self._matrix = _sp.Matrix(*args, **kwargs)

    @property
    def rows(self):
        return self._matrix.rows

    @property
    def cols(self):
        return self._matrix.cols

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def T(self):
        if not self.rows or not self.cols:
            return Matrix(self._matrix.T)
        try:
            source, denominator = _encoded(self._matrix)
            result = np.empty((self.cols, self.rows), dtype=np.int64)
            lib().msp_mat_transpose_i64(
                addr(source), addr(result), self.rows, self.cols
            )
            return _from_encoded(result, denominator)
        except (TypeError, OverflowError):
            return Matrix(self._matrix.T)

    def transpose(self):
        return self.T

    def _coerce_matrix(self, other):
        if isinstance(other, Matrix):
            return other
        if isinstance(other, _sp.MatrixBase):
            return Matrix(other)
        return None

    def _add(self, other, subtract: bool):
        rhs = self._coerce_matrix(other)
        if rhs is None:
            native = self._matrix - other if subtract else self._matrix + other
            return _wrap(native)
        if self.shape != rhs.shape:
            native = self._matrix - rhs._matrix if subtract else self._matrix + rhs._matrix
            return _wrap(native)
        if not self.rows or not self.cols:
            return Matrix(self._matrix - rhs._matrix if subtract else self._matrix + rhs._matrix)
        try:
            a, da = _encoded(self._matrix)
            b, db = _encoded(rhs._matrix)
            denominator = _lcm(da, db)
            scale_a = denominator // da
            scale_b = denominator // db
            max_a = int(np.max(np.abs(a))) * scale_a
            max_b = int(np.max(np.abs(b))) * scale_b
            bound = max_a + max_b
            if bound > I64_MAX:
                raise OverflowError
            aa = np.ascontiguousarray(a * scale_a, dtype=np.int64)
            bb = np.ascontiguousarray(b * scale_b, dtype=np.int64)
            result = np.empty_like(aa)
            lib().msp_mat_add_i64(
                addr(aa), addr(bb), addr(result), result.size, int(subtract)
            )
            return _from_encoded(result, denominator)
        except (TypeError, OverflowError):
            native = self._matrix - rhs._matrix if subtract else self._matrix + rhs._matrix
            return Matrix(native)

    def __add__(self, other):
        return self._add(other, False)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._add(other, True)

    def __rsub__(self, other):
        rhs = self._coerce_matrix(other)
        if rhs is None:
            return _wrap(other - self._matrix)
        return rhs - self

    def _matmul(self, other):
        rhs = self._coerce_matrix(other)
        if rhs is None:
            return NotImplemented
        if self.cols != rhs.rows:
            return Matrix(self._matrix * rhs._matrix)
        if not self.rows or not self.cols or not rhs.cols:
            return Matrix(self._matrix * rhs._matrix)
        try:
            a, da = _encoded(self._matrix)
            b, db = _encoded(rhs._matrix)
            denominator = da * db
            bound = (
                self.cols
                * int(np.max(np.abs(a)))
                * int(np.max(np.abs(b)))
            )
            if bound > I64_MAX:
                raise OverflowError
            result = np.empty((self.rows, rhs.cols), dtype=np.int64)
            lib().msp_mat_mul_i64(
                addr(a),
                addr(b),
                addr(result),
                self.rows,
                self.cols,
                rhs.cols,
            )
            return _from_encoded(result, denominator)
        except (TypeError, OverflowError):
            pass
        try:
            a = _float_encoded(self._matrix)
            b = _float_encoded(rhs._matrix)
            result = np.empty((self.rows, rhs.cols), dtype=np.float64)
            lib().msp_mat_mul_f64(
                addr(a),
                addr(b),
                addr(result),
                self.rows,
                self.cols,
                rhs.cols,
            )
            return Matrix(
                self.rows,
                rhs.cols,
                [_sp.Float(float(value)) for value in result.ravel()],
            )
        except TypeError:
            return Matrix(self._matrix * rhs._matrix)

    def __mul__(self, other):
        rhs = self._coerce_matrix(other)
        if rhs is not None:
            return self._matmul(rhs)
        return _wrap(self._matrix * _unwrap(other))

    def __rmul__(self, other):
        rhs = self._coerce_matrix(other)
        if rhs is not None:
            return rhs._matmul(self)
        return _wrap(_unwrap(other) * self._matrix)

    def __matmul__(self, other):
        result = self._matmul(other)
        if result is NotImplemented:
            return NotImplemented
        return result

    def det(self, method="bareiss", iszerofunc=None):
        if self.rows != self.cols:
            return self._matrix.det(method=method, iszerofunc=iszerofunc)
        if self.rows == 0:
            return _sp.Integer(1)
        if method not in ("bareiss", "domain-ge", "det_LU") or iszerofunc is not None:
            return self._matrix.det(method=method, iszerofunc=iszerofunc)
        try:
            source, denominator = _encoded(self._matrix)
            row_norm = max(sum(abs(int(value)) for value in row) for row in source)
            minor_bound = pow(max(1, row_norm), self.rows)
            if minor_bound * minor_bound > I64_MAX:
                raise OverflowError
            work = source.copy()
            value = lib().msp_det_bareiss_i64(addr(work), self.rows)
            return _sp.Rational(value, pow(denominator, self.rows))
        except (TypeError, OverflowError):
            return self._matrix.det(method=method, iszerofunc=iszerofunc)

    def tolist(self):
        return self._matrix.tolist()

    def __getitem__(self, key):
        return _wrap(self._matrix[key])

    def __setitem__(self, key, value):
        self._matrix[key] = _unwrap(value)

    def __iter__(self):
        return iter(self._matrix)

    def __len__(self):
        return len(self._matrix)

    def __eq__(self, other):
        return self._matrix == _unwrap(other)

    def __neg__(self):
        return Matrix(-self._matrix)

    def __repr__(self):
        return repr(self._matrix)

    def __str__(self):
        return str(self._matrix)

    def _sympy_(self):
        return self._matrix

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._matrix, name)
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


def zeros(*args, **kwargs):
    return Matrix(_sp.zeros(*args, **kwargs))


def ones(*args, **kwargs):
    return Matrix(_sp.ones(*args, **kwargs))


def eye(*args, **kwargs):
    return Matrix(_sp.eye(*args, **kwargs))


def diag(*args, **kwargs):
    return Matrix(_sp.diag(*(_unwrap(value) for value in args), **kwargs))
