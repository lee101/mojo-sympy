"""A Mojo backend for a focused, SymPy-compatible exact algebra subset."""

from __future__ import annotations

import sympy as _sympy

from .matrices import Matrix, diag, eye, ones, zeros
from .polys import Poly
from .simplify import (
    apart,
    cancel,
    collect,
    expand,
    factor,
    powsimp,
    radsimp,
    simplify,
    together,
    trigsimp,
)

__version__ = "0.1.0"

Symbol = _sympy.Symbol
symbols = _sympy.symbols
Integer = _sympy.Integer
Rational = _sympy.Rational
Float = _sympy.Float
Expr = _sympy.Expr
S = _sympy.S
I = _sympy.I
oo = _sympy.oo
nan = _sympy.nan
pi = _sympy.pi
E = _sympy.E
sympify = _sympy.sympify
Eq = _sympy.Eq
solve = _sympy.solve
diff = _sympy.diff
integrate = _sympy.integrate
count_ops = _sympy.count_ops
sin = _sympy.sin
cos = _sympy.cos
tan = _sympy.tan
exp = _sympy.exp
log = _sympy.log
sqrt = _sympy.sqrt


def __getattr__(name):
    return getattr(_sympy, name)


__all__ = [
    "Poly",
    "Matrix",
    "Symbol",
    "symbols",
    "Integer",
    "Rational",
    "Float",
    "Expr",
    "S",
    "expand",
    "simplify",
    "factor",
    "cancel",
    "collect",
    "together",
    "apart",
    "powsimp",
    "trigsimp",
    "radsimp",
    "zeros",
    "ones",
    "eye",
    "diag",
]
