"""Benchmarks against upstream SymPy on identical symbolic inputs."""

from __future__ import annotations

import gc
import os
import platform
import sys
import time

import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojosympy as ms  # noqa: E402


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def timeit(function, target: float = 0.12, repeat: int = 5) -> float:
    number = 1
    while True:
        gc.disable()
        start = time.perf_counter()
        for _ in range(number):
            function()
        elapsed = time.perf_counter() - start
        gc.enable()
        if elapsed >= target or number >= 1024:
            break
        number *= 2
    best = elapsed / number
    for _ in range(repeat - 1):
        gc.disable()
        start = time.perf_counter()
        for _ in range(number):
            function()
        elapsed = time.perf_counter() - start
        gc.enable()
        best = min(best, elapsed / number)
    return best


def matrix_data(rows: int, cols: int, offset: int = 0):
    return [
        [((r * 17 + c * 31 + offset) % 11) - 5 for c in range(cols)]
        for r in range(rows)
    ]


def cases():
    x = sp.symbols("x")

    coeff_a = [((i * 17) % 3) - 1 for i in range(401)]
    coeff_b = [((i * 29 + 1) % 3) - 1 for i in range(401)]
    mojo_a = ms.Poly(coeff_a, x)
    mojo_b = ms.Poly(coeff_b, x)
    sympy_a = sp.Poly(coeff_a, x)
    sympy_b = sp.Poly(coeff_b, x)
    yield "Poly multiply (degree 400)", lambda: mojo_a * mojo_b, lambda: sympy_a * sympy_b

    mojo_power = ms.Poly(x**5 + x + 1, x)
    sympy_power = sp.Poly(x**5 + x + 1, x)
    yield "Poly power (degree 5, exponent 8)", lambda: mojo_power**8, lambda: sympy_power**8

    expression = (x + 1) ** 20 * (x - 1) ** 20
    yield "expand two binomials (degree 40)", lambda: ms.expand(expression), lambda: sp.expand(expression)

    left_data = matrix_data(120, 120)
    right_data = matrix_data(120, 120, 3)
    mojo_left = ms.Matrix(left_data)
    mojo_right = ms.Matrix(right_data)
    sympy_left = sp.Matrix(left_data)
    sympy_right = sp.Matrix(right_data)
    yield "Matrix multiply (120 x 120 integers)", lambda: mojo_left * mojo_right, lambda: sympy_left * sympy_right

    rational_left = [
        [sp.Rational(value, 7) for value in row]
        for row in matrix_data(50, 50)
    ]
    rational_right = [
        [sp.Rational(value, 11) for value in row]
        for row in matrix_data(50, 50, 5)
    ]
    mojo_left_q = ms.Matrix(rational_left)
    mojo_right_q = ms.Matrix(rational_right)
    sympy_left_q = sp.Matrix(rational_left)
    sympy_right_q = sp.Matrix(rational_right)
    yield "Matrix multiply (50 x 50 rationals)", lambda: mojo_left_q * mojo_right_q, lambda: sympy_left_q * sympy_right_q

    determinant_data = matrix_data(6, 6, 7)
    mojo_det = ms.Matrix(determinant_data)
    sympy_det = sp.Matrix(determinant_data)
    yield "determinant (6 x 6 integers)", mojo_det.det, sympy_det.det


def main() -> None:
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| case | mojo-sympy | sympy | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases():
        actual = ours()
        expected = upstream()
        if actual != expected:
            raise AssertionError(f"benchmark parity failed for {name}")
        mojo_time = timeit(ours)
        sympy_time = timeit(upstream)
        ratio = sympy_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{sympy_time * 1e3:.3f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
