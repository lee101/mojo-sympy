# mojo-sympy

`mojo-sympy` is a focused port of compute-heavy SymPy algebra to Mojo. It
keeps SymPy's symbolic object model and mirrors the covered public names while
moving dense exact polynomial and matrix loops into one compiled shared
library.

This is useful when an application already represents formulas with SymPy but
spends its time multiplying dense polynomials or dense matrices. Results
remain SymPy expressions, exact rationals, or thin `Poly` and `Matrix`
compatibility wrappers; unsupported cases fall back to upstream SymPy instead
of silently changing arithmetic.

## Covered subset

| area | accelerated API |
| --- | --- |
| Polynomials | univariate `Poly` over `ZZ` and `QQ`: `+`, `-`, `*`, non-negative integer powers, `diff`, integer `eval` |
| Simplification | exact univariate polynomial expansion through `expand`; polynomial cancellation in `simplify` |
| Matrices | dense rational `Matrix` addition, subtraction, multiplication, transpose, and Bareiss determinant; dense real-float multiplication |
| Compatibility | `Poly.div`, `gcd`, `content`, matrix `inv`, `rref`, `rank`, `solve`, and common `factor`, `cancel`, `collect`, `together`, `apart`, `powsimp`, `trigsimp`, and `radsimp` names delegate to SymPy and wrap matrix/polynomial results |

The int64 kernels are exact, not modular. Before crossing the FFI boundary,
the Python layer computes conservative bounds for every result. Coefficients
that are too large, symbolic, multivariate, or outside `QQ` stay on SymPy's
arbitrary-precision implementation.

Not covered by Mojo kernels: multivariate polynomial arithmetic, polynomial
factorization and Gröbner bases, arbitrary-precision kernel arithmetic, sparse
matrices, symbolic matrix products, eigenvalue algorithms, integration,
equation solving, and the full general-purpose simplification pipeline.
Those APIs remain available through the SymPy namespace where practical, but
they are not claimed as ports.

## Install

```bash
pixi install
pixi run build
```

The first command installs the pinned Mojo nightly, Python, NumPy, SymPy, and
pytest. The build task creates `dist/libmojo-sympy.so`. Tests and benchmarks
run inside the same environment:

```bash
pixi run test
pixi run bench
```

## Usage

Run this from the repository with `pixi run python`:

```python
import mojosympy as sp

x = sp.symbols("x")
p = sp.Poly(x**3 - 2*x + 1, x)
q = (p + 1) ** 4
print(q.diff())

A = sp.Matrix([[1, 2], [3, 4]])
B = sp.Matrix([[5, 6], [7, 8]])
print(A * B)
print(A.det())

assert sp.expand((x + 1)**8) == (
    x**8 + 8*x**7 + 28*x**6 + 56*x**5
    + 70*x**4 + 56*x**3 + 28*x**2 + 8*x + 1
)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Each row compares identical symbolic inputs and checks equality
before timing.

| case | mojo-sympy | sympy | result |
| --- | ---: | ---: | ---: |
| Poly multiply (degree 400) | 0.238 ms | 12.398 ms | 52.16x faster |
| Poly power (degree 5, exponent 8) | 0.037 ms | 0.100 ms | 2.74x faster |
| expand two binomials (degree 40) | 0.001 ms | 0.002 ms | 4.03x faster |
| Matrix multiply (120 x 120 integers) | 5.329 ms | 342.221 ms | 64.22x faster |
| Matrix multiply (50 x 50 rationals) | 8.205 ms | 47.581 ms | 5.80x faster |
| determinant (6 x 6 integers) | 0.145 ms | 1.600 ms | 11.00x faster |

Immutable polynomial encodings and default expansion results use bounded
caches, matching SymPy's reuse behavior without unbounded retention. Integer
results over `ZZ` are rebuilt directly from their exact dense representation,
and polynomial power uses one allocation for its result and scratch buffers.

## How it works

`src/kernels.mojo` is one compilation unit exporting a small C ABI. Python
passes C-contiguous NumPy buffers as integer addresses through `ctypes`; Mojo
reconstructs `UnsafePointer[..., AnyOrigin[mut=True]]` values inside
non-parametric `@export` functions. A polynomial buffer stores coefficients in
ascending degree order. A matrix buffer is row-major.

Rational data is converted to an int64 numerator buffer plus a common Python
integer denominator. Mojo performs convolution, SIMD elementwise operations,
row-major matrix products, and fraction-free Bareiss elimination. Python
restores SymPy `Rational` values afterward. Python owns inputs, outputs, and
scratch buffers, so the shared library allocates nothing across the ABI and
has no cross-language lifetime management.

Polynomial power and matrix multiplication accumulate contiguous coefficient
or row spans at the native SIMD width, followed by scalar remainder loops.
Large powers split independent output coefficients across CPU workers, and
large matrix products split independent output rows. The Mojo CPU runtime is
initialized lazily, so smaller operations stay serial. Matrix wrappers retain
their exact contiguous input encoding and invalidate it on mutation, avoiding
repeat copies across the FFI boundary.

No GPU path is included. The exact multiply-add loops as implemented move a
loaded operand plus a read/modified/written accumulator for each two arithmetic
operations, well below two operations per byte. At the covered sizes, device
transfer and launch overhead would add cost rather than remove it.

## License

MIT
