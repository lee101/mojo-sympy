from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-sympy.so")
I = ctypes.c_int64

_SIGNATURES = {
    "msp_poly_add": ([I, I, I, I, I], None),
    "msp_poly_sub": ([I, I, I, I, I], None),
    "msp_poly_mul": ([I, I, I, I, I], None),
    "msp_poly_pow": ([I, I, I, I, I, I], None),
    "msp_poly_derivative": ([I, I, I, I], I),
    "msp_poly_eval_i64": ([I, I, I], I),
    "msp_mat_add_i64": ([I, I, I, I, I], None),
    "msp_mat_mul_i64_rows": ([I] * 8, None),
    "msp_mat_mul_f64_rows": ([I] * 8, None),
    "msp_mat_transpose_i64": ([I, I, I, I], None),
    "msp_det_bareiss_i64": ([I, I], I),
}

_library: ctypes.CDLL | None = None




def build() -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if (
        not os.path.exists(LIB_PATH)
        or os.path.getmtime(LIB_PATH) < os.path.getmtime(source)
    ):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = args
            fn.restype = result
    return _library


def parallel_ready() -> bool:
    """True when a large kernel can be fanned out over a thread pool."""
    return (os.cpu_count() or 1) > 1


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise TypeError("FFI buffers must be contiguous NumPy arrays")
    if value.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError(f"unsupported FFI dtype: {value.dtype}")
    if value.size == 0:
        raise ValueError("empty buffers do not cross the FFI boundary")
    address = int(value.ctypes.data)
    if address == 0:
        raise ValueError("null buffers do not cross the FFI boundary")
    if address % value.dtype.alignment:
        raise ValueError("unaligned buffers do not cross the FFI boundary")
    return address
