"""Exact polynomial and dense matrix kernels exposed through a C ABI.

Python owns every buffer. Integer inputs are admitted only after the binding
has proved that all intermediates fit in signed 64 bits.
"""

from std.sys.info import simd_width_of

comptime IW = simd_width_of[DType.int64]()
comptime FW = simd_width_of[DType.float64]()
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("msp_poly_add")
def poly_add(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, m: Int) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    var common = min(n, m)
    var i = 0
    while i + IW <= common:
        dst.unsafe_store(
            i, a.unsafe_load[width=IW](i) + b.unsafe_load[width=IW](i)
        )
        i += IW
    while i < common:
        dst[unsafe_offset=i] = (
            a[unsafe_offset=i] + b[unsafe_offset=i]
        )
        i += 1
    while i < n:
        dst[unsafe_offset=i] = a[unsafe_offset=i]
        i += 1
    while i < m:
        dst[unsafe_offset=i] = b[unsafe_offset=i]
        i += 1


@export("msp_poly_sub")
def poly_sub(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, m: Int) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    var common = min(n, m)
    var i = 0
    while i + IW <= common:
        dst.unsafe_store(
            i, a.unsafe_load[width=IW](i) - b.unsafe_load[width=IW](i)
        )
        i += IW
    while i < common:
        dst[unsafe_offset=i] = (
            a[unsafe_offset=i] - b[unsafe_offset=i]
        )
        i += 1
    while i < n:
        dst[unsafe_offset=i] = a[unsafe_offset=i]
        i += 1
    while i < m:
        dst[unsafe_offset=i] = -b[unsafe_offset=i]
        i += 1


@export("msp_poly_mul")
def poly_mul(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, m: Int) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    for k in range(n + m - 1):
        var lo = max(0, k - (m - 1))
        var hi = min(n - 1, k)
        var total = Int64(0)
        for i in range(lo, hi + 1):
            total += a[unsafe_offset=i] * b[unsafe_offset=k - i]
        dst[unsafe_offset=k] = total


@no_inline
def poly_pow_impl(
    base_addr: Int,
    dst_addr: Int,
    work_addr: Int,
    n: Int,
    exponent: Int,
    capacity: Int,
):
    var base = ip(base_addr)
    var dst = ip(dst_addr)
    var work = ip(work_addr)
    var i = 0
    while i + IW <= capacity:
        dst.unsafe_store(i, SIMD[DType.int64, IW](0))
        work.unsafe_store(i, SIMD[DType.int64, IW](0))
        i += IW
    while i < capacity:
        dst[unsafe_offset=i] = 0
        work[unsafe_offset=i] = 0
        i += 1
    dst[unsafe_offset=0] = 1
    var current = 1

    for _ in range(exponent):
        var next_size = current + n - 1
        var z = 0
        while z + IW <= next_size:
            work.unsafe_store(z, SIMD[DType.int64, IW](0))
            z += IW
        while z < next_size:
            work[unsafe_offset=z] = 0
            z += 1
        for source_i in range(current):
            var scale = SIMD[DType.int64, IW](dst[unsafe_offset=source_i])
            var j = 0
            while j + IW <= n:
                var updated = work.unsafe_load[width=IW](source_i + j) + (
                    scale * base.unsafe_load[width=IW](j)
                )
                work.unsafe_store(source_i + j, updated)
                j += IW
            while j < n:
                work[unsafe_offset=source_i + j] += (
                    dst[unsafe_offset=source_i] * base[unsafe_offset=j]
                )
                j += 1
        i = 0
        while i + IW <= next_size:
            dst.unsafe_store(i, work.unsafe_load[width=IW](i))
            i += IW
        while i < next_size:
            dst[unsafe_offset=i] = work[unsafe_offset=i]
            i += 1
        current = next_size


@export("msp_poly_pow")
def poly_pow(
    base_addr: Int,
    dst_addr: Int,
    work_addr: Int,
    n: Int,
    exponent: Int,
    capacity: Int,
) abi("C"):
    poly_pow_impl(base_addr, dst_addr, work_addr, n, exponent, capacity)


@export("msp_poly_derivative")
def poly_derivative(
    src_addr: Int, dst_addr: Int, n: Int, order: Int
) abi("C") -> Int:
    var src = ip(src_addr)
    var dst = ip(dst_addr)
    var result_n = max(1, n - order)
    if order >= n:
        dst[unsafe_offset=0] = 0
        return 1
    for i in range(order, n):
        var factor = Int64(1)
        for j in range(order):
            factor *= Int64(i - j)
        dst[unsafe_offset=i - order] = src[unsafe_offset=i] * factor
    return result_n


@export("msp_poly_eval_i64")
def poly_eval_i64(src_addr: Int, n: Int, x: Int64) abi("C") -> Int64:
    var src = ip(src_addr)
    var value = Int64(0)
    for rev in range(n):
        value = value * x + src[unsafe_offset=n - 1 - rev]
    return value


@export("msp_mat_add_i64")
def mat_add_i64(
    a_addr: Int, b_addr: Int, dst_addr: Int, size: Int, subtract: Int
) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    var i = 0
    if subtract != 0:
        while i + IW <= size:
            dst.unsafe_store(
                i, a.unsafe_load[width=IW](i) - b.unsafe_load[width=IW](i)
            )
            i += IW
        while i < size:
            dst[unsafe_offset=i] = (
                a[unsafe_offset=i] - b[unsafe_offset=i]
            )
            i += 1
    else:
        while i + IW <= size:
            dst.unsafe_store(
                i, a.unsafe_load[width=IW](i) + b.unsafe_load[width=IW](i)
            )
            i += IW
        while i < size:
            dst[unsafe_offset=i] = (
                a[unsafe_offset=i] + b[unsafe_offset=i]
            )
            i += 1


@export("msp_mat_mul_i64_rows")
def mat_mul_i64_rows(
    a_addr: Int,
    b_addr: Int,
    dst_addr: Int,
    rows: Int,
    inner: Int,
    cols: Int,
    row_start: Int,
    row_stop: Int,
) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    for r in range(row_start, row_stop):
        for j in range(cols):
            dst[unsafe_offset=r * cols + j] = 0
        for k in range(inner):
            var scale = SIMD[DType.int64, IW](
                a[unsafe_offset=r * inner + k]
            )
            var j = 0
            while j + IW <= cols:
                var updated = dst.unsafe_load[width=IW](r * cols + j) + (
                    scale * b.unsafe_load[width=IW](k * cols + j)
                )
                dst.unsafe_store(r * cols + j, updated)
                j += IW
            while j < cols:
                dst[unsafe_offset=r * cols + j] += (
                    a[unsafe_offset=r * inner + k]
                    * b[unsafe_offset=k * cols + j]
                )
                j += 1


@export("msp_mat_mul_f64_rows")
def mat_mul_f64_rows(
    a_addr: Int,
    b_addr: Int,
    dst_addr: Int,
    rows: Int,
    inner: Int,
    cols: Int,
    row_start: Int,
    row_stop: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    for r in range(row_start, row_stop):
        for j in range(cols):
            dst[unsafe_offset=r * cols + j] = 0.0
        for k in range(inner):
            var scale = SIMD[DType.float64, FW](
                a[unsafe_offset=r * inner + k]
            )
            var j = 0
            while j + FW <= cols:
                var updated = dst.unsafe_load[width=FW](r * cols + j) + (
                    scale * b.unsafe_load[width=FW](k * cols + j)
                )
                dst.unsafe_store(r * cols + j, updated)
                j += FW
            while j < cols:
                dst[unsafe_offset=r * cols + j] += (
                    a[unsafe_offset=r * inner + k]
                    * b[unsafe_offset=k * cols + j]
                )
                j += 1


@export("msp_mat_transpose_i64")
def mat_transpose_i64(
    src_addr: Int, dst_addr: Int, rows: Int, cols: Int
) abi("C"):
    var src = ip(src_addr)
    var dst = ip(dst_addr)
    for r in range(rows):
        for c in range(cols):
            dst[unsafe_offset=c * rows + r] = (
                src[unsafe_offset=r * cols + c]
            )


@export("msp_det_bareiss_i64")
def det_bareiss_i64(matrix_addr: Int, n: Int) abi("C") -> Int64:
    var a = ip(matrix_addr)
    if n == 1:
        return a[unsafe_offset=0]
    var sign = Int64(1)
    var previous = Int64(1)
    for k in range(n - 1):
        var pivot_row = k
        while (
            pivot_row < n
            and a[unsafe_offset=pivot_row * n + k] == 0
        ):
            pivot_row += 1
        if pivot_row == n:
            return 0
        if pivot_row != k:
            for j in range(n):
                var temp = a[unsafe_offset=k * n + j]
                a[unsafe_offset=k * n + j] = (
                    a[unsafe_offset=pivot_row * n + j]
                )
                a[unsafe_offset=pivot_row * n + j] = temp
            sign = -sign
        var pivot = a[unsafe_offset=k * n + k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                var numerator = a[unsafe_offset=i * n + j] * pivot - (
                    a[unsafe_offset=i * n + k]
                    * a[unsafe_offset=k * n + j]
                )
                a[unsafe_offset=i * n + j] = (
                    numerator if k == 0 else numerator // previous
                )
            a[unsafe_offset=i * n + k] = 0
        previous = pivot
    return sign * a[unsafe_offset=(n - 1) * n + n - 1]
