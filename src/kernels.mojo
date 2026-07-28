"""Exact polynomial and dense matrix kernels exposed through a C ABI.

Python owns every buffer. Integer inputs are admitted only after the binding
has proved that all intermediates fit in signed 64 bits.
"""

from std.algorithm import parallelize
from std.sys.info import simd_width_of

comptime IW = simd_width_of[DType.int64]()
comptime FW = simd_width_of[DType.float64]()
comptime POW_PARALLEL_THRESHOLD = 4194304
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


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
        dst.store(i, a.load[width=IW](i) + b.load[width=IW](i))
        i += IW
    while i < common:
        dst[i] = a[i] + b[i]
        i += 1
    while i < n:
        dst[i] = a[i]
        i += 1
    while i < m:
        dst[i] = b[i]
        i += 1


@export("msp_poly_sub")
def poly_sub(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, m: Int) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    var common = min(n, m)
    var i = 0
    while i + IW <= common:
        dst.store(i, a.load[width=IW](i) - b.load[width=IW](i))
        i += IW
    while i < common:
        dst[i] = a[i] - b[i]
        i += 1
    while i < n:
        dst[i] = a[i]
        i += 1
    while i < m:
        dst[i] = -b[i]
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
            total += a[i] * b[k - i]
        dst[k] = total


@no_inline
def poly_pow_impl(
    base_addr: Int,
    dst_addr: Int,
    work_addr: Int,
    n: Int,
    exponent: Int,
    capacity: Int,
    enable_parallel: Int,
):
    var base = ip(base_addr)
    var dst = ip(dst_addr)
    var work = ip(work_addr)
    var i = 0
    while i + IW <= capacity:
        dst.store(i, SIMD[DType.int64, IW](0))
        work.store(i, SIMD[DType.int64, IW](0))
        i += IW
    while i < capacity:
        dst[i] = 0
        work[i] = 0
        i += 1
    dst[0] = 1
    var current = 1

    @parameter
    def coefficient_chunk(task: Int):
        var task_base = ip(base_addr)
        var task_dst = ip(dst_addr)
        var task_work = ip(work_addr)
        var task_count = min(16, current + n - 1)
        var begin = task * (current + n - 1) // task_count
        var end = (task + 1) * (current + n - 1) // task_count
        for k in range(begin, end):
            var lo = max(0, k - (n - 1))
            var hi = min(current - 1, k)
            var total = Int64(0)
            for j in range(lo, hi + 1):
                total += task_dst[j] * task_base[k - j]
            task_work[k] = total

    for _ in range(exponent):
        var next_size = current + n - 1
        if enable_parallel != 0 and current * n >= POW_PARALLEL_THRESHOLD:
            parallelize[coefficient_chunk](min(16, next_size))
        else:
            var z = 0
            while z + IW <= next_size:
                work.store(z, SIMD[DType.int64, IW](0))
                z += IW
            while z < next_size:
                work[z] = 0
                z += 1
            for source_i in range(current):
                var scale = SIMD[DType.int64, IW](dst[source_i])
                var j = 0
                while j + IW <= n:
                    var updated = work.load[width=IW](source_i + j) + (
                        scale * base.load[width=IW](j)
                    )
                    work.store(source_i + j, updated)
                    j += IW
                while j < n:
                    work[source_i + j] += dst[source_i] * base[j]
                    j += 1
        i = 0
        while i + IW <= next_size:
            dst.store(i, work.load[width=IW](i))
            i += IW
        while i < next_size:
            dst[i] = work[i]
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
    enable_parallel: Int,
) abi("C"):
    poly_pow_impl(
        base_addr,
        dst_addr,
        work_addr,
        n,
        exponent,
        capacity,
        enable_parallel,
    )


@export("msp_poly_derivative")
def poly_derivative(
    src_addr: Int, dst_addr: Int, n: Int, order: Int
) abi("C") -> Int:
    var src = ip(src_addr)
    var dst = ip(dst_addr)
    var result_n = max(1, n - order)
    if order >= n:
        dst[0] = 0
        return 1
    for i in range(order, n):
        var factor = Int64(1)
        for j in range(order):
            factor *= Int64(i - j)
        dst[i - order] = src[i] * factor
    return result_n


@export("msp_poly_eval_i64")
def poly_eval_i64(src_addr: Int, n: Int, x: Int64) abi("C") -> Int64:
    var src = ip(src_addr)
    var value = Int64(0)
    for rev in range(n):
        value = value * x + src[n - 1 - rev]
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
            dst.store(i, a.load[width=IW](i) - b.load[width=IW](i))
            i += IW
        while i < size:
            dst[i] = a[i] - b[i]
            i += 1
    else:
        while i + IW <= size:
            dst.store(i, a.load[width=IW](i) + b.load[width=IW](i))
            i += IW
        while i < size:
            dst[i] = a[i] + b[i]
            i += 1


@export("msp_mat_mul_i64")
def mat_mul_i64(
    a_addr: Int,
    b_addr: Int,
    dst_addr: Int,
    rows: Int,
    inner: Int,
    cols: Int,
) abi("C"):
    var a = ip(a_addr)
    var b = ip(b_addr)
    var dst = ip(dst_addr)
    for i in range(rows * cols):
        dst[i] = 0
    for r in range(rows):
        for k in range(inner):
            var scale = SIMD[DType.int64, IW](a[r * inner + k])
            var j = 0
            while j + IW <= cols:
                var updated = dst.load[width=IW](r * cols + j) + (
                    scale * b.load[width=IW](k * cols + j)
                )
                dst.store(r * cols + j, updated)
                j += IW
            while j < cols:
                dst[r * cols + j] += a[r * inner + k] * b[k * cols + j]
                j += 1


@export("msp_mat_mul_f64")
def mat_mul_f64(
    a_addr: Int,
    b_addr: Int,
    dst_addr: Int,
    rows: Int,
    inner: Int,
    cols: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    for i in range(rows * cols):
        dst[i] = 0.0
    for r in range(rows):
        for k in range(inner):
            var scale = SIMD[DType.float64, FW](a[r * inner + k])
            var j = 0
            while j + FW <= cols:
                var updated = dst.load[width=FW](r * cols + j) + (
                    scale * b.load[width=FW](k * cols + j)
                )
                dst.store(r * cols + j, updated)
                j += FW
            while j < cols:
                dst[r * cols + j] += a[r * inner + k] * b[k * cols + j]
                j += 1


@export("msp_mat_transpose_i64")
def mat_transpose_i64(
    src_addr: Int, dst_addr: Int, rows: Int, cols: Int
) abi("C"):
    var src = ip(src_addr)
    var dst = ip(dst_addr)
    for r in range(rows):
        for c in range(cols):
            dst[c * rows + r] = src[r * cols + c]


@export("msp_det_bareiss_i64")
def det_bareiss_i64(matrix_addr: Int, n: Int) abi("C") -> Int64:
    var a = ip(matrix_addr)
    if n == 1:
        return a[0]
    var sign = Int64(1)
    var previous = Int64(1)
    for k in range(n - 1):
        var pivot_row = k
        while pivot_row < n and a[pivot_row * n + k] == 0:
            pivot_row += 1
        if pivot_row == n:
            return 0
        if pivot_row != k:
            for j in range(n):
                var temp = a[k * n + j]
                a[k * n + j] = a[pivot_row * n + j]
                a[pivot_row * n + j] = temp
            sign = -sign
        var pivot = a[k * n + k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                var numerator = a[i * n + j] * pivot - (
                    a[i * n + k] * a[k * n + j]
                )
                a[i * n + j] = numerator if k == 0 else numerator // previous
            a[i * n + k] = 0
        previous = pivot
    return sign * a[(n - 1) * n + n - 1]
