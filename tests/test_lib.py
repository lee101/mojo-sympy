import numpy as np
import pytest

from mojosympy._lib import addr


def test_addr_accepts_supported_contiguous_buffers():
    assert addr(np.ones(3, dtype=np.int64)) != 0
    assert addr(np.ones(3, dtype=np.float64)) != 0


def test_addr_rejects_wrong_dtype_noncontiguous_and_empty_buffers():
    with pytest.raises(TypeError):
        addr(np.ones(3, dtype=np.int32))
    with pytest.raises(TypeError):
        addr(np.ones(6, dtype=np.int64)[::2])
    with pytest.raises(ValueError):
        addr(np.empty(0, dtype=np.int64))
