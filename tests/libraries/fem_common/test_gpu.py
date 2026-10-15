"""Optional real-device checks; CPU-only installations skip these tests."""
import os

import numpy as np
import pytest
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from fem_common.linalg import generalized_eigenpairs


@pytest.fixture(scope="module")
def cuda():
    if not os.environ.get("CUDSS_LIBRARY"):
        pytest.skip("Set CUDSS_LIBRARY to run CUDA device tests")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA device")
    return cp


def test_cudss_complex_nonsymmetric_solve(cuda):
    from cupyx.scipy.sparse import csr_matrix
    from fem_common._cudss import CudssFactor
    a = diags((-.8 + .2j, np.arange(1, 41) + 2j, -.3j),
              (-1, 0, 1), shape=(40, 40), format="csc")
    rhs = np.arange(40) + .5j
    with CudssFactor(csr_matrix(a)) as factor:
        actual = cuda.asnumpy(factor.solve(cuda.asarray(rhs)))
        np.testing.assert_allclose(actual, splu(a).solve(rhs), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("singular_mass", [False, True])
def test_cuda_generalized_known_eigenvalues(cuda, singular_mass):
    values = np.arange(1., 51.) + .03j
    mass = 1. + np.arange(50)*.01
    a = diags(values*mass, format="csc")
    if singular_mass:
        mass[-5:] = 0
    b = diags(mass, format="csc")
    actual, vectors, method = generalized_eigenpairs(
        a, b, target=10.2-.01j, count=3, ncv=32, backend="gpu", tolerance=1e-10)
    assert method == "cudss-shift-invert"
    np.testing.assert_allclose(np.sort_complex(actual), values[[8,9,10]], atol=1e-9)
    for value, vector in zip(actual, vectors.T):
        assert np.linalg.norm(a @ vector - value * (b @ vector)) < 1e-8
