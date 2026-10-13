"""Optional real-device checks; CPU-only installations skip these tests."""
import os

import numpy as np
import pytest
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from periodic_eigensolver import solve_generalized


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
    from periodic_eigensolver._cudss import CudssFactor
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
    result = solve_generalized(a, b, sigma=10.2-.01j, num_modes=3,
                               ncv=32, backend="cuda", tol=1e-10)
    assert result.converged and result.backend == "cuda"
    np.testing.assert_allclose(np.sort_complex(result.eigenvalues), values[[8,9,10]], atol=1e-9)
    assert np.max(result.residuals) < 1e-10
    assert not result.eigenvectors.flags.writeable


def test_cuda_repeated_root_retains_independent_vectors(cuda):
    diagonal = np.r_[1., 1., np.arange(2., 31.)]
    a, b = diags(diagonal, format="csc"), eye(diagonal.size, format="csc")
    result = solve_generalized(a,b,sigma=1.1,num_modes=2,ncv=24,
                               backend="cuda",tol=1e-9)
    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, 1., atol=1e-8)
    assert np.linalg.matrix_rank(result.eigenvectors, tol=1e-5) == 2


def test_cuda_nonconvergence_is_reported(cuda):
    a = diags(np.arange(1., 201.), format="csc")
    result = solve_generalized(a,eye(200,format="csc"),sigma=.5,
        num_modes=3,ncv=6,max_restarts=0,tol=1e-14,backend="cuda")
    assert not result.converged
    assert result.steps == 6 and result.restarts == 0
    assert np.all(np.isfinite(result.residuals))
