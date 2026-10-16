"""Optional real-device checks; CPU-only installations skip these tests."""
import os

import numpy as np
import pytest
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from fem.common.linalg import generalized_eigenpairs, quadratic_eigenpairs


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
    from fem.common._cudss import CudssFactor
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


def test_cuda_repeated_roots_retain_independent_vectors(cuda):
    diagonal = np.r_[1., 1., np.arange(2., 31.)]
    values, vectors, _ = generalized_eigenpairs(
        diags(diagonal), eye(diagonal.size), target=1.1, count=2,
        ncv=24, tolerance=1e-9, backend="gpu")
    np.testing.assert_allclose(values, 1., atol=1e-8)
    assert np.linalg.matrix_rank(vectors, tol=1e-5) == 2
    for value, vector in zip(values, vectors.T):
        assert np.linalg.norm((diagonal - value) * vector) < 1e-8


def test_cuda_iteration_budget_reports_nonconvergence(cuda):
    from fem.common.errors import SolverError
    with pytest.raises(SolverError, match="did not converge"):
        generalized_eigenpairs(diags(np.arange(1., 201.)), eye(200),
                               target=.5, count=3, ncv=6, max_iterations=6,
                               tolerance=1e-14, backend="gpu")


def test_cuda_quadratic_complex_singular_mass_without_host_arpack(cuda, monkeypatch):
    """Validate the device QEP path against independent dense generalized QZ."""
    from scipy.linalg import eigvals
    from scipy.sparse import bmat, csr_matrix
    import fem.common.linalg as shared

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU eigensolving must not invoke host ARPACK")
    monkeypatch.setattr(shared, "eigs", forbidden)
    rng = np.random.default_rng(41)
    n = 12
    a0 = csr_matrix(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    a1 = csr_matrix(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    mass = np.ones(n)
    mass[-2:] = 0
    a2 = diags(mass)
    zero, identity = csr_matrix((n, n), dtype=complex), eye(n, dtype=complex)
    reference = eigvals(bmat(((zero, identity), (-a0, -a1))).toarray(),
                        bmat(((identity, zero), (zero, a2))).toarray())
    target = .7 - .2j
    reference = reference[np.isfinite(reference)]
    expected = reference[np.argsort(abs(reference - target))[:3]]
    values, vectors, _ = quadratic_eigenpairs(
        a0, a1, a2, target=target, count=3, tolerance=1e-10, backend="gpu")
    for value in expected:
        assert np.min(abs(values - value)) < 1e-8
    for value, vector in zip(values, vectors.T):
        terms = [a0 @ vector, value * (a1 @ vector), value**2 * (a2 @ vector)]
        assert np.linalg.norm(sum(terms)) / sum(np.linalg.norm(t) for t in terms) < 1e-10
