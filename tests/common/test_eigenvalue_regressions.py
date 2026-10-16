"""Difficult-pencil accuracy checks retained from the pre-cleanup eigensolver.

These check original equations and invariant subspaces independently of the
Arnoldi implementation. CUDA remains optional on CPU-only installations.
"""
import os

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from scipy.sparse import csc_matrix, eye

from fem.common.linalg import generalized_eigenpairs


@pytest.fixture(params=['cpu', 'gpu'])
def backend(request):
    if request.param == 'gpu':
        if not os.environ.get('CUDSS_LIBRARY'):
            pytest.skip('Set CUDSS_LIBRARY to exercise the optional GPU regression')
        pytest.importorskip('cupy')
    return request.param


@pytest.mark.parametrize('separation,off_diagonal', [(1e-5, 1.), (5e-9, .1)])
def test_nonnormal_nearby_roots_keep_parallel_eigenvectors(backend, separation, off_diagonal):
    roots = np.r_[1., 1.+separation, np.arange(2., 30.)]
    matrix = np.diag(roots).astype(complex)
    matrix[0, 1] = off_diagonal
    values, vectors, _ = generalized_eigenpairs(csc_matrix(matrix), eye(30),
        target=.8, count=2, tolerance=1e-9, ncv=20, max_iterations=60, backend=backend)
    assert len(values) == 2
    # Near-defective eigenvalues are sensitive; test the physical residual as
    # well as the root region and legitimate nonorthogonality of these modes.
    assert np.max(abs(values-1.)) < 1e-4
    for value, vector in zip(values, vectors.T):
        av = matrix @ vector
        residual = np.linalg.norm(av-value*vector)/(np.linalg.norm(av)+abs(value)*np.linalg.norm(vector))
        assert residual < 1e-9
    assert abs(np.vdot(vectors[:, 0], vectors[:, 1])) > .99


@pytest.mark.parametrize('separation', [0., 5e-9])
def test_repeated_and_clustered_roots_retain_invariant_subspace(backend, separation):
    roots = np.r_[1., 1.+separation, np.arange(2., 30.)]
    values, vectors, _ = generalized_eigenpairs(csc_matrix(np.diag(roots)), eye(30),
        target=.8, count=2, tolerance=1e-9, ncv=20, max_iterations=60, backend=backend)
    assert len(values) == 2
    np.testing.assert_allclose(np.sort(values.real), roots[:2], atol=1e-8, rtol=0.)
    assert np.linalg.matrix_rank(vectors, tol=1e-7) == 2
    target = np.eye(30, dtype=complex)[:, :2]
    assert max(subspace_angles(target, vectors)) < 1e-7
    for value, vector in zip(values, vectors.T):
        assert np.linalg.norm((roots-value)*vector) < 1e-8
