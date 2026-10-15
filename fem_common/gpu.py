"""Optional cuDSS sparse solves shared by FEM solvers."""
from __future__ import annotations
from contextlib import contextmanager

from scipy.sparse.linalg import LinearOperator


def _runtime():
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError("The gpu backend requires CuPy; install the cuda12 or cuda13 extra.") from exc
    from ._cudss import CudssFactor
    return cp, csr_matrix, CudssFactor


def is_available():
    """Check device, CuPy and cuDSS readiness with a tiny sparse solve."""
    try:
        from scipy.sparse import eye
        import numpy as np
        result = solve_sparse(eye(1, format="csr", dtype=complex), np.ones(1, dtype=complex))
        return bool(np.allclose(result, 1.0))
    except Exception:
        # This probe handles missing/incompatible drivers and optional libraries.
        # Actual solver calls remain outside this fallback boundary.
        return False


def solve_sparse(matrix, rhs):
    """Factor and solve on CUDA, returning a host vector; no CPU fallback."""
    cp, csr_matrix, CudssFactor = _runtime()
    with CudssFactor(csr_matrix(matrix)) as factor:
        return cp.asnumpy(factor.solve(cp.asarray(rhs)))


@contextmanager
def generalized_operator(a, b, shift):
    """Host ARPACK operator with cuDSS factorization and solves on the GPU."""
    cp, csr_matrix, CudssFactor = _runtime()
    a_gpu, b_gpu = csr_matrix(a), csr_matrix(b)
    with CudssFactor(a_gpu - shift * b_gpu) as factor:
        def apply(vector):
            return cp.asnumpy(factor.solve(b_gpu @ cp.asarray(vector)))
        yield LinearOperator(a.shape, matvec=apply, dtype=complex)


@contextmanager
def quadratic_operator(a0, a1, a2, shift):
    """Factor only Q(shift); eliminate the companion block on the GPU."""
    cp, csr_matrix, CudssFactor = _runtime()
    n = a0.shape[0]
    coupling = csr_matrix(a1 + shift * a2)
    mass = csr_matrix(a2)
    with CudssFactor(csr_matrix(a0 + shift * a1 + shift**2 * a2)) as factor:
        def apply(vector):
            device = cp.asarray(vector)
            first = factor.solve(-(coupling @ device[:n] + mass @ device[n:]))
            return cp.asnumpy(cp.concatenate((first, device[:n] + shift * first)))
        yield LinearOperator((2*n, 2*n), matvec=apply, dtype=complex)
