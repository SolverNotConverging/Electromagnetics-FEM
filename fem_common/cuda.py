"""Optional cuDSS sparse solves shared by FEM solvers."""
from __future__ import annotations


def solve_sparse(matrix, rhs):
    """Factor and solve on CUDA, returning a host vector; no CPU fallback."""
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError("CUDA solves require optional CuPy; install the cuda12 or cuda13 extra.") from exc
    from periodic_eigensolver._cudss import CudssFactor
    with CudssFactor(csr_matrix(matrix)) as factor:
        return cp.asnumpy(factor.solve(cp.asarray(rhs)))
