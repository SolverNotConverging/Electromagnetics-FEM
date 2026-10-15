"""Standard ARPACK shift-invert iteration with CPU or GPU sparse solves."""
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu

from .errors import ConfigurationError, SolverError
from .validation import finite_complex, positive_integer, positive_real


def validate_backend(backend):
    """Validate the public execution backend."""
    if not isinstance(backend, str) or backend not in ("auto", "cpu", "gpu"):
        raise ConfigurationError("backend must be 'auto', 'cpu' or 'gpu'.")
    return backend


def resolve_backend(backend="auto"):
    """Select GPU when its runtime loads successfully, otherwise SciPy CPU.

    Only automatic selection falls back. Numerical failures after selection
    propagate to the caller instead of silently changing execution backends.
    """
    validate_backend(backend)
    if backend == "auto":
        from .gpu import is_available
        return "gpu" if is_available() else "cpu"
    return backend


@contextmanager
def _generalized_operator(a, b, shift, backend):
    if backend == "gpu":
        from .gpu import generalized_operator
        with generalized_operator(a, b, shift) as operator:
            yield operator
    else:
        factor = splu(csc_matrix(a - shift * b))
        yield LinearOperator(a.shape, matvec=lambda v: factor.solve(b @ v), dtype=complex)


@contextmanager
def _quadratic_operator(a0, a1, a2, shift, backend):
    if backend == "gpu":
        from .gpu import quadratic_operator
        with quadratic_operator(a0, a1, a2, shift) as operator:
            yield operator
    else:
        n = a0.shape[0]
        factor = splu(csc_matrix(a0 + shift * a1 + shift**2 * a2))
        coupling = a1 + shift * a2

        def apply(v):
            first = factor.solve(-(coupling @ v[:n] + a2 @ v[n:]))
            return np.concatenate((first, v[:n] + shift * first))

        yield LinearOperator((2 * n, 2 * n), matvec=apply, dtype=complex)


def _iterate(operator, target, shift, count, tolerance, ncv, max_iterations, random_seed):
    size = operator.shape[0]
    if size < 3:
        raise ConfigurationError("Sparse eigensolving requires at least three unknowns.")
    count = min(positive_integer(count, "count"), size - 2)
    tolerance = positive_real(tolerance, "tolerance")
    if ncv is not None:
        ncv = positive_integer(ncv, "ncv", count + 2)
        if ncv > size:
            raise ConfigurationError("ncv must not exceed the eigenproblem size.")
    iterations = max(3000, 20 * size) if max_iterations is None else positive_integer(max_iterations, "max_iterations")
    rng = np.random.default_rng(positive_integer(random_seed, "random_seed", 0))
    initial = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    try:
        theta, vectors = eigs(operator, k=count, which="LM", v0=initial,
                              tol=tolerance, ncv=ncv, maxiter=iterations)
    except ArpackNoConvergence as exc:
        # Family solvers apply their original-physics residual filters to any
        # converged subset; an empty subset is always a numerical failure.
        if exc.eigenvalues is None or exc.eigenvectors is None or not len(exc.eigenvalues):
            raise SolverError("Sparse shift-invert iteration did not converge.") from exc
        theta, vectors = exc.eigenvalues, exc.eigenvectors
    valid = np.isfinite(theta) & (np.abs(theta) > np.finfo(float).eps)
    values = shift + 1.0 / theta[valid]
    vectors = vectors[:, valid]
    if not len(values):
        raise SolverError("Sparse shift-invert iteration returned no finite roots.")
    order = np.argsort(np.abs(values - target))
    return np.asarray(values[order]), np.asarray(vectors[:, order])


def generalized_eigenpairs(a, b, *, target, count, tolerance=1e-10, backend="auto",
                          ncv=None, max_iterations=None, random_seed=0):
    """Solve A x = lambda B x near target, including singular B."""
    backend = resolve_backend(backend)
    target = finite_complex(target, "target")
    a, b = csc_matrix(a, dtype=complex), csc_matrix(b, dtype=complex)
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ConfigurationError("A and B must be square matrices with matching shapes.")
    shift = target + 1e-6j * max(1.0, abs(target))
    with _generalized_operator(a, b, shift, backend) as operator:
        values, vectors = _iterate(operator, target, shift, count, tolerance,
                                   ncv, max_iterations, random_seed)
    return values, vectors, "cudss-shift-invert" if backend == "gpu" else "sparse-shift-invert"


def quadratic_eigenpairs(a0, a1, a2, *, target, count, tolerance=1e-10, backend="auto",
                        ncv=None, max_iterations=None, random_seed=0):
    """Solve (A0 + lambda A1 + lambda² A2)x=0 by block elimination.

    Only the n-by-n polynomial is factorized, even when A2 is singular.
    Returned vectors are the physical first block of the companion vector.
    """
    backend = resolve_backend(backend)
    target = finite_complex(target, "target")
    a0, a1, a2 = (csc_matrix(a, dtype=complex) for a in (a0, a1, a2))
    if a0.shape != a1.shape or a0.shape != a2.shape or a0.shape[0] != a0.shape[1]:
        raise ConfigurationError("Quadratic coefficient matrices must be square with matching shapes.")
    shift = target + 1e-6j * max(1.0, abs(target))
    with _quadratic_operator(a0, a1, a2, shift, backend) as operator:
        values, vectors = _iterate(operator, target, shift, count, tolerance,
                                   ncv, max_iterations, random_seed)
    return values, vectors[:a0.shape[0]], "cudss-shift-invert" if backend == "gpu" else "sparse-shift-invert"
