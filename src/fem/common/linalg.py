"""CPU ARPACK, hybrid GPU Arnoldi/CPU Ritz, or explicit resident GPU Arnoldi."""
from contextlib import contextmanager
import warnings

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu

from .errors import BackendCapabilityError, ConfigurationError, SolverError
from .validation import finite_complex, positive_integer, positive_real


def validate_backend(backend):
    """Validate the public execution backend."""
    if not isinstance(backend, str) or backend not in ("auto", "cpu", "hybrid", "gpu"):
        raise ConfigurationError("backend must be 'auto', 'cpu', 'hybrid' or 'gpu'.")
    return backend


def _cpu_fallback(reason):
    warnings.warn(
        f"backend='auto' is falling back to CPU: {reason}. "
        "Select backend='cpu' to request CPU execution explicitly.",
        RuntimeWarning, stacklevel=3)
    return "cpu"


def resolve_backend(backend="auto"):
    """Select a working CUDA/cuDSS runtime; warn on automatic CPU fallback."""
    validate_backend(backend)
    if backend == "cpu":
        return "cpu"
    from .gpu import is_available, require_available
    if backend == "auto":
        return "hybrid" if is_available() else _cpu_fallback("the CUDA/cuDSS runtime is unavailable")
    try:
        require_available()
    except Exception as exc:
        raise BackendCapabilityError(f"backend='{backend}' requested but CUDA/cuDSS is unavailable: {exc}") from exc
    return backend


def resolve_eigen_backend(backend="auto"):
    """Select hybrid for auto; require resident CUDA Ritz only for explicit gpu.

    Availability failures may select CPU only for auto. Numerical calls after
    selection propagate their errors without switching implementations.
    """
    selected = resolve_backend(backend)
    if selected == "cpu":
        return selected
    from ._gpu_resident_arnoldi import probe_resident_available
    from ._gpu_arnoldi import probe_hybrid_available
    try:
        (probe_resident_available if selected == "gpu" else probe_hybrid_available)()
    except Exception as exc:
        reason = f"the {'resident CUDA' if selected == 'gpu' else 'hybrid'} eigensolver is unavailable: {exc}"
        if backend == "auto":
            return _cpu_fallback(reason)
        raise BackendCapabilityError(f"backend='{backend}' requested but {reason}") from exc
    return selected


@contextmanager
def _generalized_operator(a, b, shift, backend):
    if backend in ("gpu", "hybrid"):
        from .gpu import generalized_operator
        with generalized_operator(a, b, shift) as operator:
            operator.backend = backend
            yield operator
    else:
        factor = splu(csc_matrix(a - shift * b))
        yield LinearOperator(a.shape, matvec=lambda v: factor.solve(b @ v), dtype=complex)


@contextmanager
def _quadratic_operator(a0, a1, a2, shift, backend):
    if backend in ("gpu", "hybrid"):
        from .gpu import quadratic_operator
        with quadratic_operator(a0, a1, a2, shift) as operator:
            operator.backend = backend
            yield operator
    else:
        n = a0.shape[0]
        factor = splu(csc_matrix(a0 + shift * a1 + shift**2 * a2))
        coupling = a1 + shift * a2

        def apply(v):
            first = factor.solve(-(coupling @ v[:n] + a2 @ v[n:]))
            return np.concatenate((first, v[:n] + shift * first))

        yield LinearOperator((2 * n, 2 * n), matvec=apply, dtype=complex)


def _iterate(operator, target, shift, count, tolerance, ncv, max_iterations, random_seed,
             vector_rows=None):
    from .gpu import _DeviceOperator
    if isinstance(operator, _DeviceOperator):
        from ._gpu_arnoldi import iterate, iterate_hybrid
        implementation = iterate_hybrid if operator.backend == "hybrid" else iterate
        return implementation(operator, target, shift, count, tolerance, ncv, max_iterations, random_seed,
                       vector_rows=vector_rows)
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
    backend = resolve_eigen_backend(backend)
    target = finite_complex(target, "target")
    a, b = csc_matrix(a, dtype=complex), csc_matrix(b, dtype=complex)
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ConfigurationError("A and B must be square matrices with matching shapes.")
    shift = target + 1e-6j * max(1.0, abs(target))
    with _generalized_operator(a, b, shift, backend) as operator:
        values, vectors = _iterate(operator, target, shift, count, tolerance,
                                   ncv, max_iterations, random_seed)
    return values, vectors, "cudss-hybrid-shift-invert" if backend == "hybrid" else "cudss-shift-invert" if backend == "gpu" else "sparse-shift-invert"


def quadratic_eigenpairs(a0, a1, a2, *, target, count, tolerance=1e-10, backend="auto",
                        ncv=None, max_iterations=None, random_seed=0):
    """Solve (A0 + lambda A1 + lambda² A2)x=0 by block elimination.

    Only the n-by-n polynomial is factorized, even when A2 is singular.
    Returned vectors are the physical first block of the companion vector.
    """
    backend = resolve_eigen_backend(backend)
    target = finite_complex(target, "target")
    a0, a1, a2 = (csc_matrix(a, dtype=complex) for a in (a0, a1, a2))
    if a0.shape != a1.shape or a0.shape != a2.shape or a0.shape[0] != a0.shape[1]:
        raise ConfigurationError("Quadratic coefficient matrices must be square with matching shapes.")
    shift = target + 1e-6j * max(1.0, abs(target))
    with _quadratic_operator(a0, a1, a2, shift, backend) as operator:
        values, vectors = _iterate(operator, target, shift, count, tolerance,
                                   ncv, max_iterations, random_seed, vector_rows=a0.shape[0])
    return values, vectors[:a0.shape[0]], "cudss-hybrid-shift-invert" if backend == "hybrid" else "cudss-shift-invert" if backend == "gpu" else "sparse-shift-invert"
