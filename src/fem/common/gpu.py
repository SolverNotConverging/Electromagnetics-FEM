"""Optional cuDSS sparse solves shared by FEM solvers."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _DeviceOperator:
    shape: tuple[int, int]
    matvec: object
    residual: object
    backend: str = "gpu"


def _check_architecture(compute_capability, runtime_version):
    """Reject known unsupported device/toolkit combinations before allocating."""
    sm = int(compute_capability)
    if sm < 60:
        raise RuntimeError("cuDSS requires Pascal (compute capability 6.0) or newer; use backend='cpu'.")
    if runtime_version >= 13000 and sm < 75:
        raise RuntimeError(
            "CUDA 13 requires Turing (compute capability 7.5) or newer. "
            "For Pascal/Volta install the cuda12 extra and matching CUDA 12/cuDSS runtime, "
            "or use backend='cpu'."
        )


def _runtime():
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError("The gpu backend requires CuPy; install the cuda12 or cuda13 extra.") from exc
    _check_architecture(cp.cuda.Device().compute_capability, cp.cuda.runtime.runtimeGetVersion())
    from ._cudss import CudssFactor, configure_dll_search
    # Sparse arithmetic may compile kernels before the factor is constructed.
    # Register NVRTC's companion DLLs before that first compilation on Windows.
    configure_dll_search()
    return cp, csr_matrix, CudssFactor


def require_available():
    """Verify CUDA/cuDSS execution, raising when the runtime is unusable."""
    from scipy.sparse import eye
    import numpy as np
    result = solve_sparse(eye(1, format="csr", dtype=complex), np.ones(1, dtype=complex))
    if not np.allclose(result, 1.0):
        raise RuntimeError("The CUDA/cuDSS readiness solve returned an incorrect result.")


def is_available():
    """Check device, CuPy and cuDSS readiness with a tiny sparse solve."""
    try:
        require_available()
        return True
    except Exception:
        # Only readiness probes fall back; actual numerical failures propagate.
        return False


def solve_sparse(matrix, rhs):
    """Factor and solve on CUDA, returning a host vector; no CPU fallback."""
    cp, csr_matrix, CudssFactor = _runtime()
    with CudssFactor(csr_matrix(matrix)) as factor:
        return cp.asnumpy(factor.solve(cp.asarray(rhs)))


@contextmanager
def generalized_operator(a, b, shift):
    """Device shift-invert operator, factor and original-pencil residual."""
    cp, csr_matrix, CudssFactor = _runtime()
    a_gpu, b_gpu = csr_matrix(a), csr_matrix(b)
    with CudssFactor(a_gpu - shift * b_gpu) as factor:
        def apply(vector):
            return factor.solve(b_gpu @ vector)

        def residual(vectors, values):
            ax, bx = a_gpu @ vectors, b_gpu @ vectors
            scale = cp.linalg.norm(ax, axis=0) + cp.abs(values) * cp.linalg.norm(bx, axis=0)
            return cp.linalg.norm(ax - bx * values[None, :], axis=0) / cp.where(scale == 0, 1, scale)

        yield _DeviceOperator(a.shape, apply, residual)


@contextmanager
def quadratic_operator(a0, a1, a2, shift):
    """Factor only Q(shift); eliminate the companion block on the GPU."""
    cp, csr_matrix, CudssFactor = _runtime()
    n = a0.shape[0]
    constant, linear, mass = (csr_matrix(a) for a in (a0, a1, a2))
    coupling = linear + shift * mass
    with CudssFactor(constant + shift * linear + shift**2 * mass) as factor:
        def apply(vector):
            first = factor.solve(-(coupling @ vector[:n] + mass @ vector[n:]))
            return cp.concatenate((first, vector[:n] + shift * first))

        def residual(vectors, values):
            x = vectors[:n]
            terms = (constant @ x, (linear @ x) * values[None, :],
                     (mass @ x) * values[None, :]**2)
            scale = sum(cp.linalg.norm(term, axis=0) for term in terms)
            return cp.linalg.norm(sum(terms), axis=0) / cp.where(scale == 0, 1, scale)

        yield _DeviceOperator((2*n, 2*n), apply, residual)
