"""Optional CUDA refined Arnoldi; sparse matrices and basis stay on device.

cuDSS handles numeric factorization and shifted solves. Only the small
Hessenberg eigendecomposition/refined SVDs run on the CPU, using exactly the
same extraction as the CPU backend (including repeated-root handling).
"""
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, eye

from ._cudss import CudssFactor, configure_dll_search


def solve_cuda(A, B, *, sigma, num_modes, tol, ncv, max_restarts, random_seed):
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError("CUDA Arnoldi requires optional CuPy; install cupy-cuda12x or cupy-cuda13x.") from exc
    from .refined import (_default_arnoldi_ncv, _refined_candidate_data,
                          _candidate_score, _result_from_candidate)
    configure_dll_search()

    A, B = csc_matrix(A, dtype=np.complex128), csc_matrix(B, dtype=np.complex128)
    if not (np.isfinite(A.data).all() and np.isfinite(B.data).all()):
        raise ValueError("A and B must contain only finite values.")
    n = A.shape[0]
    ncv = _default_arnoldi_ncv(n, num_modes, ncv)
    a_gpu, b_gpu = csr_matrix(A), csr_matrix(B)
    module = cp.RawModule(code=Path(__file__).with_name("arnoldi_cuda.cu").read_text(),
                          options=("--std=c++11",))
    project = module.get_function("project")
    subtract = module.get_function("subtract_projection")
    rng = np.random.default_rng(random_seed)
    start = cp.asarray(np.ones(n) + 1e-3 * (rng.standard_normal(n) + 1j*rng.standard_normal(n)))
    best, steps = None, 0
    with CudssFactor(a_gpu - sigma * b_gpu) as factor:
        for restart in range(max_restarts + 1):
            basis = cp.zeros((n, ncv + 1), dtype=cp.complex128, order="F")
            hbar = cp.zeros((ncv + 1, ncv), dtype=cp.complex128, order="F")
            basis[:, 0] = start / cp.linalg.norm(start)
            coefficients = cp.empty(ncv, dtype=cp.complex128)
            completed = 0
            for column in range(ncv):
                work = factor.solve(b_gpu @ basis[:, column])
                active = column + 1
                for _ in range(2):
                    project((active,), (256,), (basis, work, coefficients, np.int32(n), np.int32(active)))
                    subtract(((n + 255)//256,), (256,),
                             (basis, work, coefficients, np.int32(n), np.int32(active)))
                    hbar[:active, column] += coefficients[:active]
                norm = float(cp.linalg.norm(work).get())
                hbar[active, column] = norm
                completed = active
                if norm <= 1e-14:
                    if num_modes == 1 or active == ncv:
                        break
                    work = cp.asarray(rng.standard_normal(n) + 1j*rng.standard_normal(n))
                    for _ in range(2):
                        work -= basis[:, :active] @ (basis[:, :active].conj().T @ work)
                    norm = float(cp.linalg.norm(work).get())
                    if norm <= 1e-14:
                        break
                basis[:, active] = work / norm
            steps += completed
            if completed < num_modes:
                raise RuntimeError("CUDA Arnoldi breakdown before enough modes were extracted.")
            # Use an identity basis to obtain the small refined coefficients;
            # never copy the n-by-ncv basis to host memory.
            identity = eye(completed, dtype=np.complex128, format="csc")
            values, coefficients_cpu, _, projected = _refined_candidate_data(
                identity, identity, np.eye(completed, completed + 1, dtype=np.complex128),
                cp.asnumpy(hbar[:completed+1, :completed]), sigma, num_modes,
                kernel_backend="python")
            vectors = basis[:, :completed] @ cp.asarray(coefficients_cpu)
            vectors /= cp.linalg.norm(vectors, axis=0)[None, :]
            ax, bx = a_gpu @ vectors, b_gpu @ vectors
            scale = cp.linalg.norm(ax, axis=0) + cp.abs(cp.asarray(values)) * cp.linalg.norm(bx, axis=0)
            physical = cp.asnumpy(cp.linalg.norm(ax - bx*cp.asarray(values)[None, :], axis=0)
                                  / cp.where(scale == 0, 1, scale))
            if not np.isfinite(physical).all():
                raise RuntimeError("CUDA refined extraction produced nonfinite residuals.")
            candidate = (values, cp.asnumpy(vectors), physical, projected)
            if best is None or _candidate_score(candidate) < _candidate_score(best):
                best = candidate
            if np.all(physical <= tol):
                return _result_from_candidate(candidate, restart_count=restart,
                    step_count=steps, converged=True, backend="cuda")
            if restart < max_restarts:
                unconverged = best[2] > tol
                selected = best[1][:, unconverged] if unconverged.any() else best[1]
                start = cp.asarray(selected.sum(axis=1) + 1e-3*(rng.standard_normal(n) + 1j*rng.standard_normal(n)))
    return _result_from_candidate(best, restart_count=max_restarts,
        step_count=steps, converged=False, backend="cuda")
