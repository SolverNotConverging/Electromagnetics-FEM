"""CPU ARPACK/refined vs CUDA/cuDSS, with synchronized end-to-end timings.

Matrices are assembled once outside timings; GPU upload, cuDSS analysis and
factorization, Arnoldi, extraction, and output download are all included.
Warmups are excluded. Results must converge and match before speedups count.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
import traceback
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import scipy
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import LinearOperator, eigs, splu
from scipy.optimize import linear_sum_assignment
from periodic_eigensolver import solve_generalized, native_backend_available


def stencil(side):
    """3D complex, nonsymmetric pencil with nonidentity positive mass."""
    t = diags((-1.0, 2.0, -0.97), (-1, 0, 1), shape=(side, side), format="csc")
    identity = eye(side, format="csc")
    plane = kron(identity, t) + kron(t, identity)
    a = kron(identity, plane) + kron(t, eye(side*side, format="csc"))
    indices = np.arange(side**3)
    a = (a + diags(0.001*np.sin(indices*.61) + .02j)).astype(np.complex128).tocsc()
    b = diags(1.0 + .05*np.cos(indices*.31), format="csc", dtype=np.complex128)
    return a, b, -.05+.01j, side**3


def antenna(size):
    import runpy
    from fem_periodic_modes.src.assembly_3d import linearized_pencil_3d
    example = runpy.run_path(str(ROOT / "fem_periodic_modes/examples/3d_leaky_wave_antenna.py"))
    solver = example["build_solver"]()
    solver.mesh(max_element_size=size*1e-3, wavelength_elements=4)
    a, b = linearized_pencil_3d(solver.system)
    return a, b, .5-.02j, solver.system.ndofs


def cpu_eigs(a, b, sigma, k, ncv, tol, seed):
    lu = splu((a-sigma*b).tocsc())
    count = 0
    def apply(x):
        nonlocal count
        count += 1
        return lu.solve(b @ x)
    op = LinearOperator(a.shape, matvec=apply, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    v0 = np.ones(a.shape[0]) + 1e-3*(rng.standard_normal(a.shape[0]) + 1j*rng.standard_normal(a.shape[0]))
    mu, vectors = eigs(op, k=k, which="LM", ncv=ncv, tol=tol*.01,
                       maxiter=5000, v0=v0)
    values = sigma + 1/mu
    ax, bx = a @ vectors, b @ vectors
    residuals = np.linalg.norm(ax-bx*values[None, :], axis=0) / (
        np.linalg.norm(ax, axis=0) + abs(values)*np.linalg.norm(bx, axis=0))
    return values, residuals, bool(np.all(residuals <= tol)), count


def run(a, b, sigma, args, backend):
    if backend == "cpu_eigs":
        return cpu_eigs(a, b, sigma, args.modes, args.ncv, args.tol, args.seed)
    result = solve_generalized(a, b, sigma=sigma, num_modes=args.modes,
        ncv=args.ncv, tol=args.tol, max_restarts=args.max_restarts,
        random_seed=args.seed, backend="cuda" if backend == "cuda_cudss" else "auto")
    return result.eigenvalues, result.residuals, result.converged, result.steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("stencil", "antenna"), default="stencil")
    parser.add_argument("--sides", type=int, nargs="+", default=[8, 12, 20, 28])
    parser.add_argument("--mesh-mm", type=float, nargs="+", default=[4., 3., 2.])
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-restarts", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "build/cuda-benchmark")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # Put the runtime compiler cache and temporary files in the output folder.
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str((args.output / "cupy-cache").resolve()))
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    metadata = dict(gpu=props["name"].decode(), compute_capability=[props["major"], props["minor"]],
        gpu_memory_bytes=props["totalGlobalMem"], cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
        platform=platform.platform(), cpu=platform.processor(), numpy=np.__version__,
        scipy=scipy.__version__, cpu_refined_backend="cython" if native_backend_available() else "python",
        cudss_library=os.environ.get("CUDSS_LIBRARY"),
        options={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        includes="upload, analysis, factorization, Arnoldi, extraction, download; excludes assembly and warmup",
        cpu_threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    results = []
    backends = ["cpu_eigs", "cpu_refined", "cuda_cudss"]
    sizes = args.sides if args.problem == "stencil" else args.mesh_mm
    for size in sizes:
        print(f"Assembling {args.problem} size {size}...", flush=True)
        a,b,sigma,field_dofs = stencil(size) if args.problem == "stencil" else antenna(size)
        times = {key: [] for key in backends}
        outcomes, errors = {}, {}
        # Warm every path at each size, including JIT and cuDSS initialization.
        for backend in backends:
            try:
                outcomes[backend] = run(a,b,sigma,args,backend)
                cp.cuda.get_current_stream().synchronize()
            except Exception as exc:
                errors[backend] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                print(f"  {backend}: {errors[backend]}", flush=True)
        for repeat in range(args.repeats):
            order = backends[repeat % 3:] + backends[:repeat % 3]
            for backend in order:
                if backend in errors: continue
                cp.cuda.get_current_stream().synchronize()
                started = perf_counter()
                try:
                    outcomes[backend] = run(a,b,sigma,args,backend)
                    cp.cuda.get_current_stream().synchronize()
                    times[backend].append(perf_counter()-started)
                except Exception as exc:
                    errors[backend] = f"{type(exc).__name__}: {exc}"
        reference = outcomes.get("cpu_eigs")
        for backend in backends:
            outcome = outcomes.get(backend)
            mismatch = None
            if reference is not None and outcome is not None:
                distances = abs(reference[0][:, None]-outcome[0][None, :])
                left,right = linear_sum_assignment(distances)
                mismatch = float(np.max(distances[left,right] / np.maximum(1.,abs(reference[0][left]))))
            valid = backend not in errors and outcome is not None and outcome[2] and mismatch is not None and mismatch < 1e-5
            elapsed = median(times[backend]) if times[backend] else None
            row = dict(problem=args.problem,size=size,field_dofs=field_dofs,pencil_dofs=a.shape[0],
                nnz_a=a.nnz,nnz_b=b.nnz,backend=backend,median_seconds=elapsed,
                min_seconds=min(times[backend]) if times[backend] else None,
                max_seconds=max(times[backend]) if times[backend] else None,
                samples_seconds=json.dumps(times[backend]),
                max_residual=float(np.max(outcome[1])) if outcome is not None else None,
                eigenvalue_error=mismatch,converged=bool(outcome[2]) if outcome else False,
                valid=bool(valid),steps=outcome[3] if outcome else None,
                speedup_vs_eigs=median(times['cpu_eigs'])/elapsed if valid and elapsed and times['cpu_eigs'] else None,
                error=errors.get(backend,""))
            results.append(row)
            print(f"  {backend}: DOFs={a.shape[0]}, seconds={elapsed}, valid={valid}, residual={row['max_residual']}", flush=True)
        # Save incrementally so long server runs retain completed sizes.
        with (args.output / "results.csv").open("w", newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        (args.output / "results.json").write_text(json.dumps(dict(metadata=metadata,results=results),indent=2))
        cp.get_default_memory_pool().free_all_blocks()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(7,4.5))
    for backend in backends:
        selected=[r for r in results if r['backend']==backend and r['valid']]
        axis.loglog([r['pencil_dofs'] for r in selected],[r['median_seconds'] for r in selected],"o-",label=backend)
    axis.set(xlabel="Eigensolver pencil DOFs",ylabel="Median solve time (s)",title=metadata['gpu'])
    axis.grid(True,which="both",alpha=.25);axis.legend();fig.tight_layout()
    fig.savefig(args.output / "timings.png",dpi=160)
    if any(not row["valid"] for row in results):
        raise SystemExit("Some solves failed validation; see results.csv and results.json.")


if __name__ == "__main__":
    main()
