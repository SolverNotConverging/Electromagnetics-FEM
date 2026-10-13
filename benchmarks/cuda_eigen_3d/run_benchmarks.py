"""Periodic 3D unit-cell CPU/GPU sweep validating every timing sample.

Run with the project's Python environment and CUDSS_LIBRARY configured.
Each case runs in a fresh process so BLAS thread settings take effect before
NumPy is imported. Solver implementations are reused without modification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("cpu_eigs", "cpu_refined", "cuda_cudss")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def cases():
    result = []
    meshes = {"iris": (4., 3., 2.), "siw": (2., 1.4, 1.),
              "perforated_waveguide": (5., 3.5, 2.5), "dielectric_rod": (4., 3., 2.)}
    for problem, sizes in meshes.items():
        for level, size in zip(("coarse", "medium", "fine"), sizes):
            result.append(dict(name=f"{problem}-{level}", problem=problem,
                size=size, modes=4, ncv=80, tol=1e-8, threads=1, max_restarts=24, seed=0))
    for problem, sizes in meshes.items():
        result.append(dict(name=f"{problem}-medium-threads10", problem=problem,
            size=sizes[1], modes=4, ncv=80, tol=1e-8, threads=10, max_restarts=24, seed=0))
        result.append(dict(name=f"{problem}-medium-modes14", problem=problem,
            size=sizes[1], modes=14, ncv=96, tol=1e-8, threads=1, max_restarts=24, seed=0))
    for problem, sizes in meshes.items():
        for threads in (2, 4, 8, 20):
            result.append(dict(name=f"{problem}-medium-threads{threads}", problem=problem,
                size=sizes[1], modes=4, ncv=80, tol=1e-8, threads=threads, max_restarts=24, seed=0))
    for problem, size in {"iris": 1.4, "siw": .75, "perforated_waveguide": 2., "dielectric_rod": 1.7}.items():
        result.append(dict(name=f"{problem}-large", problem=problem,
            size=size, modes=4, ncv=80, tol=1e-8, threads=1, max_restarts=24, seed=0))
    return result


def worker(case, output, repeats):
    # Parent configures these before this process imports numerical libraries.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy
    import cupy as cp
    from benchmarks.cuda_eigen_3d.validate_magnitudes import magnitude_error
    from scipy.optimize import linear_sum_assignment
    from benchmarks.periodic_eigensolver.benchmark_cuda import cpu_eigs
    from benchmarks.cuda_eigen_3d.geometries import assemble
    from benchmarks.cuda_eigen_3d.thread_runtime import loaded_openblas
    from periodic_eigensolver import solve_generalized, native_backend_available

    output.mkdir(parents=True, exist_ok=True)
    record = dict(case=case, status="assembling", warmups=[], samples=[], errors={})
    result_path = output / "results.json"
    save(result_path, record)
    started = time.perf_counter()
    a, b, sigma, geometry_info = assemble(case["problem"], case["size"])
    field_dofs = geometry_info["field_dofs"]
    record.update(assembly_seconds=time.perf_counter()-started, field_dofs=field_dofs,
                  pencil_dofs=a.shape[0], nnz_a=a.nnz, nnz_b=b.nnz,
                  sigma=[sigma.real, sigma.imag], geometry=geometry_info, status="running")
    props = cp.cuda.runtime.getDeviceProperties(0)
    record["runtime"] = dict(gpu=props["name"].decode(), gpu_memory_bytes=props["totalGlobalMem"],
        cupy=cp.__version__, numpy=np.__version__, scipy=scipy.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
        cpu_refined_backend="cython" if native_backend_available() else "python",
        threads={name: os.environ.get(name) for name in THREAD_VARS},
        blas_libraries=loaded_openblas())
    reference = None

    def trial(backend, phase, repeat):
        nonlocal reference
        cp.cuda.get_current_stream().synchronize()
        begin = time.perf_counter()
        if backend == "cpu_eigs":
            values, residuals, converged, steps = cpu_eigs(
                a, b, sigma, case["modes"], case["ncv"], case["tol"], case["seed"])
            restarts = None
        else:
            solved = solve_generalized(a, b, sigma=sigma, num_modes=case["modes"],
                ncv=case["ncv"], tol=case["tol"], max_restarts=case["max_restarts"],
                random_seed=case["seed"], backend="cuda" if backend == "cuda_cudss" else "auto")
            values, residuals, converged, steps = solved.eigenvalues, solved.residuals, solved.converged, solved.steps
            restarts = solved.restarts
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - begin
        finite = bool(np.isfinite(values).all() and np.isfinite(residuals).all())
        good = bool(converged and finite and len(values) == case["modes"] and np.max(residuals) <= case["tol"])
        if phase == "warmup" and backend == "cpu_eigs" and good:
            reference = np.array(values, copy=True)
        mismatch = complex_mismatch = None
        if reference is not None and finite and len(values) == len(reference):
            distances = abs(reference[:, None] - values[None, :])
            left, right = linear_sum_assignment(distances)
            complex_mismatch = float(np.max(distances[left, right] / np.maximum(1., abs(reference[left]))))
            mismatch = magnitude_error(reference, values)
        sample = dict(backend=backend, phase=phase, repeat=repeat, seconds=elapsed,
            converged=bool(converged), max_residual=float(np.max(residuals)),
            complex_eigenvalue_error=complex_mismatch, eigenvalue_error=mismatch, valid=bool(good and mismatch is not None and mismatch < 1e-5),
            steps=int(steps), restarts=restarts,
            eigenvalues=[[float(x.real), float(x.imag)] for x in values],
            residuals=[float(x) for x in residuals])
        record["warmups" if phase == "warmup" else "samples"].append(sample)
        print(f"  {phase} {repeat} {backend}: {elapsed:.4f}s valid={sample['valid']} residual={sample['max_residual']:.3e}", flush=True)

    for phase, repeat in [("warmup", -1)] + [("timed", i) for i in range(repeats)]:
        order = list(BACKENDS) if phase == "warmup" else list(BACKENDS[repeat % 3:] + BACKENDS[:repeat % 3])
        for backend in order:
            if backend in record["errors"]:
                continue
            record["active"] = dict(backend=backend, phase=phase, repeat=repeat)
            save(result_path, record)
            try:
                trial(backend, phase, repeat)
            except Exception as exc:
                record["errors"][backend] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            save(result_path, record)
    summaries = []
    for backend in BACKENDS:
        samples = [s for s in record["samples"] if s["backend"] == backend]
        seconds = [s["seconds"] for s in samples]
        summaries.append(dict(backend=backend, valid=len(samples) == repeats and all(s["valid"] for s in samples)
                             and backend not in record["errors"],
            median_seconds=statistics.median(seconds) if seconds else None,
            min_seconds=min(seconds) if seconds else None, max_seconds=max(seconds) if seconds else None,
            samples_seconds=seconds, max_residual=max((s["max_residual"] for s in samples), default=None),
            max_eigenvalue_error=max((s["eigenvalue_error"] for s in samples if s["eigenvalue_error"] is not None), default=None),
            max_complex_eigenvalue_error=max((s["complex_eigenvalue_error"] for s in samples if s["complex_eigenvalue_error"] is not None), default=None),
            steps=[s["steps"] for s in samples], restarts=[s["restarts"] for s in samples],
            error=record["errors"].get(backend)))
    cpu = summaries[0]
    for row in summaries:
        row["speedup_vs_eigs"] = cpu["median_seconds"] / row["median_seconds"] if cpu["valid"] and row["valid"] else None
    record.update(status="complete", summaries=summaries,
                  host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record.pop("active", None)
    save(result_path, record)


def capture(command):
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", nargs="+", help="Optional case names from the predefined sweep")
    parser.add_argument("--timeout", type=float, default=900, help="Seconds allowed per complete case")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(json.loads(args.worker), args.output, args.repeats)
        return
    selected = [case for case in cases() if not args.case or case["name"] in args.case]
    if not selected or (args.case and set(args.case) - {c["name"] for c in selected}):
        parser.error("Unknown or empty case selection")
    metadata = dict(commit=capture(["git", "rev-parse", "HEAD"]).strip(),
        git_status=capture(["git", "status", "--short"]), command=sys.argv,
        platform=platform.platform(), python=sys.version, gpu_selection=args.gpu,
        cpu=capture(["lscpu"]), gpu_before=capture(["nvidia-smi"]),
        load_before=os.getloadavg(), packages=capture([sys.executable, "-m", "pip", "freeze"]),
        methodology="Sequential cases; one excluded warmup/backend/case; rotated backend order; assembly excluded; transfers, analysis, factorization, iteration and extraction included. Every timed sample checked against a converged CPU ARPACK warmup. Host RSS is cumulative across backends, not per-backend GPU peak memory.",
        cases=selected, repeats=args.repeats)
    source_paths = list((ROOT / "periodic_eigensolver/src").glob("*")) + list(Path(__file__).parent.glob("*.py"))
    metadata["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_paths if p.is_file()}
    save(args.output / "metadata.json", metadata)
    progress = dict(status="running", completed=[], total=len(selected))
    combined = []
    for index, case in enumerate(selected):
        print(f"[{index+1}/{len(selected)}] {case['name']}", flush=True)
        progress["active"] = case["name"]
        save(args.output / "progress.json", progress)
        folder = args.output / case["name"]
        env = os.environ.copy()
        env.update({key: str(case["threads"]) for key in THREAD_VARS})
        env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUPY_CACHE_DIR=str(args.output / "cupy-cache"),
                   MPLCONFIGDIR=str(args.output / "matplotlib-cache"), PYTHONUNBUFFERED="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(case),
                   "--output", str(folder), "--repeats", str(args.repeats)]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, timeout=args.timeout)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"
        elapsed = time.perf_counter()-started
        progress["completed"].append(dict(case=case["name"], exit_code=exit_code, seconds=elapsed))
        result_file = folder / "results.json"
        if result_file.exists():
            data = json.loads(result_file.read_text())
            for summary in data.get("summaries", []):
                combined.append(dict(case=case["name"], problem=case["problem"], size=case["size"],
                    modes=case["modes"], ncv=case["ncv"], tol=case["tol"], threads=case["threads"],
                    pencil_dofs=data["pencil_dofs"], field_dofs=data["field_dofs"], **summary))
        save(args.output / "summary.json", combined)
        if combined:
            with (args.output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(combined[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(combined)
        save(args.output / "progress.json", progress)
    progress.update(status="complete", active=None)
    save(args.output / "progress.json", progress)
    metadata.update(gpu_after=capture(["nvidia-smi"]), load_after=os.getloadavg())
    save(args.output / "metadata.json", metadata)
    if any(c["exit_code"] != 0 for c in progress["completed"]) or any(not r["valid"] for r in combined):
        raise SystemExit("Some cases failed or did not converge; see saved raw results.")


if __name__ == "__main__":
    main()
