"""Replay 3402f85's benchmarks and compare full/reduced operators on cached pencils.

Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS to 1 before
launching Python. Run phases sequentially to avoid competing GPU workloads.
Historical sources are extracted under outputs, leaving the working tree intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigs, splu

from fem.common import gpu
from fem.common._gpu_arnoldi import iterate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/benchmarks/historical_gpu_reproduction'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical_sources(output, revision):
    snapshot = output / f'snapshot_{revision}'
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        data = subprocess.check_output(['git', 'archive', '--format=zip', revision], cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if not (snapshot / name).resolve().is_relative_to(snapshot.resolve()):
                    raise ValueError(f'Unsafe archive path: {name}')
            archive.extractall(snapshot)
    # Runtime setup only; the historical Windows loader predates NVIDIA's
    # current bin/x86_64 wheel layout. Do not edit historical numerical code.
    from fem.common._cudss import find_cudss_library
    os.environ['CUDSS_LIBRARY'] = find_cudss_library()
    gpu._runtime()
    sys.path.insert(0, str(snapshot))
    path = snapshot / 'benchmarks/periodic_eigensolver/benchmark_cuda.py'
    expected = subprocess.check_output(
        ['git', 'show', f'{revision}:benchmarks/periodic_eigensolver/benchmark_cuda.py'], cwd=ROOT)
    assert path.read_bytes().replace(b'\r\n', b'\n') == expected.replace(b'\r\n', b'\n'), \
        'Historical benchmark source changed'
    scope = runpy.run_path(str(path))
    return snapshot, scope


def exact(args, scope):
    original = scope['antenna']
    matrices = args.output / 'matrices'
    matrices.mkdir(exist_ok=True)

    def antenna(size):
        a, b, sigma, dofs = original(size)
        tag = f'antenna_{size:g}'
        save_npz(matrices / f'{tag}_a.npz', a)
        save_npz(matrices / f'{tag}_b.npz', b)
        digest = hashlib.sha256()
        for matrix in (a, b):
            for values in (matrix.data, matrix.indices, matrix.indptr):
                digest.update(values.tobytes())
        (matrices / f'{tag}.json').write_text(json.dumps(dict(
            sigma=[sigma.real, sigma.imag], field_dofs=dofs, shape=a.shape,
            nnz_a=a.nnz, nnz_b=b.nnz, matrix_sha256=digest.hexdigest()), indent=2))
        return a, b, sigma, dofs

    scope['main'].__globals__['antenna'] = antenna
    problem = args.phase.removeprefix('exact-')
    sys.argv = ['benchmark_cuda.py', '--problem', problem, '--modes', '4',
                '--ncv', '80' if problem == 'antenna' else '64', '--tol', '1e-8',
                '--max-restarts', '24', '--repeats', str(args.repeats),
                '--output', str(args.output / f'exact_{problem}')]
    scope['main']()


def relative_residuals(a, b, values, vectors):
    ax, bx = a @ vectors, b @ vectors
    scale = np.linalg.norm(ax, axis=0) + abs(values) * np.linalg.norm(bx, axis=0)
    return np.linalg.norm(ax - bx * values[None, :], axis=0) / np.where(scale == 0, 1, scale)


def compare(args, scope):
    import cupy as cp
    import periodic_eigensolver
    from fem.common._cudss import CudssFactor

    rows, profiles = [], []
    for size in args.sizes_mm:
        tag = args.output / 'matrices' / f'antenna_{size:g}'
        a, b = load_npz(f'{tag}_a.npz'), load_npz(f'{tag}_b.npz')
        info = json.loads(Path(f'{tag}.json').read_text())
        n, sigma = info['field_dofs'], complex(*info['sigma'])
        a0, a1, a2 = -a[n:, :n], -a[n:, n:], b[n:, n:]
        print(f'Comparing cached antenna {size:g} mm, {n} physical DOF', flush=True)
        cpu_stats = {}

        def factor(matrix):
            start = perf_counter()
            lu = splu(matrix.tocsc())
            cpu_stats.update(factor_seconds=perf_counter() - start,
                             factor_rows=matrix.shape[0], factor_nnz=int(lu.L.nnz + lu.U.nnz),
                             solve_seconds=0., solves=0)

            def solve(rhs):
                start = perf_counter()
                result = lu.solve(rhs)
                cpu_stats['solve_seconds'] += perf_counter() - start
                cpu_stats['solves'] += 1
                return result
            return SimpleNamespace(solve=solve)

        def reduced_cpu(profile=False):
            lu = (factor if profile else splu)((a0 + sigma*a1 + sigma**2*a2).tocsc())
            coupling = a1 + sigma*a2

            def apply(v):
                first = lu.solve(-(coupling @ v[:n] + a2 @ v[n:]))
                return np.concatenate((first, v[:n] + sigma*first))
            from scipy.sparse.linalg import LinearOperator
            operator = LinearOperator(a.shape, matvec=apply, dtype=complex)
            rng = np.random.default_rng(0)
            start = np.ones(2*n) + 1e-3*(rng.standard_normal(2*n) + 1j*rng.standard_normal(2*n))
            theta, vectors = eigs(operator, k=4, which='LM', ncv=80, tol=1e-10,
                                 maxiter=5000, v0=start)
            return sigma + 1/theta, vectors

        def full_gpu():
            with gpu.generalized_operator(a, b, sigma) as operator:
                return iterate(operator, sigma, sigma, 4, 1e-8, 80, 25*80, 0)

        def reduced_gpu():
            with gpu.quadratic_operator(a0, a1, a2, sigma) as operator:
                return iterate(operator, sigma, sigma, 4, 1e-8, 80, 25*80, 0)

        def historical_gpu():
            result = periodic_eigensolver.solve_generalized(
                a, b, sigma=sigma, num_modes=4, ncv=80, tol=1e-8,
                max_restarts=24, random_seed=0, backend='cuda')
            assert result.converged
            return result.eigenvalues, result.eigenvectors

        # Separate instrumented pass: CPU factorization and triangular solves.
        cpu_function = scope['cpu_eigs']
        with patch.dict(cpu_function.__globals__, {'splu': factor}):
            start = perf_counter()
            reference, residual, converged, steps = cpu_function(a, b, sigma, 4, 80, 1e-8, 0)
            elapsed = perf_counter() - start
        assert converged and max(residual) < 1e-8
        profiles.append(dict(size_mm=size, backend='historical_full_cpu', total_seconds=elapsed,
                             other_seconds=elapsed-cpu_stats['factor_seconds']-cpu_stats['solve_seconds'],
                             **cpu_stats))
        start = perf_counter()
        reduced_cpu(profile=True)
        elapsed = perf_counter() - start
        profiles.append(dict(size_mm=size, backend='reduced_cpu', total_seconds=elapsed,
                             other_seconds=elapsed-cpu_stats['factor_seconds']-cpu_stats['solve_seconds'],
                             **cpu_stats))

        operations = dict(historical_full_gpu=historical_gpu, restored_full_gpu=full_gpu,
                          reduced_cpu=reduced_cpu, restored_reduced_gpu=reduced_gpu)
        times = {key: [] for key in operations}
        validation = {key: dict(max_eigenvalue_difference=0., max_residual=0.)
                      for key in operations}
        for operation in operations.values():
            operation()
        for repeat in range(args.repeats):
            order = list(operations)
            order = order[repeat:] + order[:repeat]
            for name in order:
                cp.cuda.get_current_stream().synchronize()
                start = perf_counter()
                values, vectors = operations[name]()
                cp.cuda.get_current_stream().synchronize()
                times[name].append(perf_counter() - start)
                assert len(values) == 4
                distances = abs(reference[:, None] - values[None, :])
                ii, jj = linear_sum_assignment(distances)
                error = float(max(distances[ii, jj]))
                residual = float(max(relative_residuals(a, b, values, vectors)))
                if error > 1e-6 or residual > 1e-8:
                    raise RuntimeError((name, error, residual))
                validation[name]['max_eigenvalue_difference'] = max(
                    validation[name]['max_eigenvalue_difference'], error)
                validation[name]['max_residual'] = max(validation[name]['max_residual'], residual)
                print(f'  {name} repeat {repeat+1}: {times[name][-1]:.3f}s', flush=True)
        for name, samples in times.items():
            rows.append(dict(size_mm=size, physical_dofs=n, pencil_dofs=2*n, backend=name,
                             median_seconds=median(samples), min_seconds=min(samples),
                             max_seconds=max(samples), samples_seconds=json.dumps(samples),
                             **validation[name]))
        # Fenced GPU phase timings are diagnostic only, separate from medians.
        for name, operation in [('restored_full_gpu', full_gpu),
                                ('restored_reduced_gpu', reduced_gpu)]:
            phases = {3: 0., 4: 0., 1008: 0.}
            calls = {3: 0, 4: 0, 1008: 0}
            original_execute = CudssFactor._execute

            def execute(self, phase):
                self.stream.synchronize()
                start = perf_counter()
                original_execute(self, phase)
                self.stream.synchronize()
                phases[phase] += perf_counter() - start
                calls[phase] += 1
            with patch.object(CudssFactor, '_execute', execute):
                start = perf_counter()
                operation()
                elapsed = perf_counter() - start
            profiles.append(dict(size_mm=size, backend=name, total_seconds=elapsed,
                                 factor_seconds=phases[3]+phases[4],
                                 factor_rows=2*n if name == 'restored_full_gpu' else n,
                                 factor_nnz='', solve_seconds=phases[1008], solves=calls[1008],
                                 other_seconds=elapsed-sum(phases.values())))
        write_csv(args.output / 'controlled_comparison.csv', rows)
        write_csv(args.output / 'phase_profiles.csv', profiles)


def public_solve(args):
    import cupy as cp
    import fem
    from fem.periodic.assembly_3d import linearized_pencil_3d
    import fem.periodic.solver_3d as implementation

    rows, matrix_checks = [], []
    mm = 1e-3
    for size in args.sizes_mm:
        # Keep arithmetic and names identical to the historical build_solver.
        solver = fem.PeriodicModeSolver3D(
            frequency=20e9, x_range=(-9*mm, 9*mm), y_range=(-8*mm, 8*mm),
            z_range=(0., 10*mm), boundary=fem.materials.PEC)
        solver.add_box(name='dielectric_rod', x_range=(-3*mm, 3*mm),
                       y_range=(-2*mm, 2*mm), z_range=(0., 10*mm),
                       material=fem.Material(name='dielectric rod', epsilon=6.))
        solver.add_box(name='periodic_metal_patch', x_range=(-2*mm, 2*mm),
                       y_range=(2*mm, 2.2*mm), z_range=(3.5*mm, 6.5*mm),
                       material=fem.materials.PEC)
        solver.add_pml(thickness=3*mm, direction='all')
        solver.mesh(max_element_size=size*mm, wavelength_elements=4)
        a, b = linearized_pencil_3d(solver.system)
        tag = args.output / 'matrices' / f'antenna_{size:g}'
        for name, matrix in [('a', a), ('b', b)]:
            reference = load_npz(f'{tag}_{name}.npz')
            same_shape = matrix.shape == reference.shape
            error = (float(np.linalg.norm((matrix-reference).data) /
                           max(np.linalg.norm(reference.data), 1e-30)) if same_shape else None)
            matrix_checks.append(dict(size_mm=size, matrix=name, same_shape=same_shape,
                                      relative_matrix_difference=error,
                                      current_nnz=matrix.nnz, historical_nnz=reference.nnz))
        options = dict(num_modes=2, neff_guess=.5-.02j, ncv=80,
                       eigensolver_tolerance=1e-8, max_refinements=0)
        for backend in ('cpu', 'gpu'):
            solver.solve(backend=backend, **options)
        original_candidates = implementation._sparse_candidates
        original_make_mode = solver._make_mode
        for repeat in range(args.repeats):
            results = {}
            for backend in (('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')):
                stats = dict(eigenproblem_seconds=0., field_seconds=0., field_calls=0)

                def candidates(*positional, **keywords):
                    start = perf_counter()
                    result = original_candidates(*positional, **keywords)
                    stats['eigenproblem_seconds'] += perf_counter() - start
                    stats['candidate_count'] = positional[2]
                    return result

                def make_mode(*positional, **keywords):
                    start = perf_counter()
                    try:
                        return original_make_mode(*positional, **keywords)
                    finally:
                        stats['field_seconds'] += perf_counter() - start
                        stats['field_calls'] += 1

                with patch.object(implementation, '_sparse_candidates', candidates), \
                        patch.object(solver, '_make_mode', make_mode):
                    cp.cuda.get_current_stream().synchronize()
                    start = perf_counter()
                    modes = solver.solve(backend=backend, **options)
                    cp.cuda.get_current_stream().synchronize()
                    elapsed = perf_counter() - start
                results[backend] = np.array([m.neff for m in modes])
                rows.append(dict(size_mm=size, physical_dofs=solver.system.ndofs,
                                 backend=backend, repeat=repeat, total_seconds=elapsed,
                                 filter_misc_seconds=elapsed-stats['eigenproblem_seconds']-stats['field_seconds'],
                                 max_residual=max(m.residual for m in modes),
                                 max_gauss=max(m.gauss_residual for m in modes),
                                 neff=json.dumps([[m.neff.real, m.neff.imag] for m in modes]),
                                 **stats))
                print(f'Public {size:g} mm {backend}: {elapsed:.3f}s; '
                      f'eigenproblem {stats["eigenproblem_seconds"]:.3f}s, '
                      f'fields {stats["field_seconds"]:.3f}s', flush=True)
            distances = abs(results['cpu'][:, None] - results['gpu'][None, :])
            ii, jj = linear_sum_assignment(distances)
            assert max(distances[ii, jj]) < 1e-6
        write_csv(args.output / 'public_solve_samples.csv', rows)
        (args.output / 'public_matrix_checks.json').write_text(json.dumps(matrix_checks, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['exact-antenna', 'exact-stencil', 'compare', 'public'], required=True)
    parser.add_argument('--revision', default='3402f85')
    parser.add_argument('--sizes-mm', nargs='+', type=float, default=[4., 3., 2.])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.repeats < 1:
        parser.error('repeats must be positive')
    _, scope = historical_sources(args.output, args.revision)
    if args.phase.startswith('exact-'):
        exact(args, scope)
    elif args.phase == 'compare':
        compare(args, scope)
    else:
        public_solve(args)


if __name__ == '__main__':
    main()
