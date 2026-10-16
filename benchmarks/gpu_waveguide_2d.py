"""Measure validated 2D waveguide CPU/GPU solves on identical assembled meshes.

Run with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for a reproducible
single-threaded BLAS baseline. Meshing/assembly and field validation run on CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment

from fem import ModeSolver2D, materials, shapes


CASES = ('pec', 'pec_lossy', 'pec_post', 'pec_sibc', 'pec_sibc_lossy')
OUTPUT = Path(__file__).resolve().parents[1] / 'outputs/benchmarks/gpu_waveguide_2d'


def make_solver(case, level, order):
    lossy = 'lossy' in case
    epsilon = 2.0 - 0.02j if lossy else 1.0
    solver = ModeSolver2D(
        frequency=299_792_458.0, x_range=1.0, y_range=0.5,
        boundary=materials.PEC, background_material=materials.Material(epsilon=epsilon),
    )
    if 'sibc' in case or case == 'pec_post':
        material = (materials.SurfaceImpedance(impedance=20 + 20j)
                    if 'sibc' in case else materials.PEC)
        solver.add_geometry(
            name='post', shape=shapes.Rectangle(bounds=((0.56, 0.68), (0.11, 0.23))),
            material=material,
        )
    start = perf_counter()
    solver.mesh(resolution=(level + 1, level // 2 + 1), element_order=order,
                material_aware=False, interface_refinement=None, boundary_refinement=None)
    assembly_seconds = perf_counter() - start
    target = complex(np.sqrt(epsilon - 0.25)) if lossy else 0.82
    return solver, target, assembly_seconds


def solve(solver, target, backend, modes):
    return solver.solve(
        backend=backend, num_modes=modes, neff_guess=target, max_refinements=0,
        dense_linearization_limit=4, eigensolver_tolerance=1e-10,
        residual_tolerance=1e-8, divergence_tolerance=1e-6,
    )


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', nargs='+', type=int, default=[12, 24, 48, 80])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--modes', type=int, default=1)
    parser.add_argument('--order', type=int, choices=[1, 2], default=1)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.levels) < 6 or args.repeats < 1 or args.modes < 1:
        parser.error('levels >= 6, repeats >= 1 and modes >= 1 are required')
    args.output.mkdir(parents=True, exist_ok=True)

    # Require actual GPU execution; never time an automatic CPU fallback.
    import cupy as cp
    from fem.common.gpu import solve_sparse
    from scipy.sparse import eye
    start = perf_counter()
    solve_sparse(eye(1, format='csr', dtype=complex), np.ones(1, dtype=complex))
    probe_seconds = perf_counter() - start
    warm, target, _ = make_solver('pec', 8, args.order)
    for backend in ('cpu', 'gpu'):
        solve(warm, target, backend, args.modes)
    metadata = dict(
        python=platform.python_version(), platform=platform.platform(),
        processor=platform.processor(), numpy=np.__version__, scipy=scipy.__version__,
        cupy=cp.__version__, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), gpu_probe_seconds=probe_seconds,
        blas_thread_environment={k: os.environ.get(k) for k in
                                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        settings=vars(args) | {'output': str(args.output)},
        methodology='Warm runtime and per-configuration untimed solves; fresh factorization '
                     'per solve; alternating backend order; '
                     'same assembled mesh; CPU ARPACK / device GPU refined Arnoldi; CPU field validation for both; '
                     'total = mesh/assembly + median public solve time; no adaptive refinement.',
    )
    try:
        from threadpoolctl import threadpool_info
        metadata['threadpools'] = threadpool_info()
    except ImportError:
        pass
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    samples, summary = [], []
    for case in args.cases:
        for level in sorted(set(args.levels)):
            solver, target, assembly = make_solver(case, level, args.order)
            print(f'Start {case}, level={level}, order={args.order}, '
                  f'DOF={solver.system.ndofs}', flush=True)
            # Compile size/dtype-specific CUDA kernels outside measured samples.
            for backend in ('cpu', 'gpu'):
                solve(solver, target, backend, args.modes)
            times = {'cpu': [], 'gpu': []}
            results = {}
            differences = []
            for repeat in range(args.repeats):
                for backend in (('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')):
                    cp.cuda.get_current_stream().synchronize()
                    start = perf_counter()
                    result = solve(solver, target, backend, args.modes)
                    cp.cuda.get_current_stream().synchronize()
                    elapsed = perf_counter() - start
                    if any(m.neff.imag > 1e-8 for m in result):
                        raise RuntimeError('A passive case returned a growing forward mode')
                    times[backend].append(elapsed)
                    results[backend] = np.array([mode.neff for mode in result])
                    samples.append(dict(
                        case=case, level=level, dofs=solver.system.ndofs,
                        backend=backend, repeat=repeat, solve_seconds=elapsed,
                        max_residual=max(m.residual for m in result),
                        max_divergence=max(m.divergence_residual for m in result),
                        neff=json.dumps([[m.neff.real, m.neff.imag] for m in result]),
                    ))
                cost = abs(results['cpu'][:, None] - results['gpu'][None, :])
                left, right = linear_sum_assignment(cost)
                differences.append(float(np.max(cost[left, right] / np.maximum(
                    abs(results['cpu'][left]), 1e-12))))
            cpu, gpu = (statistics.median(times[b]) for b in ('cpu', 'gpu'))
            row = dict(
                case=case, level=level, order=args.order, dofs=solver.system.ndofs,
                companion_dofs=2 * solver.system.ndofs, elements=len(solver.mesh_data.elements),
                mesh_assembly_seconds=assembly, cpu_seconds=cpu, gpu_seconds=gpu,
                solve_speedup=cpu / gpu, total_speedup=(assembly + cpu) / (assembly + gpu),
                cpu_min_seconds=min(times['cpu']), cpu_max_seconds=max(times['cpu']),
                gpu_min_seconds=min(times['gpu']), gpu_max_seconds=max(times['gpu']),
                max_relative_neff_difference=max(differences),
            )
            summary.append(row)
            write_csv(args.output / 'samples.csv', samples)
            write_csv(args.output / 'summary.csv', summary)
            print(f'{case:16} DOF={row["dofs"]:7} CPU={cpu:.3f}s GPU={gpu:.3f}s '
                  f'speedup={cpu/gpu:.2f}x total={row["total_speedup"]:.2f}x '
                  f'neff difference={max(differences):.2e}', flush=True)
            if max(differences) > 1e-6:
                raise RuntimeError('CPU/GPU modes disagree; inspect samples.csv')
    fig = Figure(figsize=(11, 4.5))
    axes = fig.subplots(1, 2)
    for case in args.cases:
        selected = [r for r in summary if r['case'] == case]
        for ax, key in zip(axes, ('solve_speedup', 'total_speedup')):
            ax.semilogx([r['dofs'] for r in selected], [r[key] for r in selected], 'o-', label=case)
    for ax, title in zip(axes, ('Public solve', 'Mesh + assembly + solve')):
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set(xlabel='Reduced physical DOF (companion size = 2 × DOF)',
               ylabel='CPU time / GPU time', title=title)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'speedup.png', dpi=160)
    print(f'Report: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
