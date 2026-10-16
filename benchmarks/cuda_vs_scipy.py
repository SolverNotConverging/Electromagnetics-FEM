"""Paired public-solver CUDA versus SciPy ARPACK benchmark of FEM and sibling FDFD.

Run using FEM's CUDA-enabled Python. GPU jobs must run sequentially. Results are
checkpointed after each sample. CPU/GPU use identical meshes and tolerances;
production Krylov defaults are retained except ncv=96 for FDFD periodic 2D
and FEM periodic 3D, which expose that control through their public APIs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import gc
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
from statistics import median
import sys
from time import perf_counter
import traceback
import tomllib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'FDFD/src'))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cupy as cp
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_info, threadpool_limits

import fem
import fdfd
from gpu_waveguide_2d import make_solver as fem_waveguide
from gpu_periodic_3d import make_solver as fem_periodic_3d


def configurations():
    rows = []
    for package in ('FEM', 'FDFD'):
        wave_sizes = (48, 96, 160, 256) if package == 'FEM' else (64, 128, 256, 384)
        periodic_sizes = (48, 96, 160, 320, 512) if package == 'FEM' else (64, 128, 256, 384)
        for family, sizes in (('waveguide_2d', wave_sizes), ('periodic_2d', periodic_sizes)):
            for size in sizes:
                rows.append(dict(package=package, family=family, case='pec', size=size))
            for size in sizes[1:3]:
                rows.append(dict(package=package, family=family, case='pec_lossy', size=size))
        if package == 'FEM':
            for size in (96, 160):
                rows.append(dict(package=package, family='waveguide_2d', case='pec_sibc_lossy', size=size))
        rows.append(dict(package=package, family='periodic_2d', case='dielectric_insert',
                         size=320 if package == 'FEM' else 256))
    for size in (1.6, .9, .6):
        rows.append(dict(package='FEM', family='periodic_3d', case='iris_pec', size=size))
    rows.append(dict(package='FEM', family='periodic_3d', case='iris_lossy', size=.9))
    for row in rows:
        row['id'] = '_'.join(str(row[k]) for k in ('package', 'family', 'case', 'size'))
    return rows


def make_problem(config):
    package, family, case, size = (config[k] for k in ('package', 'family', 'case', 'size'))
    eps = 2.0 - .02j if 'lossy' in case else 1.
    if package == 'FEM' and family == 'waveguide_2d':
        solver, guess, assembly = fem_waveguide(case, size, 1)
        return solver, guess, assembly
    if family == 'periodic_3d':
        return fem_periodic_3d(case, size)
    started = perf_counter()
    if family == 'periodic_2d':
        if package == 'FEM':
            solver = fem.PeriodicModeSolver2D(frequency=10e9, x_range=.02, z_range=.01,
                polarization='TE', background_material=fem.Material(epsilon=eps))
            if case == 'dielectric_insert':
                solver.add_rectangle(x_range=(.006,.014), z_range=(.004,.006),
                    material=fem.Material(epsilon=4.-.04j))
            solver.mesh(resolution=(size+1, size//2+1))
        else:
            solver = fdfd.PeriodicModeSolver2D(frequency=10e9, x_range=(0., .02), z_range=(0., .01),
                background_material=fdfd.Material(name='fill', epsilon=eps))
            if case == 'dielectric_insert':
                solver.add_rectangle(x_range=(.006,.014), z_range=(.004,.006),
                    material=fdfd.Material(name='insert', epsilon=4.-.04j))
            solver.mesh(resolution=(size, size//2), subpixels=1)
        guess = complex(np.sqrt(eps-(scipy.constants.c/(2*10e9*.02))**2))-.025
        if case == 'dielectric_insert':
            guess = .85-.005j
    else:
        nx, ny = size, size//2
        dx, dy = 1./nx, .5/ny
        outer = ((-dx, 1.+dx), (-dy, .5+dy))
        solver = fdfd.ModeSolver2D(frequency=scipy.constants.c, x_range=outer[0], y_range=outer[1],
            background_material=fdfd.Material(name='fill', epsilon=eps))
        solver.add_geometry(shape=fdfd.shapes.Difference(shape=fdfd.shapes.Rectangle(bounds=outer),
            tool=fdfd.shapes.Rectangle(bounds=((0., 1.), (0., .5)))), material=fdfd.materials.PEC)
        solver.mesh(resolution=(nx+2, ny+2), subpixels=1)
        guess = complex(np.sqrt(eps-.25))-.025
    return solver, guess, perf_counter()-started


@contextmanager
def instrument(config, stages):
    """Observe eigenvalue work without replacing algorithms or reusing factors."""
    with ExitStack() as stack:
        def observe(module_name, attribute):
            module = importlib.import_module(module_name)
            original = getattr(module, attribute)
            def call(*args, **kwargs):
                start = perf_counter()
                result = original(*args, **kwargs)
                cp.cuda.get_current_stream().synchronize()
                stages['eigen_seconds'] += perf_counter()-start
                if config['package'] == 'FDFD':
                    a = args[0]
                    stages.update(dofs=a.shape[0], eigen_dimension=a.shape[0], nnz=a.nnz,
                        candidates=kwargs['k'], ncv=kwargs.get('ncv'))
                return result
            stack.enter_context(patch.object(module, attribute, call))
        if config['package'] == 'FDFD':
            observe('fdfd.core.eigs', 'solve')
        elif config['family'] == 'waveguide_2d':
            observe('fem.waveguide.solver_2d', 'solve_qep_candidates')
        else:
            observe('fem.common.linalg', 'quadratic_eigenpairs')
            if config['family'] == 'periodic_3d':
                module = importlib.import_module('fem.periodic._gpu_resident')
                original = module.quadratic_operator
                @contextmanager
                def operator(*args, **kwargs):
                    start = perf_counter()
                    with original(*args, **kwargs) as op:
                        yield op
                    cp.cuda.get_current_stream().synchronize()
                    stages['eigen_seconds'] += perf_counter()-start
                stack.enter_context(patch.object(module, 'quadratic_operator', operator))
        yield


def run_solve(solver, guess, config, backend, modes, tol):
    stages = dict(eigen_seconds=0.)
    cp.cuda.get_current_stream().synchronize()
    with instrument(config, stages):
        started = perf_counter()
        if config['package'] == 'FEM':
            options = dict(backend=backend, num_modes=modes, neff_guess=guess,
                max_refinements=0, dense_linearization_limit=4, eigensolver_tolerance=tol,
                residual_tolerance=1e-8, divergence_tolerance=1e-6)
            if config['family'] != 'waveguide_2d':
                options.update(eigensolver='sparse')
            if config['family'] == 'periodic_2d':
                options.pop('divergence_tolerance')
            if config['family'] == 'periodic_3d':
                options.update(ncv=96)
            result = solver.solve(**options)
        else:
            options = dict(device='cuda' if backend == 'gpu' else 'cpu', method='arpack',
                num_modes=modes, guess=guess, tol=tol)
            if config['family'] == 'periodic_2d':
                options.update(polarization='TE', ncv=96)
            result = solver.solve(**options)
        cp.cuda.get_current_stream().synchronize()
        elapsed = perf_counter()-started
    if config['package'] == 'FEM':
        stages.update(dofs=solver.system.ndofs, eigen_dimension=2*solver.system.ndofs,
            nnz=sum(a.nnz for a in (solver.system.A0, solver.system.A1, solver.system.A2)))
        residual = max(m.residual for m in result)
        method = result.metadata.get('eigensolver', '')
    else:
        residual = float(max(result.info.residuals))
        method = result.info.method
        assert result.info.device == ('cuda' if backend == 'gpu' else 'cpu')
    neff = np.asarray(result.neff)
    assert len(neff) == modes and np.isfinite(neff).all()
    assert residual <= 1e-8, residual
    assert stages['eigen_seconds'] > 0
    return dict(backend=backend, public_seconds=elapsed, **stages, residual=float(residual),
        neff=[[v.real, v.imag] for v in neff], method=method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/benchmarks/cuda_vs_scipy_20261016')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--modes', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output/'results.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(results=[], failures=[])
    def save():
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    props = cp.cuda.runtime.getDeviceProperties(0)
    environment = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        cupy=cp.__version__, fem=fem.__version__, gpu=props['name'].decode(), gpu_vram=int(props['totalGlobalMem']),
        fdfd=tomllib.loads((ROOT.parent/'FDFD/pyproject.toml').read_text())['project']['version'],
        source_commits={name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
            text=True).strip() for name,path in (('FEM', ROOT), ('FDFD', ROOT.parent/'FDFD'))},
        cpu=platform.processor(), cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        methodology=__doc__, repeats=args.repeats, modes=args.modes, tolerance=args.tolerance,
        fdfd_periodic_3d='Excluded per user instruction',
        timings='public: premeshed solver.solve; eigen: factorization + iteration + eigenpair export, '
            'with original-pencil residuals in FDFD and resident FEM 3D; '
            'first CUDA per mesh includes resident geometry setup, later medians reuse static geometry only; '
            'fresh factorization for every solve. Compilation/runtime probe warmed once globally.')
    from fem.common.linalg import resolve_eigen_backend
    from fdfd.core.eigs import check_eigen_device
    started = perf_counter()
    assert resolve_eigen_backend('gpu') == 'gpu' and check_eigen_device('cuda') == 'cuda'
    environment['runtime_probe_seconds'] = perf_counter()-started
    with threadpool_limits(limits=1):
        environment['threadpools'] = threadpool_info()
        (args.output/'environment.json').write_text(json.dumps(environment, indent=2))
        for config in configurations():
            if args.ids and config['id'] not in args.ids:
                continue
            if any(r['id'] == config['id'] for r in state['results']):
                continue
            print(f"START {config['id']}", flush=True)
            samples = []
            try:
                solver, guess, assembly = make_problem(config)
                first = []
                for backend in ('cpu', 'gpu'):
                    row = run_solve(solver, guess, config, backend, args.modes, args.tolerance)
                    first.append(row)
                    print(f"  first {backend}: {row['public_seconds']:.3f}s eigen {row['eigen_seconds']:.3f}s DOF {row['dofs']}", flush=True)
                for repeat in range(args.repeats):
                    for backend in (('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')):
                        row = run_solve(solver, guess, config, backend, args.modes, args.tolerance)
                        samples.append(dict(repeat=repeat, **row))
                        state['in_progress'] = dict(config=config, assembly_seconds=assembly, first=first, samples=samples)
                        save()
                        print(f"  repeat {repeat+1} {backend}: {row['public_seconds']:.3f}s eigen {row['eigen_seconds']:.3f}s", flush=True)
                references = [np.asarray([complex(*v) for v in row['neff']]) for row in first+samples]
                differences = []
                for candidate in references[1:]:
                    cost = abs(references[0][:, None]-candidate[None, :])
                    left, right = linear_sum_assignment(cost)
                    differences.append(float(max(cost[left,right]/np.maximum(abs(references[0][left]),1e-12))))
                assert max(differences) < 1e-6, differences
                times = {b: {k: median(r[k] for r in samples if r['backend'] == b)
                         for k in ('public_seconds','eigen_seconds')} for b in ('cpu','gpu')}
                summary = dict(**config, dofs=first[0]['dofs'], eigen_dimension=first[0]['eigen_dimension'],
                    assembly_seconds=assembly, cpu_seconds=times['cpu']['public_seconds'],
                    gpu_seconds=times['gpu']['public_seconds'], cpu_eigen_seconds=times['cpu']['eigen_seconds'],
                    gpu_eigen_seconds=times['gpu']['eigen_seconds'],
                    speedup=times['cpu']['public_seconds']/times['gpu']['public_seconds'],
                    eigen_speedup=times['cpu']['eigen_seconds']/times['gpu']['eigen_seconds'],
                    total_speedup=(assembly+first[0]['public_seconds'])/(assembly+first[1]['public_seconds']),
                    mesh_plus_warm_solve_speedup=(assembly+times['cpu']['public_seconds'])/(assembly+times['gpu']['public_seconds']),
                    first_solve_speedup=first[0]['public_seconds']/first[1]['public_seconds'],
                    max_relative_neff_difference=max(differences), max_residual=max(r['residual'] for r in first+samples),
                    first=first, samples=samples)
                state['results'].append(summary)
                state.pop('in_progress', None)
                save()
                print(f"DONE {config['id']}: public {summary['speedup']:.2f}x eigen {summary['eigen_speedup']:.2f}x", flush=True)
                del solver
            except Exception:
                state['failures'].append(dict(config=config, traceback=traceback.format_exc(), samples=samples))
                save()
                print(traceback.format_exc(), flush=True)
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
