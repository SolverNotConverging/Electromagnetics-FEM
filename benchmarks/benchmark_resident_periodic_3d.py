"""Compare the complete warmed 3D periodic hybrid and resident GPU solve paths.

Includes physical filters, field normalization/reconstruction, interface-jump
estimation and final host result export. Assembly is excluded; resident setup
and first-use solve times are recorded separately. Neither path reuses factors.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from gpu_periodic_3d import make_solver


def solve_path(solver, resident, **options):
    """Compare hybrid and resident Ritz with the same GPU postprocessing."""
    return solver.solve(**(options | dict(backend='gpu' if resident else 'hybrid')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes-mm', type=float, nargs='+', default=[.9, .6])
    parser.add_argument('--case', choices=['iris_pec', 'iris_lossy'], default='iris_pec')
    parser.add_argument('--modes', type=int, default=2)
    parser.add_argument('--ncv', type=int, default=96)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmarks/resident_periodic_3d'))
    args = parser.parse_args()
    if any(not np.isfinite(s) or s <= 0 for s in args.sizes_mm) or args.repeats < 1:
        parser.error('positive finite sizes and repeats are required')
    args.output.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from fem.common.gpu import _runtime
    _runtime()
    environment = dict(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cupy=cp.__version__,
        settings=vars(args) | {'output': str(args.output)},
        threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        methodology=__doc__, note='Application control downloads are two bytes per Arnoldi cycle. '
        'cuDSS symbolic analysis and cuSOLVER pinned host workspace are library-internal CPU work.')
    (args.output/'environment.json').write_text(json.dumps(environment, indent=2))
    samples, summary = [], []
    first_solves = []
    for size in sorted(set(args.sizes_mm), reverse=True):
        solver, target, assembly = make_solver(args.case, size)
        options = dict(num_modes=args.modes, neff_guess=target, max_refinements=0,
                       ncv=args.ncv, eigensolver_tolerance=1e-8)
        print(f'Case {args.case}, {size} mm, physical DOF {solver.system.ndofs}, '
              f'companion DOF {2*solver.system.ndofs}', flush=True)
        cold, setup = {}, 0.
        for resident in (False, True):
            start = perf_counter()
            result = solve_path(solver, resident, backend='gpu', **options)
            cp.cuda.get_current_stream().synchronize()
            cold[resident] = perf_counter()-start
            if resident:
                setup = result.metadata['gpu_execution']['prepare_seconds']
            print(f'  {"resident" if resident else "hybrid"} first solve: '
                  f'{cold[resident]:.3f}s (resident setup {setup:.3f}s)', flush=True)
            first_solves.append(dict(size_mm=size, dofs=solver.system.ndofs,
                path='resident' if resident else 'hybrid', seconds=cold[resident], setup_seconds=setup))
            (args.output/'first_solves.json').write_text(json.dumps(first_solves, indent=2))
        times = {False: [], True: []}
        differences, field_errors, adaptive_errors = [], [], []
        for repeat in range(args.repeats):
            results = {}
            for resident in ((False, True) if repeat % 2 == 0 else (True, False)):
                cp.cuda.get_current_stream().synchronize()
                start = perf_counter()
                result = solve_path(solver, resident, backend='gpu', **options)
                cp.cuda.get_current_stream().synchronize()
                elapsed = perf_counter()-start
                times[resident].append(elapsed)
                results[resident] = result
                samples.append(dict(size_mm=size, physical_dofs=solver.system.ndofs,
                    path='resident' if resident else 'hybrid', repeat=repeat, seconds=elapsed,
                    max_residual=max(m.residual for m in result), max_gauss=max(m.gauss_residual for m in result),
                    adaptive_residual=result.metadata['adaptive_residual'],
                    execution=json.dumps(dict(result.metadata.get('gpu_execution', {})))))
                print(f'  {"resident" if resident else "hybrid"} repeat {repeat+1}: {elapsed:.3f}s', flush=True)
                with (args.output/'samples.csv').open('w', newline='') as stream:
                    writer=csv.DictWriter(stream, fieldnames=samples[0]); writer.writeheader();writer.writerows(samples)
            old, new = results[False], results[True]
            cost = abs(old.neff[:, None]-new.neff[None, :])
            left, right = linear_sum_assignment(cost)
            differences.append(float(max(cost[left, right])))
            field_errors.append(max(float(np.linalg.norm(old[i].coefficients-new[j].coefficients)/
                                          np.linalg.norm(old[i].coefficients)) for i,j in zip(left,right)))
            adaptive_errors.append(abs(old.metadata['adaptive_residual']-new.metadata['adaptive_residual'])/
                                   max(abs(old.metadata['adaptive_residual']), 1e-30))
            if differences[-1] > 1e-6 or field_errors[-1] > 1e-5 or adaptive_errors[-1] > 1e-6:
                raise RuntimeError(('GPU paths disagree', differences[-1], field_errors[-1], adaptive_errors[-1]))
        hybrid, resident = median(times[False]), median(times[True])
        summary.append(dict(size_mm=size, physical_dofs=solver.system.ndofs,
            companion_dofs=2*solver.system.ndofs, elements=solver.mesh_data.mesh.nelements,
            mesh_assembly_seconds=assembly, hybrid_first_seconds=cold[False], resident_first_seconds=cold[True],
            resident_setup_seconds=setup, hybrid_seconds=hybrid, resident_seconds=resident,
            warm_speedup=hybrid/resident, first_solve_speedup=cold[False]/cold[True],
            max_neff_difference=max(differences), max_relative_coefficient_error=max(field_errors),
            max_relative_adaptive_error=max(adaptive_errors)))
        for name, data in [('samples.csv', samples), ('summary.csv', summary)]:
            with (args.output/name).open('w', newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=data[0]); writer.writeheader(); writer.writerows(data)
        print(f'  WARM SPEEDUP {hybrid/resident:.2f}x; first solve {cold[False]/cold[True]:.2f}x', flush=True)
        del solver
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
