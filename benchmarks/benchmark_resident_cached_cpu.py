"""Control experiment: resident CUDA versus hybrid + the same batched CPU estimator.

CPU trace data are copied once from the prepared resident geometry, outside
timings. This isolates the warmed arithmetic from trace preparation/caching.
The original hybrid benchmark repeatedly rebuilds individual seam traces.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gpu_periodic_3d import make_solver
from benchmark_resident_periodic_3d import solve_path


def host_estimator(session):
    import cupy as cp
    from fem.periodic._gpu_resident import _Basis

    def basis(source):
        host = SimpleNamespace(**{name: cp.asnumpy(getattr(source, name))
                                  for name in ('dofs', 'values', 'curls', 'dx', 'epsilon', 'mu')})
        host.fields = lambda full, neff: _Basis.fields(host, full, neff, array_module=np)
        return host

    def traces(source):
        if source is None:
            return None
        left, right, cells = source
        return ((basis(left[0]), cp.asnumpy(left[1])),
                (basis(right[0]), cp.asnumpy(right[1])), cp.asnumpy(cells))
    return SimpleNamespace(volume=basis(session.volume), system=session.system,
                           h=cp.asnumpy(session.h), interior=traces(session.interior),
                           seam=traces(session.seam))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes-mm', type=float, nargs='+', default=[.9, .6])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmarks/resident_cached_cpu'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from fem.periodic._gpu_resident import ResidentSystem3D
    import fem.periodic.adaptive as adaptive
    original_run = adaptive.run_adaptive
    samples, summary = [], []
    for size in sorted(set(args.sizes_mm), reverse=True):
        solver, target, _ = make_solver('iris_pec', size)
        options = dict(num_modes=2, neff_guess=target, max_refinements=0,
                       ncv=96, eigensolver_tolerance=1e-8, backend='gpu')
        reference = solve_path(solver, True, **options)
        prepared = host_estimator(solver._gpu_resident_session)

        def estimate(result, prepared=prepared, elements=solver.mesh_data.info.elements):
            combined = np.zeros(elements)
            maximum = 0.
            for mode in result:
                indicator, residual = ResidentSystem3D.estimate(
                    prepared, mode.coefficients, mode.neff, array_module=np)
                combined = np.maximum(combined, indicator/max(indicator.sum(), np.finfo(float).tiny)*residual**2)
                maximum = max(maximum, residual)
            return combined, maximum

        def cpu_cached_run(owner, solve_once, original_estimate, refine, _estimate=estimate, **controls):
            return original_run(owner, solve_once, _estimate, refine, **controls)

        def run(resident, solver=solver, options=options, callback=cpu_cached_run):
            if resident:
                return solve_path(solver, True, **options)
            with patch.object(adaptive, 'run_adaptive', callback):
                return solve_path(solver, False, **options)

        run(False)
        times = {False: [], True: []}
        for repeat in range(args.repeats):
            for resident in ((False, True) if repeat % 2 == 0 else (True, False)):
                start = perf_counter()
                result = run(resident)
                cp.cuda.get_current_stream().synchronize()
                elapsed = perf_counter()-start
                times[resident].append(elapsed)
                err = abs(result.metadata['adaptive_residual']-reference.metadata['adaptive_residual'])
                if err > 1e-9:
                    raise RuntimeError(('Estimator disagreement', err))
                samples.append(dict(size_mm=size, dofs=solver.system.ndofs,
                    path='resident' if resident else 'hybrid_cached_cpu_estimator', repeat=repeat,
                    seconds=elapsed, adaptive_difference=err))
                print(samples[-1], flush=True)
        summary.append(dict(size_mm=size, dofs=solver.system.ndofs,
            hybrid_cached_cpu_seconds=median(times[False]), resident_seconds=median(times[True]),
            resident_speedup=median(times[False])/median(times[True])))
        for name, data in [('samples.csv', samples), ('summary.csv', summary)]:
            with (args.output/name).open('w', newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=data[0]); writer.writeheader();writer.writerows(data)
        print(summary[-1], flush=True)
        del solver, prepared, reference, run, estimate, cpu_cached_run
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    (args.output/'methodology.json').write_text(json.dumps(dict(description=__doc__,
        note='Warm timings only; host trace arrays reused unchanged. Both paths use GPU factorization/iteration.'), indent=2))


if __name__ == '__main__':
    main()
