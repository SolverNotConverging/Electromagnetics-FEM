"""Time public scattering and its direct Maxwell solve on the same CPU/GPU mesh.

The direct timer includes CPU PEC condensation, fresh sparse analysis/factorization,
substitution, transfers and the physical linear residual check. Public timing also
includes repeated assembly, equivalent sources, fields, projections and the jump
estimator. Meshing and incident-mode preparation are excluded. No factors are reused.
"""
import argparse
import csv
import gc
from html import escape
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.constants import c

from fem import materials
from fem.scattering import ScatteringSolver2D


def make_solver(case, size_um):
    solver = ScatteringSolver2D(frequency=c/1.55e-6,
        x_range=(0., 1e-6), z_range=(-3e-6, 3e-6), boundary=materials.PEC)
    solver.add_rectangle(x_range=solver.x_range, z_range=(-.3e-6, .3e-6),
                         material=materials.Material(epsilon=2.-.02j))
    if case == 'matched':
        solver.set_matched_ports()
    else:
        solver.add_pml(thickness=.8e-6, direction='z', target_reflection=1e-7)
    solver.mesh(max_element_size=size_um*1e-6, wavelength_elements=6)
    solver.solve_modes(backend='cpu', num_modes=1, neff_guess=1.,
                       num_elements=64, max_refinements=0)
    solver.set_incident_mode(0, amplitude=1e-13)
    return solver


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, summary, environment):
    rows = ''.join(
        f"<tr><td>{row['case']}</td><td>{int(row['dofs']):,}</td>"
        f"<td>{float(row['cpu_direct_seconds']):.4f}</td><td>{float(row['gpu_direct_seconds']):.4f}</td>"
        f"<td>{float(row['direct_speedup']):.2f}</td><td>{float(row['cpu_public_seconds']):.3f}</td>"
        f"<td>{float(row['gpu_public_seconds']):.3f}</td><td>{float(row['public_speedup']):.2f}</td></tr>"
        for row in summary)
    maximum_error = max(float(row['max_coefficient_error']) for row in summary)
    maximum_residual = max(float(row['max_linear_residual']) for row in summary)
    html = f"""<!doctype html><meta charset="utf-8"><title>Direct scattering: cuDSS check</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#182331}}
p{{line-height:1.6}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}
td:first-child,th:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#eff3f6;padding:16px}}</style>
<h1>Direct scattering cuDSS check</h1>
<p>The existing scattering direct solver already selects cuDSS for <code>backend="gpu"</code>
and for <code>backend="auto"</code> when CUDA is available. The automatic GPU warmup was verified.
The measured 2D cases below do not show a speedup on this laptop.</p>
<p><b>GPU:</b> {escape(environment['gpu'])}. CuPy {escape(environment['cupy'])}.
CPU BLAS/OpenMP thread limits: {escape(str(environment['threads']))}.
Each row is the median of {int(environment['settings']['repeats'])} warmed alternating CPU/GPU repetitions with fresh factors.</p>
<table><tr><th>Boundary</th><th>Total DOF</th><th>CPU direct (s)</th><th>GPU direct (s)</th><th>Direct CPU/GPU</th>
<th>CPU complete (s)</th><th>GPU complete (s)</th><th>Complete CPU/GPU</th></tr>{rows}</table>
<p>Speedup is CPU/GPU; values below one mean GPU is slower. The direct stage includes CPU PEC
condensation, fresh sparse analysis/factorization, substitution, matrix/RHS uploads, coefficient
download and the linear residual check. Complete solve time also includes matrix/source assembly,
fields, modal projection and the adaptive estimator. Mesh creation and incident-mode preparation
are excluded. CPU assembly/postprocessing remains the majority of complete-solve time.</p>
<p>The model is a 1 × 6 μm PEC guide at 1.55 μm wavelength, containing a 0.6 μm slab with
relative epsilon 2 − 0.02j. Both backends use the same mesh and incident fields; assembled matrices,
sources and boundary data were checked identical. Maximum relative coefficient difference:
{maximum_error:.3e}; maximum linear residual: {maximum_residual:.3e}. S11/S21 also agree
within rtol 1e-7 and atol 1e-9.</p>
<p><a href="summary.csv">Summary CSV</a> · <a href="samples.csv">Every sample</a> ·
<a href="environment.json">Environment and settings</a></p>
<pre>python benchmarks/benchmark_scattering_cudss.py --sizes-um 0.12 0.06 0.03 --repeats 3</pre>
"""
    (output/'report.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes-um', type=float, nargs='+', default=[.12, .06, .03])
    parser.add_argument('--cases', choices=['pml', 'matched'], nargs='+', default=['pml', 'matched'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmarks/scattering_cudss'))
    args = parser.parse_args()
    if args.repeats < 1 or any(not np.isfinite(size) or size <= 0 for size in args.sizes_um):
        parser.error('positive finite sizes and repeats are required')
    args.output.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from fem.common.gpu import _runtime, is_available
    import fem.scattering.sources as sources
    _runtime()
    if not is_available():
        raise RuntimeError('A working cuDSS GPU is required')
    environment = dict(description=__doc__, python=platform.python_version(),
        platform=platform.platform(), cupy=cp.__version__,
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        threads={key:os.environ.get(key) for key in
                 ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        settings=vars(args) | {'output':str(args.output)})
    (args.output/'environment.json').write_text(json.dumps(environment, indent=2))
    samples, summary = [], []
    original = sources.solve_prescribed_pec
    for case in args.cases:
        for size in sorted(set(args.sizes_um), reverse=True):
            solver = make_solver(case, size)
            captured = []

            def direct(system, load, _captured=captured, **options):
                start = perf_counter()
                solution = original(system, load, **options)
                _captured.append((perf_counter()-start, system, load, options['boundary_values'], solution))
                return solution

            def run(backend, solver=solver, captured=captured, direct=direct):
                captured.clear()
                cp.cuda.get_current_stream().synchronize()
                start = perf_counter()
                with patch.object(sources, 'solve_prescribed_pec', direct):
                    result = solver.solve(backend=backend, max_refinements=0)
                cp.cuda.get_current_stream().synchronize()
                elapsed = perf_counter()-start
                assert len(captured) == 1
                return result, elapsed, captured[0]

            reference, _, baseline = run('cpu')
            warm_gpu, _, warm_capture = run('auto')
            assert warm_gpu.solve_info['method'] == 'cudss-direct'
            assert baseline[1].matrix.shape == warm_capture[1].matrix.shape
            assert (baseline[1].matrix != warm_capture[1].matrix).nnz == 0
            np.testing.assert_array_equal(baseline[2], warm_capture[2])
            np.testing.assert_array_equal(baseline[3], warm_capture[3])
            dofs = baseline[1].ndofs
            free = dofs-len(baseline[1].pec_dofs)
            print(f'{case}, {size} um: {dofs} DOF ({free} free)', flush=True)
            per_case = []
            for repeat in range(args.repeats):
                for backend in (('cpu','gpu') if repeat % 2 == 0 else ('gpu','cpu')):
                    result, public_seconds, capture = run(backend)
                    direct_seconds, _, _, _, field = capture
                    error = np.linalg.norm(field.coefficients-baseline[4].coefficients)/np.linalg.norm(baseline[4].coefficients)
                    np.testing.assert_allclose([result.S11,result.S21], [reference.S11,reference.S21],
                                               rtol=1e-7, atol=1e-9)
                    if error > 1e-7 or result.solve_info['relative_residual'] > 1e-10:
                        raise RuntimeError(('Direct backend disagreement', error, result.solve_info))
                    row = dict(case=case, size_um=size, dofs=dofs, free_dofs=free,
                        backend=backend, repeat=repeat, public_seconds=public_seconds,
                        direct_seconds=direct_seconds, relative_coefficient_error=float(error),
                        linear_residual=result.solve_info['relative_residual'], method=result.solve_info['method'])
                    samples.append(row)
                    per_case.append(row)
                    print(f'  {backend}: direct {direct_seconds:.4f}s, public {public_seconds:.4f}s',flush=True)
            timings = {backend:{stage:median(row[stage] for row in per_case if row['backend']==backend)
                       for stage in ('public_seconds','direct_seconds')} for backend in ('cpu','gpu')}
            summary.append(dict(case=case, size_um=size, dofs=dofs, free_dofs=free,
                cpu_direct_seconds=timings['cpu']['direct_seconds'], gpu_direct_seconds=timings['gpu']['direct_seconds'],
                direct_speedup=timings['cpu']['direct_seconds']/timings['gpu']['direct_seconds'],
                cpu_public_seconds=timings['cpu']['public_seconds'],gpu_public_seconds=timings['gpu']['public_seconds'],
                public_speedup=timings['cpu']['public_seconds']/timings['gpu']['public_seconds'],
                max_coefficient_error=max(row['relative_coefficient_error'] for row in per_case),
                max_linear_residual=max(row['linear_residual'] for row in per_case)))
            write_csv(args.output/'samples.csv',samples)
            write_csv(args.output/'summary.csv',summary)
            write_report(args.output, summary, environment)
            print(summary[-1],flush=True)
            del solver, reference, warm_gpu, baseline, warm_capture, result, capture, field, captured, direct, run
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
