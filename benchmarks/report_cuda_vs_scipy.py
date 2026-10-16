"""Render measured CUDA versus SciPy solver results as CSV, PNG, and HTML."""
import argparse
import csv
import html
import json
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    state = json.loads((args.directory/'results.json').read_text())
    env = json.loads((args.directory/'environment.json').read_text())
    for r in state['results']:
        # Include resident geometry preparation in a fresh-mesh total. Retain
        # the separate mesh-plus-warm arithmetic for historical comparisons.
        assembly = r['assembly_seconds']
        r['mesh_plus_warm_solve_speedup'] = (assembly+r['cpu_seconds'])/(assembly+r['gpu_seconds'])
        r['total_speedup'] = (assembly+r['first'][0]['public_seconds'])/(assembly+r['first'][1]['public_seconds'])
    completed = {r['id'] for r in state['results']}
    for failure in state['failures']:
        failure['resolved'] = failure['config']['id'] in completed
    (args.directory/'results.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    results = sorted(state['results'], key=lambda r: (
        r['package'] != 'FEM', ('waveguide_2d','periodic_2d','periodic_3d').index(r['family']),
        r['case'] not in ('pec','iris_pec'), r['case'], r['dofs']))
    if not results:
        raise RuntimeError('No completed benchmark cases')
    fields = [k for k in results[0] if k not in ('first','samples')]
    with (args.directory/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)
    figure = Figure(figsize=(13, 4.3))
    axes = figure.subplots(1, 3)
    for ax, family in zip(axes, ('waveguide_2d', 'periodic_2d', 'periodic_3d')):
        for package, color in (('FEM','#2367ae'), ('FDFD','#e07c22')):
            selected = sorted((r for r in results if r['family'] == family and r['package'] == package
                               and r['case'] in ('pec','iris_pec')), key=lambda r:r['dofs'])
            if selected:
                ax.plot([r['dofs'] for r in selected], [r['speedup'] for r in selected], 'o-', color=color,
                        label=f'{package}: public solve')
                ax.plot([r['dofs'] for r in selected], [r['eigen_speedup'] for r in selected], 's--', color=color,
                        label=f'{package}: eigenvalue stage')
        ax.axhline(1, color='#666', linestyle=':', label='CPU / GPU = 1')
        ax.set(xscale='log', yscale='log', xlabel='Physical / reduced DOF', ylabel='CPU time / GPU time',
               title=family.replace('_',' ').title())
        ax.grid(alpha=.2)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.directory/'speedup.png', dpi=160)
    body = []
    for r in results:
        cells = [r['package'], r['family'].replace('_',' '), r['case'], f"{r['dofs']:,}",
            f"{r['eigen_dimension']:,}", f"{r['cpu_eigen_seconds']:.4f}", f"{r['gpu_eigen_seconds']:.4f}",
            f"{r['eigen_speedup']:.2f}×", f"{r['cpu_seconds']:.4f}", f"{r['gpu_seconds']:.4f}",
            f"{r['speedup']:.2f}×", f"{r['total_speedup']:.2f}×", f"{r['max_relative_neff_difference']:.1e}"]
        body.append('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>')
    max_error = max(r['max_relative_neff_difference'] for r in results)
    max_residual = max(r['max_residual'] for r in results)
    first_3d = ''.join('<tr>'+''.join('<td>'+c+'</td>' for c in (
        r['case'], f"{r['dofs']:,}", f"{r['first'][0]['public_seconds']:.3f}",
        f"{r['first'][1]['public_seconds']:.3f}", f"{r['first_solve_speedup']:.2f}×"))+'</tr>'
        for r in results if r['family']=='periodic_3d')
    failures = ''.join('<li>'+html.escape(f['config']['id'])+': '+html.escape(f['traceback'].splitlines()[-1])+'</li>'
                       for f in state['failures'] if not any(r['id']==f['config']['id'] for r in results))
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>FEM / FDFD CUDA vs SciPy benchmark</title>
<style>body{{font:15px system-ui;margin:35px auto;max-width:1550px;padding:0 22px;color:#203040}}
h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:9px 7px;border-bottom:1px solid #d6dee5;text-align:right}}th{{background:#edf3f8}}
td:nth-child(-n+3){{text-align:left}}img{{width:100%;max-width:1400px}}.note{{background:#edf3f8;padding:16px}}
a{{color:#2367ae}}</style><h1>CUDA resident solvers versus SciPy eigs</h1>
<p>Measured on 16 October 2026 · Intel Core i9-13900H · {html.escape(env['gpu'])} (8 GB)
· single-thread CPU BLAS/OpenMP · Python {env['python']} · SciPy {env['scipy']} · CuPy {env['cupy']}</p>
<p class="note">Speedup = CPU seconds / GPU seconds; above 1 means CUDA is faster.
These measurements compare the released CUDA paths with each project's SciPy ARPACK + SuperLU path,
using identical assembled problems and requesting {env['modes']} physical mode(s) at tolerance {env['tolerance']:g}.
Median of {env['repeats']} alternating paired repeats after one untimed solve per backend and mesh.</p>
<img src="speedup.png" alt="Measured GPU speedup versus degrees of freedom">
<p>{len(results)} validated configurations; maximum relative CPU/GPU effective-index difference
{max_error:.2e}; maximum returned-mode residual {max_residual:.2e}.</p>
<table><thead><tr>{''.join('<th>'+s+'</th>' for s in ('Project','Problem','Material case','DOF','Eigenproblem size',
'CPU eigen (s)','GPU eigen (s)','Eigen speedup','CPU public (s)','GPU public (s)','Public speedup',
'New mesh + first solve speedup','Max neff difference'))}</tr></thead><tbody>{''.join(body)}</tbody></table>
<h2>First solve on each FEM 3D mesh</h2><p>Includes initial resident geometry preparation.
CUDA runtime and compiler caches were already warmed; this is not an installation-to-first-result timing.</p>
<table><thead><tr><th>Case</th><th>DOF</th><th>CPU public (s)</th><th>GPU public (s)</th>
<th>First-mesh solve speedup</th></tr></thead><tbody>{first_3d}</tbody></table>
<h2>Timing boundaries</h2><p>The eigenvalue stage includes fresh sparse factorization, iteration, dense Ritz work,
and final eigenpair export. FDFD includes its original-pencil backward-error check in this timer;
the resident FEM 3D timer includes the GPU polynomial residual check. CPU FEM residual filters run after
the eigenvalue timer. Public solve timings also include filters, field reconstruction/normalization,
error estimation and final result export. Meshing is excluded from public timings and included once
in the new mesh + first solve comparison, which also includes initial GPU geometry preparation.
FDFD rebuilds sparse modal matrices inside each public solve.</p>
<p>CUDA context, capability probes and compiler caches are warmed globally. First GPU solves per mesh are
recorded separately in results.json and include initial resident geometry preparation; warm repeats
reuse this static geometry but never reuse LU factors. No adaptive refinement is performed.</p>
<p>Waveguide Krylov settings use the production defaults. FEM waveguides request extra eigenpairs to filter
physical modes and use a wider default CUDA subspace than ARPACK; this is a comparison of the current
public implementations, not identical Arnoldi algorithms. FDFD periodic 2D and FEM periodic 3D use
ncv=96 on both backends. FEM periodic 2D retains its public defaults. FEM's quadratic eigenproblem
has twice the physical DOF; FDFD reports the actual reduced pencil size.</p>
<p>FDFD periodic 3D remains excluded per the earlier request. Laptop results do not establish Titan RTX performance.</p>
{('<h2>Unresolved cases</h2><ul>'+failures+'</ul>') if failures else ''}
<p><a href="summary.csv">Summary CSV</a> · <a href="results.json">Raw paired timings and eigenvalues</a>
· <a href="environment.json">Environment and verified thread pools</a></p></html>'''
    (args.directory/'report.html').write_text(report, encoding='utf-8')
    print(args.directory/'report.html')


if __name__ == '__main__':
    main()
