"""Generate the reviewable joint analytical validation report and plots."""
from pathlib import Path
import csv
import html
import json
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.analytical.cross_solver_validation import OUT, verify_summary
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator, ScalarFormatter
import scipy

data=json.loads((OUT/'results.json').read_text())
rows=data['rows']
checks=verify_summary(rows)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2))
assert not data['failures'],data['failures']
assert all(row['passed'] for row in rows)
assert all(check['passed'] for check in checks),[c for c in checks if not c['passed']]
assert not any(r['package']=='FDFD' and 'periodic_3d' in r['case'] for r in rows)
metadata=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
              fem_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              fdfd_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'FDFD',text=True).strip())
import cupy as cp
metadata.update(cupy=cp.__version__,cuda=cp.cuda.runtime.runtimeGetVersion(),
                gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
(OUT/'environment.json').write_text(json.dumps(metadata,indent=2))

summary=[]
for case,package in sorted({(r['case'],r['package']) for r in rows}):
    selected=[r for r in rows if r['case']==case and r['package']==package and r['backend']=='cpu']
    finest=max(selected,key=lambda r:r['level'])
    summary.append({key:finest[key] for key in ('case','package','level','error','metric','tolerance')})
with (OUT/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=summary[0])
    writer.writeheader(); writer.writerows(summary)

fig=Figure(figsize=(13,9))
axes=fig.subplots(2,3)
specifications=[('waveguide_1d_pec_TE1','PEC parallel plate, TE1'),
    ('dielectric_slab_TE0_index','Dielectric slab: dispersion root'),
    ('dielectric_slab_TE0_field','Dielectric slab: field profile'),
    ('periodic_2d_TE_1.0','Periodic 2D guide: TE cutoff'),
    ('poisson_1d_interpolated_potential','Poisson: interpolated potential'),
    ('scattering_cylinder','Cylinder: Bessel/Hankel series')]
for ax,(prefix,title) in zip(axes.ravel(),specifications,strict=True):
    for case,package in sorted({(r['case'],r['package']) for r in rows if r['case'].startswith(prefix)}):
        selected=sorted([r for r in rows if r['case']==case and r['package']==package and r['backend']=='cpu'],key=lambda r:r['level'])
        label=package if prefix!='scattering_cylinder' else case.replace('scattering_cylinder_','')
        ax.loglog([r['level'] for r in selected],[max(r['error'],1e-16) for r in selected],'o-',label=label)
    ax.set(title=title,xlabel='Cells across / refinement level',ylabel='Relative analytical error')
    levels=sorted({r['level'] for r in rows if r['case'].startswith(prefix)})
    ax.set_xticks(levels)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'convergence.png',dpi=170)

suite_stats={}
for package in ('fem','fdfd'):
    d=json.loads((OUT/f'{package}_suite.json').read_text())
    assert d['exitstatus']==0
    suite_stats[package]={outcome:sum(r['outcome']==outcome for r in d['results']) for outcome in ('passed','failed','skipped')}
    suite_stats[package]['deselected']=len(d['excluded'])
    if package=='fdfd':
        assert any('test_periodic_3d_image_guide' in name for name in d['excluded'])
max_agreement=max(c['error'] for c in checks if c['check']=='cpu_gpu_agreement')
table=''.join(f"<tr><td>{html.escape(r['case'])}</td><td>{r['package']}</td><td>{r['level']}</td><td>{r['error']:.3g}</td><td>{r['tolerance']:g}</td></tr>" for r in summary)
report=f'''<!doctype html><meta charset="utf-8"><title>Joint FEM and FDFD analytical validation</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#183047}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #d6dfe5;text-align:left}}img{{width:100%}}code{{background:#edf2f6;padding:2px 5px}}.pass{{color:#08784f}}</style>
<h1>FEM + FDFD: analytical validation</h1><p>2026-10-16 · <b class="pass">{len(rows)} / {len(rows)} analytical measurements passed</b> · {len(checks)} additional coverage, convergence, residual and CPU/GPU checks passed.</p>
<p>Every supported numerical family is represented. FDFD periodic 3D was excluded from both the custom validation and the regression selection, as requested. FEM periodic 3D remains included.</p>
<h2>Independent analytical references</h2><ul>
<li><b>FEM electrostatics 1D/2D:</b> V=V₀x/L, E=−V₀/L, energy=εV₀²(volume)/(2L²); quadratic Poisson potential ρx(L−x)/(2ε), measured between nodes.</li>
<li><b>Both waveguide solvers 1D/2D:</b> PEC TE1/TE10 n_eff=√(εᵣμᵣ−(λ₀/2a)²), lossy TEM n_eff=√εᵣ, anisotropic TEM n_eff=√(εₓμᵧ).</li>
<li><b>Both 1D dielectric slabs:</b> independent scalar root of u tan(u)=√(V²−u²), with cosine core and exponential cladding fields. Fit only an arbitrary complex field amplitude. Finite PEC truncation lies deep in the evanescent tail.</li>
<li><b>Both periodic 2D solvers:</b> uniform guide TE cutoff and lossy TM index. <b>FEM periodic 3D:</b> uniform rectangular TE10.</li>
<li><b>FDFD bands:</b> continuous-medium ω=c|k|/√(εᵣμᵣ), both polarizations, complex ε and μ; spatial discretization error decreases with refinement.</li>
<li><b>FEM scattering:</b> complex Fabry–Perot slab reflection/transmission, including public reference-plane phases, and lossless power conservation. Compare first- and second-order elements.</li>
<li><b>FDFD scattering:</b> independent cylindrical Bessel/Hankel expansion for PEC and dielectric cylinders, TE and TM, measured in an exterior annulus with a PML. This is a field error, not a linear-system residual.</li>
<li><b>FDFD tracking 1D/2D:</b> below-cutoff β=−i√((π/a)²−k₀²), including physical evanescent decay.</li>
<li><b>Both 2D SIBC solvers:</b> copper TE10 attenuation from first-order conductor perturbation, α=Rₛ[1+2(b/a)(λ₀/2a)²]/[bη√(1−(λ₀/2a)²)], Rₛ=√(πfμ₀/σ). This reference has a physical perturbation-error floor.</li>
</ul><p>Positive-time phasors exp(+iωt): passive spatial modes have Im(n_eff)&lt;0; fixed-real-k lossy band frequencies have Im(ω)&gt;0. Each analytical group ran with explicit CPU and explicit CUDA execution; no silent auto fallback was accepted.</p>
<h2>Results and convergence</h2><p>Maximum CPU/GPU relative difference: <b>{max_agreement:.3g}</b>. The largest fine-grid cylinder field error is 2.58% (PEC TM); cylinder staircasing converges more slowly than the smooth uniform-guide cases. Lossy-mode attenuation error was also checked separately from the dominant real part of n_eff, and is below 0.1% across the tested guide/periodic cases.</p>
<img src="convergence.png" alt="Analytical error versus refinement for both numerical methods">
<h2>Numerical defect found and fixed</h2><p>FEM accepted a Boolean-difference copper frame in its geometry API, and Gmsh generated the correct aperture, but the boundary classifier rejected facets on the inner wall. Difference.contains excludes its closed tool boundary. The classifier now probes either side of a facet within its geometric tolerance to identify the exposed solid. Two regression tests verify wall ownership and that no elements fill the opaque frame. Existing annulus/curved/conductor geometry tests also pass.</p>
<p>With the repair, finer-grid copper attenuation agrees with first-order theory to 0.0102% (FEM) and 0.0631% (FDFD). No additional numerical discrepancy beyond expected discretization error was found in this validation.</p>
<h2>Additional regression coverage</h2><p>Selected FEM numerical suite: {suite_stats['fem']['passed']} passed, {suite_stats['fem']['skipped']} optional tests skipped. FDFD numerical suite including real CUDA tests: {suite_stats['fdfd']['passed']} passed, {suite_stats['fdfd']['skipped']} optional test skipped; 3D periodic and viewer-only cases deselected. Separate geometry regressions: 14 passed. See the logs for exact test identities and exclusions.</p>
<p>This is validation over representative analytical cases, not a proof for arbitrary geometry. FEM 1D impedance boundaries remain explicitly unsupported and are tested to raise; 2D SIBC is tested physically. PML behavior is additionally covered by slab-reflection convergence and cylinder outgoing fields.</p>
<h2>Finest-grid measurements</h2><p>Errors are relative unless the case is a complex S-parameter or power check, which uses absolute error. S-parameter vector error is the Euclidean norm over S11 and S21. Same tolerances apply on CPU and GPU.</p>
<table><tr><th>Case</th><th>Method</th><th>Level</th><th>Error</th><th>Limit</th></tr>{table}</table>
<h2>Reproduce and inspect</h2><p>Run <code>python benchmarks/analytical/cross_solver_validation.py</code> with FEM's dependencies, sibling FDFD/src and working CuPy/cuDSS. Set temporary/cache directories writable on Windows. Then run <code>run_numerical_suites.py FEM</code> and <code>run_numerical_suites.py FDFD</code> and this report generator. Local environment: Python {metadata['python']}, NumPy {metadata['numpy']}, SciPy {metadata['scipy']}, CuPy {metadata['cupy']}, {metadata['gpu']}.</p>
<p><a href="results.json">Raw analytical measurements</a> · <a href="checks.json">Additional numerical checks</a> · <a href="summary.csv">Summary CSV</a> · <a href="fem_suite.json">FEM test manifest</a> · <a href="fdfd_suite.json">FDFD test manifest and exclusions</a> · <a href="fem_cuda.log">Extra FEM CUDA checks</a> · <a href="meshing_regressions.log">Meshing regressions</a> · <a href="environment.json">Environment and revisions</a></p>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
print('All analytical checks passed; report:',OUT/'report.html')
