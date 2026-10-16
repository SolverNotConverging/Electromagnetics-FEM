"""Audit results and generate CPU/hybrid/resident comparison plots and report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();output=args.output
    metadata=json.loads((output/'metadata.json').read_text())
    progress=json.loads((output/'progress.json').read_text())
    rows=json.loads((output/'summary.json').read_text())
    errors=[];samples=[];records=[]
    if progress['status']!='complete' or any(c['exit_code']!=0 for c in progress['completed']):
        errors.append('Incomplete/failed processes')
    if len(rows)!=3*len(metadata['cases']) or not all(r['valid'] for r in rows):
        errors.append('Missing or invalid summary rows')
    for case in metadata['cases']:
        data=json.loads((output/case['name']/'results.json').read_text());records.append(data)
        samples.extend(data['samples'])
        expected={(b,i) for b in ('cpu_eigs','gpu_hybrid','gpu_resident') for i in range(metadata['repeats'])}
        if len(data['samples'])!=len(expected) or {(s['backend'],s['repeat']) for s in data['samples']}!=expected:
            errors.append(case['name']+': missing/duplicate samples')
        reference=next(s for s in data['warmups'] if s['backend']=='cpu_eigs')
        ref=np.array([complex(*v) for v in reference['eigenvalues']])
        for sample in data['samples']+data['warmups']:
            values=np.array([complex(*v) for v in sample['eigenvalues']])
            valid=len(values)==4 and np.isfinite(values).all() and sample['valid']
            if valid:
                distance=abs(abs(ref)[:,None]-abs(values)[None,:])
                left,right=linear_sum_assignment(distance)
                valid=max(distance[left,right]/np.maximum(1.,abs(ref[left])))<1e-5
            valid=valid and len(sample['residuals'])==4 and all(0<=r<=1e-8 for r in sample['residuals'])
            if sample['backend']=='gpu_resident':
                valid=valid and bool(sample['execution']) and all(e['ritz_backend']=='cusolver-geev-svd'
                    and e['application_control_download_bytes']==2*e['cycles'] for e in sample['execution'])
            if not valid:
                errors.append(f"{case['name']}: invalid {sample['backend']} {sample['phase']}")
        if any(p['num_threads']!=1 for p in data['threadpools'] if p['user_api']=='blas'):
            errors.append(case['name']+': unexpected BLAS thread count')
        if case['name'].endswith('-large') and data['field_dofs']<=10000:
            errors.append(case['name']+': large mesh below 10000 DOFs')
    root=Path(__file__).resolve().parents[2]
    for name,digest in metadata['source_sha256'].items():
        if name.startswith('src/') and hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            errors.append('Production source changed: '+name)
    audit=dict(passed=not errors,errors=errors,configurations=len(records),timed_samples=len(samples),
        warmups=sum(len(r['warmups']) for r in records),max_residual=max(s['max_residual'] for s in samples),
        max_magnitude_error=max(s['magnitude_error'] for s in samples))
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    if errors:
        raise SystemExit(str(errors))
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    grouped=[]
    for case in metadata['cases']:
        item={r['backend']:r for r in rows if r['case']==case['name']}
        cpu,hybrid,resident=(item[k] for k in ('cpu_eigs','gpu_hybrid','gpu_resident'))
        grouped.append(dict(case=case['name'],problem=case['problem'],mesh_mm=case['size'],
            field_dofs=cpu['field_dofs'],pencil_dofs=cpu['pencil_dofs'],
            cpu_seconds=cpu['median_seconds'],hybrid_seconds=hybrid['median_seconds'],resident_seconds=resident['median_seconds'],
            resident_speedup_vs_cpu=cpu['median_seconds']/resident['median_seconds'],
            resident_speedup_vs_hybrid=hybrid['median_seconds']/resident['median_seconds']))
    largest={problem:max((r for r in grouped if r['problem']==problem),key=lambda r:r['field_dofs'])['case']
             for problem in {r['problem'] for r in grouped}}
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(grouped[0]),lineterminator='\n');writer.writeheader();writer.writerows(grouped)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'iris':'Iris filter','siw':'SIW filter','perforated_waveguide':'Perforated waveguide','dielectric_rod':'Patched dielectric rod'}
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(problem,label) in zip(axes.flat,names.items()):
        for backend,legend in (('cpu_eigs','CPU ARPACK'),('gpu_hybrid','Hybrid CUDA'),('gpu_resident','Resident CUDA')):
            selected=sorted((r for r in rows if r['problem']==problem and r['backend']==backend),key=lambda r:r['field_dofs'])
            ax.errorbar([r['field_dofs'] for r in selected],[r['median_seconds'] for r in selected],
                yerr=[[r['median_seconds']-r['min_seconds'] for r in selected],
                      [r['max_seconds']-r['median_seconds'] for r in selected]],marker='o',capsize=3,label=legend)
        ax.set(xscale='log',yscale='log',xlabel='Reduced FEM DOFs',ylabel='Eigenvalue stage (s)',title=label)
        ax.grid(alpha=.25,which='both');ax.legend(fontsize=8)
    fig.suptitle('Current reduced-QEP CPU, hybrid GPU and resident GPU\nMedian and observed range; four eigenpairs, ncv=80, one CPU thread')
    fig.savefig(output/'timings.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for problem,label in names.items():
        selected=sorted((r for r in grouped if r['problem']==problem),key=lambda r:r['field_dofs'])
        for ax,key in zip(axes,('resident_speedup_vs_cpu','resident_speedup_vs_hybrid')):
            ax.plot([r['field_dofs'] for r in selected],[r[key] for r in selected],'o-',label=label)
    for ax,title in zip(axes,('CPU / resident CUDA','Hybrid / resident CUDA')):
        ax.axhline(1,color='black',ls='--');ax.set(xscale='log',xlabel='Reduced FEM DOFs',ylabel='Speedup (>1 favors resident)',title=title)
        ticks=[v for v in [2000,5000,10000,20000,50000,100000,200000] if v<=max(r['field_dofs'] for r in grouped)*1.1]
        ax.set_xticks(ticks,labels=[f'{v//1000}k' for v in ticks]);ax.tick_params(axis='x',which='minor',labelbottom=False)
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(output/'speedups.png',dpi=170);plt.close(fig)
    runtime=records[0]
    execution=[e for s in samples if s['backend']=='gpu_resident' for e in s['execution']]
    lines=['Resident GPU comparison — 2026-10-16','===================================','',
        f"Source main commit: {metadata['commit']}. {len(records)} configurations, {len(samples)} timed solves; all validated.",
        f"Maximum QEP residual: {audit['max_residual']:.3e}; maximum magnitude mismatch: {audit['max_magnitude_error']:.3e}.",'',
        f"Resident iteration diagnostics: steps {sorted({e['steps'] for e in execution})}, cycles {sorted({e['cycles'] for e in execution})}. This sweep measures matrix-size scaling, not difficult multi-restart convergence.",'',
        'Hardware and software','---------------------','',
        'Two Intel Xeon Silver 4210 CPUs at 2.20 GHz, 20 physical cores / 40 logical threads; 109.28 GiB OS-visible RAM.',
        f"One {runtime['gpu']['name']} (physical GPU {metadata['gpu_selection']}, 24 GiB VRAM), NVIDIA driver 595.45.04, CUDA runtime 13.2.",
        f"NumPy {runtime['numpy']}, SciPy {runtime['scipy']}, CuPy {runtime['cupy']}, cuDSS 0.8.0.10. BLAS thread limit one, verified at runtime.",
        'Full topology, installed package versions, GPU/load snapshots, commands and source hashes are in metadata.json. Shared host; timings include background variability.','',
        'When sweeps are combined, source_metadata_*.json retains each original run. Mesh densities were measured sequentially; small differences between GPU medians should not be treated as statistically established improvements.','',
        'Method','------','',
        'Same physical unit cells as the October 13 benchmark; mesh targets and actual DOFs are recorded for each case. Four eigenpairs, ncv=80, tolerance 1e-8, seed 0; fresh reduced Q(shift) factorization each solve.',
        'One excluded warmup/backend/case; three timed repeats with rotated backend order. Assembly is excluded. Transfers, readiness checks, factorization, iteration and eigenpair export are included.',
        'Both GPU paths perform the same readiness checks. Hybrid uses the retained iterate_hybrid control and exports full companion eigenvectors; resident exports the physical block. The production numerical implementations are unchanged.',
        'Host QEP validation runs after timing for every backend. GPU paths additionally perform their production QEP residual checks inside iteration; CPU ARPACK does not include that extra physical-residual pass in its stage timing.',
        'One-to-one eigenvalue-magnitude agreement accepts symmetry partners. These timings precede physical mode filtering, reconstruction and adaptive estimation; they are not complete public-solve timings.',
        'The old benchmark factored a 2n companion pencil. Current CPU and GPU paths factor n-dimensional Q(shift), so ratios from the earlier report cannot isolate the kernel change. This paired hybrid/resident comparison uses the same reduced operator.','',
        'Historical CPU timing also used ARPACK tolerance 1e-10 and included external residual evaluation; current CPU uses 1e-8 with independent host validation outside timing. Therefore old/new CPU speedups are not a controlled factorization-only experiment. Saved CPU root magnitudes across the 16 original meshes agree within 4.3e-14 (historical_root_comparison.json).','',
        'Results','-------','', '.. csv-table::',
        '   :header: "Case", "FEM DOFs", "CPU s", "Hybrid GPU s", "Resident GPU s", "CPU/resident", "Hybrid/resident"','']
    for r in grouped:
        lines.append(f"   \"{r['case']}\", {r['field_dofs']}, {r['cpu_seconds']:.4f}, {r['hybrid_seconds']:.4f}, {r['resident_seconds']:.4f}, {r['resident_speedup_vs_cpu']:.2f}, {r['resident_speedup_vs_hybrid']:.2f}")
    lines.extend(['','.. image:: timings.png','','.. image:: speedups.png','','Interpretation','--------------',''])
    for r in grouped:
        if r['case']==largest[r['problem']]:
            lines.append(f"* {names[r['problem']]} at {r['field_dofs']:,} FEM DOFs: resident speedup {r['resident_speedup_vs_cpu']:.2f}x versus CPU and {r['resident_speedup_vs_hybrid']:.2f}x versus hybrid GPU.")
    lines.extend(['','A ratio below one means the resident path is slower. Device residency by itself does not guarantee a speedup, particularly for small projected dense problems.',
        'With ncv=80, the projected dense eigensystem stays about 80-by-80 even as FEM DOFs grow. Hybrid already performs sparse factorization/solves on GPU and handles this small Ritz problem on CPU. Resident moves Ritz eigensolving/SVD to CUDA and launches conditional breakdown-handling kernels plus a second norm each Arnoldi step. These implementation differences are plausible overhead sources; this benchmark does not separately profile their costs.',
        'Every resident sample records CUDA geev/SVD execution, operator steps/cycles and two explicit application control bytes per cycle. Vendor-library internal workspace/control transfers are outside that count.','',
        'Files: comparison.csv, summary.csv/json, audit.json, metadata.json, and per-case results.json including first/warmup timings, all complex eigenvalues and residuals.',''])
    (output/'REPORT.rst').write_text('\n'.join(lines))
    print(json.dumps(audit,indent=2))
    for r in grouped:
        if r['case']==largest[r['problem']]: print(r)


if __name__=='__main__':
    main()
