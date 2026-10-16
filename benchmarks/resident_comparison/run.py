"""Current reduced-QEP CPU / hybrid CUDA / resident CUDA comparison.

Four historical 3D cells, four mesh densities, identical assembled operators.
Fresh factors per call; ncv=80, four roots, one CPU thread, three timed repeats.
Magnitude matching accepts the symmetry partners requested for this benchmark.
This measures the eigenvalue stage, before physical filtering/reconstruction.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BACKENDS = ('cpu_eigs', 'gpu_hybrid', 'gpu_resident')
MESHES = {'iris': (4., 3., 2., 1.4), 'siw': (2., 1.4, 1., .75),
          'perforated_waveguide': (5., 3.5, 2.5, 2.), 'dielectric_rod': (4., 3., 2., 1.7)}


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def capture(*args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def configurations():
    return [dict(name=f'{problem}-{level}', problem=problem, size=size)
            for problem, sizes in MESHES.items()
            for level, size in zip(('coarse', 'medium', 'fine', 'large'), sizes)]


def worker(case, output, repeats):
    import cupy as cp
    import numpy as np
    import scipy
    from scipy.optimize import linear_sum_assignment
    from threadpoolctl import threadpool_info
    from benchmarks.resident_comparison.geometries import assemble
    from fem.common.linalg import quadratic_eigenpairs, resolve_eigen_backend
    from fem.common.gpu import quadratic_operator
    from fem.common._gpu_arnoldi import iterate_hybrid
    from fem.common import _gpu_resident_arnoldi as resident

    output.mkdir(parents=True, exist_ok=True)
    record = dict(case=case, status='assembling', samples=[], warmups=[], errors=[])
    path = output/'results.json'
    save(path, record)
    started = time.perf_counter()
    solver, target, info = assemble(case['problem'], case['size'])
    system = solver.system
    a0, a1, a2 = system.A0, system.A1, system.A2
    digest = hashlib.sha256()
    for matrix in (a0,a1,a2):
        for values in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(values.tobytes())
    record.update(assembly_seconds=time.perf_counter()-started, geometry=info,
        field_dofs=system.ndofs, pencil_dofs=2*system.ndofs,
        matrix_sha256=digest.hexdigest(), nnz=[a.nnz for a in (a0,a1,a2)],
        target=[target.real,target.imag], numpy=np.__version__, scipy=scipy.__version__, cupy=cp.__version__,
        threadpools=threadpool_info(), status='running', ncv=80, modes=4,
        eigen_tolerance=1e-8, residual_tolerance=1e-8, max_operator_applications=2000)
    props=cp.cuda.runtime.getDeviceProperties(0)
    record['gpu']=dict(name=props['name'].decode(), vram=props['totalGlobalMem'],
        runtime=cp.cuda.runtime.runtimeGetVersion(), driver=cp.cuda.runtime.driverGetVersion())
    assert resolve_eigen_backend('gpu') == 'gpu'
    original = resident.iterate_device
    reference = None

    def trial(backend, phase, repeat):
        nonlocal reference
        execution=[]
        def track(*args,**kwargs):
            answer=original(*args,**kwargs)
            execution.append(answer[3])
            return answer
        cp.cuda.get_current_stream().synchronize()
        started=time.perf_counter()
        with patch.object(resident,'iterate_device',track):
            if backend=='gpu_hybrid':
                # Match the readiness checks performed by the public GPU API.
                assert resolve_eigen_backend('gpu') == 'gpu'
                shift=target+1e-6j*max(1.,abs(target))
                with quadratic_operator(a0,a1,a2,shift) as operator:
                    values,vectors=iterate_hybrid(operator,target,shift,4,1e-8,80,2000,0)
                vectors=vectors[:system.ndofs]
            else:
                values,vectors,_=quadratic_eigenpairs(a0,a1,a2,target=target,count=4,
                    backend='gpu' if backend=='gpu_resident' else 'cpu',tolerance=1e-8,
                    ncv=80,max_iterations=2000,random_seed=0)
        cp.cuda.get_current_stream().synchronize()
        elapsed=time.perf_counter()-started
        terms=(a0@vectors,(a1@vectors)*values,(a2@vectors)*values**2)
        scale=sum(np.linalg.norm(t,axis=0) for t in terms)
        residuals=np.linalg.norm(sum(terms),axis=0)/np.where(scale==0,1,scale)
        good=bool(len(values)==4 and np.isfinite(values).all() and np.isfinite(residuals).all()
                  and max(residuals)<=1e-8)
        if backend=='cpu_eigs' and phase=='warmup' and good:
            reference=values.copy()
        error=None
        if reference is not None and len(values)==len(reference) and np.isfinite(values).all():
            distances=abs(abs(reference)[:,None]-abs(values)[None,:])
            left,right=linear_sum_assignment(distances)
            error=float(max(distances[left,right]/np.maximum(1.,abs(reference[left]))))
        if backend=='gpu_resident' and not execution:
            raise RuntimeError('Resident core was not invoked')
        row=dict(backend=backend,phase=phase,repeat=repeat,seconds=elapsed,
            eigenvalues=[[float(v.real),float(v.imag)] for v in values],
            residuals=residuals.tolist(),max_residual=float(max(residuals)),magnitude_error=error,
            execution=execution,valid=bool(good and error is not None and error<1e-5))
        record['warmups' if phase=='warmup' else 'samples'].append(row)
        print(f"  {phase} {repeat} {backend}: {elapsed:.4f}s valid={row['valid']} residual={row['max_residual']:.2e}",flush=True)

    for phase,repeat in [('warmup',-1)]+[('timed',i) for i in range(repeats)]:
        order=BACKENDS if phase=='warmup' else BACKENDS[repeat%3:]+BACKENDS[:repeat%3]
        for backend in order:
            record['active']=dict(backend=backend,phase=phase,repeat=repeat)
            save(path,record)
            try:
                trial(backend,phase,repeat)
            except Exception:
                error=traceback.format_exc()
                record['errors'].append(dict(backend=backend,phase=phase,repeat=repeat,error=error))
                print(error,flush=True)
            save(path,record)
    record['summaries']=[]
    for backend in BACKENDS:
        samples=[s for s in record['samples'] if s['backend']==backend]
        times=[s['seconds'] for s in samples]
        warm=[s for s in record['warmups'] if s['backend']==backend]
        valid=(len(samples)==repeats and len(warm)==1 and all(s['valid'] for s in samples+warm)
               and not any(e['backend']==backend for e in record['errors']))
        record['summaries'].append(dict(backend=backend,valid=valid,
            median_seconds=statistics.median(times) if times else None,
            min_seconds=min(times) if times else None,max_seconds=max(times) if times else None,
            max_residual=max((s['max_residual'] for s in samples),default=None),
            max_magnitude_error=max((s['magnitude_error'] for s in samples if s['magnitude_error'] is not None),default=None)))
    record.update(status='complete',active=None)
    save(path,record)
    return all(r['valid'] for r in record['summaries'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',nargs='+')
    parser.add_argument('--mesh-case',action='append',default=[],metavar='NAME:PROBLEM:MM',
                        help='Run custom mesh targets instead of the standard configurations')
    parser.add_argument('--gpu',default='3')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--worker')
    args=parser.parse_args()
    args.output=args.output.resolve()
    if args.worker:
        raise SystemExit(0 if worker(json.loads(args.worker),args.output,args.repeats) else 1)
    available=configurations()
    if args.mesh_case:
        available=[]
        for spec in args.mesh_case:
            try:
                name,problem,size=spec.split(':')
                size=float(size)
                if problem not in MESHES or size<=0 or not name or Path(name).name!=name:
                    raise ValueError(spec)
                available.append(dict(name=name,problem=problem,size=size))
            except ValueError:
                parser.error('Expected NAME:PROBLEM:positive-mesh-mm')
        if len({c['name'] for c in available})!=len(available):
            parser.error('Mesh case names must be unique')
    selected=[c for c in available if not args.case or c['name'] in args.case]
    if not selected or (args.case and set(args.case)-{c['name'] for c in selected}):
        parser.error('Unknown case')
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    metadata=dict(commit=capture('git','rev-parse','HEAD').strip(), command=sys.argv,
        python=sys.version,platform=platform.platform(),cpu=capture('lscpu'),
        memory=Path('/proc/meminfo').read_text(),gpu_before=capture('nvidia-smi'),gpu_selection=args.gpu,
        packages=capture(sys.executable,'-m','pip','freeze'),cases=selected,repeats=args.repeats,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((ROOT/'src/fem/common').glob('*'))+list(Path(__file__).parent.glob('*.py')) if p.is_file()},
        methodology=__doc__,load_before=os.getloadavg())
    save(args.output/'metadata.json',metadata)
    progress=dict(status='running',total=len(selected),completed=[])
    combined=[]
    for case in selected:
        print('START',case['name'],flush=True)
        progress['active']=case['name'];save(args.output/'progress.json',progress)
        env=os.environ.copy()
        env.update({k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')})
        env.update(CUDA_VISIBLE_DEVICES=args.gpu,MPLBACKEND='Agg',CUPY_CACHE_DIR=str(args.output/'cupy-cache'))
        started=time.perf_counter()
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(case),
                '--output',str(args.output/case['name']),'--repeats',str(args.repeats)],env=env,timeout=1800)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code='timeout'
        progress['completed'].append(dict(case=case['name'],exit_code=code,seconds=time.perf_counter()-started))
        path=args.output/case['name']/'results.json'
        if path.exists():
            data=json.loads(path.read_text())
            for row in data.get('summaries',[]):
                combined.append(dict(case=case['name'],problem=case['problem'],size=case['size'],
                    field_dofs=data['field_dofs'],pencil_dofs=data['pencil_dofs'],**row))
        save(args.output/'summary.json',combined)
        save(args.output/'progress.json',progress)
    progress.update(status='complete',active=None);save(args.output/'progress.json',progress)
    metadata.update(gpu_after=capture('nvidia-smi'),load_after=os.getloadavg());save(args.output/'metadata.json',metadata)
    if combined:
        with (args.output/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(combined[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(combined)
    if len(combined)!=3*len(selected) or not all(r['valid'] for r in combined) or any(c['exit_code']!=0 for c in progress['completed']):
        raise SystemExit('Incomplete or invalid cases; inspect raw records')


if __name__=='__main__':
    main()
