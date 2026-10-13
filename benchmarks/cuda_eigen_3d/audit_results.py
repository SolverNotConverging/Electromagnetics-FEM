"""Audit completed raw samples, thread-sweep comparability, and solver hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.cuda_eigen_3d.validate_magnitudes import eigenvalues, magnitude_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.output / 'metadata.json').read_text())
    rows = json.loads((args.output / 'summary.json').read_text())
    progress = json.loads((args.output / 'progress.json').read_text())
    errors, records, thread_checks = [], {}, []
    backends = {'cpu_eigs', 'cpu_refined', 'cuda_cudss'}
    if len(rows) != 3 * len(metadata['cases']) or not all(r['valid'] for r in rows):
        errors.append('Missing or invalid summary rows')
    if progress['status'] != 'complete' or any(c['exit_code'] != 0 for c in progress['completed']):
        errors.append('Incomplete or failed case process')
    for case in metadata['cases']:
        record = json.loads((args.output / case['name'] / 'results.json').read_text())
        records[case['name']] = record
        libraries = record['runtime'].get('blas_libraries')
        if libraries is not None and (not libraries or any(x['configured_threads'] != case['threads'] for x in libraries)):
            errors.append(f"{case['name']}: OpenBLAS runtime thread limit mismatch")
        reference = next(s for s in record['warmups'] if s['backend'] == 'cpu_eigs')
        expected = {(backend, i) for backend in backends for i in range(metadata['repeats'])}
        observed = {(s['backend'], s['repeat']) for s in record['samples']}
        if observed != expected or len(record['samples']) != len(expected):
            errors.append(f"{case['name']}: missing/duplicate timed samples")
        if {s['backend'] for s in record['warmups']} != backends or len(record['warmups']) != 3:
            errors.append(f"{case['name']}: missing/duplicate warmups")
        for sample in record['warmups'] + record['samples']:
            error = magnitude_error(eigenvalues(reference), eigenvalues(sample))
            if (not sample['valid'] or not sample['converged'] or error is None or error >= 1e-5
                or len(sample['residuals']) != case['modes']
                or not all(0 <= r <= case['tol'] for r in sample['residuals'])):
                errors.append(f"{case['name']}: invalid {sample['backend']} {sample['phase']} {sample['repeat']}")
        if case['name'].endswith('-large') and record['field_dofs'] <= 10000:
            errors.append(f"{case['name']}: below requested DOF threshold")
    for name, record in records.items():
        if '-medium-threads' not in name:
            continue
        baseline = records[name.split('-threads')[0]]
        current = next(s for s in record['warmups'] if s['backend'] == 'cpu_eigs')
        reference = next(s for s in baseline['warmups'] if s['backend'] == 'cpu_eigs')
        error = magnitude_error(eigenvalues(reference), eigenvalues(current))
        same = all(record[key] == baseline[key] for key in ('field_dofs', 'pencil_dofs', 'nnz_a', 'nnz_b', 'sigma', 'geometry'))
        thread_checks.append(dict(case=name, same_mesh_statistics=same, magnitude_error=error))
        if not same or error is None or error >= 1e-8:
            errors.append(f'{name}: differs from baseline thread problem')
    root = Path(__file__).resolve().parents[2]
    for batch in metadata.get('measurement_runs', [metadata]):
        for path, digest in batch['source_sha256'].items():
            if path.startswith('periodic_eigensolver/') and hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
                errors.append(f'Solver source changed: {path}')
    audit = dict(configurations=len(records), backend_rows=len(rows),
        timed_samples=sum(len(r['samples']) for r in records.values()),
        warmups=sum(len(r['warmups']) for r in records.values()),
        thread_comparability=thread_checks, errors=errors, passed=not errors)
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
