"""Combine completed, disjoint mesh sweeps while retaining their provenance."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    metadata = []
    progress = []
    rows = []
    for source in args.inputs:
        metadata.append(json.loads((source/'metadata.json').read_text()))
        progress.append(json.loads((source/'progress.json').read_text()))
        rows.extend(json.loads((source/'summary.json').read_text()))
    first = metadata[0]
    for item, state in zip(metadata, progress):
        assert state['status'] == 'complete'
        assert all(c['exit_code'] == 0 for c in state['completed'])
        for key in ('commit', 'repeats', 'gpu_selection'):
            assert item[key] == first[key], key
        assert {k:v for k,v in item['source_sha256'].items() if k.startswith('src/')} == {
            k:v for k,v in first['source_sha256'].items() if k.startswith('src/')}
    cases = [case for item in metadata for case in item['cases']]
    assert len({case['name'] for case in cases}) == len(cases)
    args.output.mkdir(parents=True, exist_ok=False)
    for i, (source, item) in enumerate(zip(args.inputs, metadata)):
        shutil.copy2(source/'metadata.json', args.output/f'source_metadata_{i}.json')
        for case in item['cases']:
            destination = args.output/case['name']
            destination.mkdir()
            shutil.copy2(source/case['name']/'results.json', destination/'results.json')
    merged = dict(first, cases=cases, combined_from=[str(p.resolve()) for p in args.inputs],
                  provenance_note='Separate sequential sweeps; per-sweep metadata retained in source_metadata_*.json.')
    state = dict(status='complete', total=len(cases), active=None,
                 completed=[c for item in progress for c in item['completed']])
    for name, value in [('metadata', merged), ('progress', state), ('summary', rows)]:
        (args.output/f'{name}.json').write_text(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
