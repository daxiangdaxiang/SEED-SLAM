"""Frozen local native/Schur/Jacobian audit; truth is opened after all solves."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from audit_native_checkpoint import sha256, typed_state


def state_difference(rows, reference):
    def keyed(items):
        result = {(r['robot'], r['kind'], r['index']): r for r in items}
        if len(result) != len(items):
            raise ValueError('duplicate state keys')
        return result
    actual, expected = keyed(rows), keyed(reference)
    if actual.keys() != expected.keys():
        raise ValueError('inconsistent state keys')
    distance, angle = 0., 0.
    for key, row in actual.items():
        other = expected[key]
        distance = max(distance, math.hypot(float(row['x'])-float(other['x']),
                                           float(row['y'])-float(other['y'])))
        if row['kind'] == 'pose':
            delta = float(row['yaw'])-float(other['yaw'])
            angle = max(angle, abs(math.atan2(math.sin(delta), math.cos(delta))))
    return dict(max_translation_to_exact_lm=distance, max_yaw_to_exact_lm=angle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--start', action='append', required=True, help='NAME=state.csv')
    parser.add_argument('--robot', type=int, default=5)
    parser.add_argument('--duration', type=float, default=300.)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 30 or not math.isfinite(args.duration):
        parser.error('require a finite duration > 30 for the fixed evaluation window')
    starts = {}
    for item in args.start:
        name, path = item.split('=', 1)
        if not name or not name.replace('_', '').isalnum() or name in starts:
            parser.error('start labels must be unique alphanumeric/underscore names')
        starts[name] = Path(path)
    args.output.mkdir(parents=True, exist_ok=False)
    events = args.stream/'events.jsonl'
    metadata = json.loads((args.stream/'metadata.json').read_text())
    if sha256(events) != metadata['event_sha256']:
        raise ValueError('stream hash mismatch')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'experiments/audit_native_checkpoint.py',
               root/'experiments/evaluate_incremental_reference.py',
               root/'experiments/native_local/audit.cpp', root/'cmake/native_local_model.cmake',
               root/'CMakeLists.txt', root/'tests/test_native_local_audit.py',
               root/'tests/test_audit_native_local.py']
    for name in ['IncrementalLocalModel', 'NativeSchurModel', 'NativeLocalOptimizer', 'NativeBetweenPose2Factor']:
        sources += [root/f'src/{name}.cpp', root/f'include/DPGO/{name}.h']
    sources += [root/f'tests/test{name}.cpp' for name in
                ['IncrementalLocalModel', 'NativeSchurModel', 'NativeLocalOptimizer']]
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot/source.name)
    libraries = subprocess.check_output(['ldd', str(args.binary)], text=True)
    linked = [Path(row.split('=>', 1)[1].split()[0]) for row in libraries.splitlines() if '=>' in row]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    binary_sha256=sha256(args.binary), dynamic_libraries=libraries,
                    library_hashes={str(p): sha256(p) for p in linked if p.is_file()},
                    stream_metadata=metadata, state_hashes={n: sha256(p) for n, p in starts.items()},
                    source_hashes={p.name: sha256(p) for p in snapshot.iterdir()}, runs=[],
                    scope='Fixed local prefix, same Gaussian likelihood, not online/distributed or matched wall time')
    def save_manifest():
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save_manifest()
    for name, path in starts.items():
        state = args.output/f'{name}_initial.json'
        state.write_text(json.dumps(typed_state(path), allow_nan=False)+'\n')
        command = [str(args.binary), str(events), str(state), str(args.output/name), str(args.robot), str(args.duration)]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as error:
            manifest['runs'].append(dict(start=name, command=command, exit_code=None, failure='timeout'))
            save_manifest()
            raise RuntimeError(f'{name} exceeded 300 seconds') from error
        (args.output/f'{name}.log').write_text(run.stdout+run.stderr)
        manifest['runs'].append(dict(start=name, command=command, exit_code=run.returncode))
        save_manifest()
        if run.returncode:
            raise RuntimeError(f'{name} failed: {run.stderr}')
        print(f'{name} completed', flush=True)
    # No estimator process can access truth through this evaluation stage.
    sys.path.insert(0, str(root))
    from evaluate_incremental_reference import read_rows, evaluate_rows
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import load_manifest, audit_dataset
    dataset = load_manifest(root/f"benchmarks/mrclam/manifests/dataset{metadata['dataset']}.json")
    source = audit_dataset(dataset, args.raw)
    truth = load_ground_truth(dataset, source.dataset_directory)
    labels = dict(dataset.evaluation.object_ground_truth_label_map)
    first = metadata['first_event_time_s']
    summaries = []
    for name in starts:
        folder = args.output/name
        stats = json.loads((folder/'summary.json').read_text())
        reference = read_rows(folder/'exact_lm_estimates.csv')
        for run in stats['runs']:
            rows = read_rows(folder/f"{run['method']}_estimates.csv")
            difference = state_difference(rows, reference)
            filtered = [r for r in rows if r['kind'] == 'landmark' or float(r['timestamp_s'])-first > 30.]
            metrics, _ = evaluate_rows(filtered, truth, labels)
            summaries.append(dict(dataset=metadata['dataset'], start=name,
                                  initial_cost=stats['initial_cost'], **run, **difference, **metrics))
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
