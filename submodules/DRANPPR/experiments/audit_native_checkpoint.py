"""Two-start/two-objective frozen-data audit; truth is loaded only after all solves."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def typed_state(path):
    rows = []
    with Path(path).open() as stream:
        for original in csv.DictReader(stream):
            kind = original['kind']
            if kind not in ['pose', 'landmark']:
                raise ValueError('unsupported state kind')
            row = dict(robot=int(original['robot']), kind=kind, index=int(original['index']))
            for key in ['x', 'y'] + (['timestamp_s', 'yaw'] if kind == 'pose' else []):
                row[key] = float(original[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f'nonfinite {key}')
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--start', action='append', required=True, help='NAME=state.csv')
    parser.add_argument('--robot', type=int, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 30:
        parser.error('require duration > 30 for the predeclared evaluation window')
    starts = {}
    for item in args.start:
        name, path = item.split('=', 1)
        if not name or not name.replace('_', '').isalnum() or name in starts:
            parser.error('start names must be unique alphanumeric/underscore labels')
        starts[name] = Path(path)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.stream/'metadata.json').read_text())
    events = args.stream/'events.jsonl'
    if sha256(events) != metadata['event_sha256']:
        raise ValueError('stream hash mismatch')
    root = Path(__file__).resolve().parents[1]
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    for path in [Path(__file__), root/'experiments/native_risam/checkpoint.cpp',
                 root/'experiments/native_risam/CMakeLists.txt',
                 root/'experiments/evaluate_incremental_reference.py',
                 root/'tests/test_native_risam_checkpoint.py', root/'tests/test_audit_native_checkpoint.py']:
        shutil.copy2(path, snapshot/path.name)
    libraries = subprocess.check_output(['ldd', str(args.binary)], text=True)
    linked = [Path(line.split('=>', 1)[1].split()[0]) for line in libraries.splitlines() if '=>' in line]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    binary_sha256=sha256(args.binary), dynamic_libraries=libraries,
                    dynamic_library_hashes={str(p):sha256(p) for p in linked if p.is_file()},
                    stream_metadata=metadata, state_hashes={n:sha256(p) for n,p in starts.items()},
                    source_hashes={p.name:sha256(p) for p in snapshot.iterdir()}, runs=[],
                    scope='Local fixed-prefix diagnostic; not online or full distributed riMESA',
                    controls='Same factors, gauge, noise and 100-iteration LM budget; change only start and observation loss')
    completed = []
    for name, path in starts.items():
        state_json = args.output/f'{name}_initial.json'
        state_json.write_text(json.dumps(typed_state(path), allow_nan=False)+'\n')
        for mode in ['gaussian', 'sig_final']:
            out = args.output/f'{name}_{mode}'
            command = [str(args.binary), str(events), str(state_json), str(out),
                       str(args.robot), str(args.duration), mode]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (args.output/f'{name}_{mode}.log').write_text(result.stdout+result.stderr)
            manifest['runs'].append(dict(start=name, mode=mode, command=command, exit_code=result.returncode))
            (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f'{name}/{mode} failed: {result.stderr}')
            completed.append((name, mode, out))
            print(f'{name}/{mode}: completed', flush=True)
    # The estimator processes have all exited before any ground-truth file is opened.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'experiments'))
    from evaluate_incremental_reference import evaluate_rows, read_rows
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    dataset_manifest = load_manifest(root/f"benchmarks/mrclam/manifests/dataset{metadata['dataset']}.json")
    audit = audit_dataset(dataset_manifest, args.raw)
    truth = load_ground_truth(dataset_manifest, audit.dataset_directory)
    label_map = dict(dataset_manifest.evaluation.object_ground_truth_label_map)
    with events.open() as source:
        first = json.loads(next(source))['timestamp_s']
    summaries = []
    for name, mode, out in completed:
        stats = json.loads((out/'summary.json').read_text())
        for state in ['before', 'after']:
            rows = [r for r in read_rows(out/f'{state}_estimates.csv')
                    if r['kind']=='landmark' or float(r['timestamp_s'])-first>30.]
            aggregate, _ = evaluate_rows(rows, truth, label_map)
            weights = read_rows(out/f'{state}_weights.csv')
            summaries.append(dict(start=name, mode=mode, state=state, **aggregate, **stats[state],
                                  lm_iterations=stats['lm_iterations'], lm_seconds=stats['lm_seconds'],
                                  observations=stats['observations'],
                                  gm_weight_sum=sum(float(r['gm_weight']) for r in weights),
                                  gm_weight_below_point1=sum(float(r['gm_weight'])<.1 for r in weights)))
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
