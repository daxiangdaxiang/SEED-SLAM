"""Chronological joint replay gate with retained failures and posthoc-only truth."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.source import audit_dataset, load_manifest
from experiments.audit_native_checkpoint import sha256, typed_state
from experiments.evaluate_incremental_reference import evaluate_rows, read_rows
from experiments.evaluate_native_joint import evaluate_joint_rows
from experiments.prepare_native_joint import prepare


def validate_ledger(output):
    summary = json.loads((output/'summary.json').read_text())
    updates = read_rows(output/'updates.csv')
    if len(updates) != summary['frames']:
        raise ValueError('frame count mismatch')
    if any(float(r['cost_after']) > float(r['cost_before'])+1e-8 for r in updates):
        raise ValueError('fixed-graph step increased physical cost')
    for field, key in [('control_bytes', 'control_application_bytes'),
                       ('optimizer_bytes', 'optimizer_application_bytes'), ('pcg_iterations', 'pcg_iterations')]:
        if sum(int(r[field]) for r in updates) != summary[key]:
            raise ValueError(f'{field} ledger mismatch')
    byte_count = 0
    with (output/'control_messages.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if abs(row['from']-row['to']) != 1 or not 1 <= min(row['from'], row['to']) <= 4:
                raise ValueError('control message outside declared chain')
            if row['application_bytes'] != 40+8*len(row['payload']):
                raise ValueError('packet byte mismatch')
            byte_count += row['application_bytes']
    if byte_count != summary['control_application_bytes']:
        raise ValueError('control packet total mismatch')
    summary['descent_violations'] = 0
    summary['ledger_verified'] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--stream-root', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True, help='dataset:duration')
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--mode', action='append', choices=[
        'joint', 'joint_cached', 'joint_one', 'joint_cached_one', 'independent'])
    args = parser.parse_args()
    modes = args.mode or ['joint', 'independent']
    if args.evaluate_only:
        manifest = json.loads((args.output/'manifest.json').read_text())
        completed = []
        for case in args.case:
            d, t = case.split(':')
            dataset, duration = int(d), float(t)
            for mode in modes:
                tag = f'd{dataset}_{int(duration)}_{mode}'
                run = next(r for r in manifest['runs'] if r['tag'] == tag)
                if run['exit_code'] != 0:
                    continue
                folder = args.output/tag
                completed.append((dataset, duration, mode, folder, validate_ledger(folder)))
        evaluate_results(completed, Path(__file__).resolve().parents[1], args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'experiments/native_local/joint_replay.cpp',
        root/'experiments/native_local/joint_step.h', root/'experiments/native_local/joint.cpp',
        root/'experiments/prepare_native_joint.py', root/'experiments/evaluate_native_joint.py',
        root/'experiments/evaluate_incremental_reference.py', root/'cmake/native_local_model.cmake',
        root/'tests/test_native_joint_replay.py', root/'tests/testNativeInterface.cpp']
    for name in ['IncrementalLocalModel', 'NativeSchurModel', 'NativeLocalOptimizer', 'NativeBetweenPose2Factor', 'NativeInterface']:
        sources += [root/f'include/DPGO/{name}.h', root/f'src/{name}.cpp']
    for source in sources:
        target = args.output/'source_snapshot'/source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    libraries = subprocess.check_output(['ldd', str(args.binary)], text=True)
    linked = [Path(line.split('=>', 1)[1].split()[0]) for line in libraries.splitlines() if '=>' in line]
    manifest = dict(binary_sha256=sha256(args.binary), oracle_sha256=sha256(args.oracle),
        source_hashes={str(p.relative_to(root)):sha256(p) for p in sources}, dynamic_libraries=libraries,
        library_hashes={str(p):sha256(p) for p in linked if p.is_file()},
        scope='One nonlinear step per arrived frame, declared chart/PCG ablations; not deadline-bounded or incrementally eliminated',
        modes=modes,
        streams={}, runs=[])
    def execute(tag, command):
        start = time.perf_counter()
        with (args.output/f'{tag}.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_900s'
        manifest['runs'].append(dict(tag=tag, command=command, exit_code=code,
            wall_seconds=time.perf_counter()-start))
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(manifest['runs'][-1]), flush=True)
        return code == 0
    completed = []
    for case in args.case:
        dataset_text, duration_text = case.split(':')
        dataset, duration = int(dataset_text), float(duration_text)
        stream = args.stream_root/f'dataset{dataset}'
        metadata = json.loads((stream/'metadata.json').read_text())
        events = stream/'events.jsonl'
        if sha256(events) != metadata['event_sha256']:
            raise ValueError('event hash mismatch')
        manifest['streams'][case] = metadata
        for mode in modes:
            tag = f'd{dataset}_{int(duration)}_{mode}'
            output = args.output/tag
            command = [str(args.binary), str(events), str(output), mode, str(duration), '1:2', '2:3', '3:4', '4:5']
            if not execute(tag, command):
                continue
            summary = validate_ledger(output)
            completed.append((dataset, duration, mode, output, summary))
            if mode.startswith('joint') and summary['final_components'] == 1:
                with events.open() as source:
                    package = prepare((json.loads(s) for s in source), typed_state(output/'final_estimates.csv'),
                                      [(1,2),(2,3),(3,4),(4,5)], duration)
                package_file = args.output/f'{tag}_completion.json'
                package_file.write_text(json.dumps(package, allow_nan=False)+'\n')
                oracle_output = args.output/f'{tag}_completion'
                execute(f'{tag}_completion', [str(args.oracle), str(package_file), str(oracle_output), 'central_lm', '100'])
    evaluate_results(completed, root, args)


def evaluate_results(completed, root, args):
    # Every estimator and completion process has exited before loading ground truth.
    results = []
    for dataset, duration, mode, output, summary in completed:
        manifest_data = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        audit = audit_dataset(manifest_data, args.raw)
        truth = load_ground_truth(manifest_data, audit.dataset_directory)
        labels = dict(manifest_data.evaluation.object_ground_truth_label_map)
        final = read_rows(output/'final_estimates.csv')
        row = dict(dataset=dataset, duration=duration, **summary,
            final_per_robot_alignment=evaluate_rows(final, truth, labels)[0])
        if summary['final_components'] == 1:
            row['final_global_alignment'] = evaluate_joint_rows(final, truth, labels)
        connected = [s['all_connected_elapsed_s'] for d,t,m,p,s in completed
            if d == dataset and t == duration and m.startswith('joint') and s['all_connected_elapsed_s'] is not None]
        if connected:
            # Compare exactly the same suffix, including when variants connect at different times.
            tail_start = max(connected)
            row['matched_tail_start_elapsed_s'] = tail_start
            causal = read_rows(output/'causal_estimates.csv')
            first = min(float(r['timestamp_s']) for r in causal)
            causal = [r for r in causal if float(r['timestamp_s'])-first >= tail_start]
            row['causal_connected_tail_per_robot_alignment'] = evaluate_rows(causal, truth, labels)[0]
            if mode.startswith('joint') and summary['all_connected_elapsed_s'] is not None:
                row['causal_connected_tail_global_alignment'] = evaluate_joint_rows(causal, truth, labels)
        completion = args.output/f'd{dataset}_{int(duration)}_{mode}_completion'
        if mode.startswith('joint') and (completion/'summary.json').is_file():
            row['completion'] = json.loads((completion/'summary.json').read_text())
            row['completion'].pop('trace', None)
            row['completion']['global_alignment'] = evaluate_joint_rows(read_rows(completion/'final_estimates.csv'), truth, labels)
        results.append(row)
    (args.output/'evaluation.json').write_text(json.dumps(dict(results=results,
        evaluation='Posthoc only. Main final/global metrics require one connected component; causal global metrics use only the stable fully-connected tail.',
        limitation='Independent mode has fewer equality constraints; not an equal-information distributed SOTA baseline.'), indent=2, allow_nan=False)+'\n')
    print(f'Completed {len(completed)} causal replays; all exits retained in manifest.', flush=True)


if __name__ == '__main__':
    main()
